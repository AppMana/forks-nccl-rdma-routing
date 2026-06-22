// SPDX-License-Identifier: Apache-2.0
/*
 * Integration tests for the verbs RC backend (rrIb*) over a soft-RoCE (rxe)
 * device on loopback -- a REAL verbs device on one host, so the QP / post_send /
 * post_recv / poll_cq path runs in CI without a second node. This is the layer
 * that the unit/dispatch/netperf tests never exercised, where the DSV4
 * WC FLUSH lives.
 *
 * Setup (once per host):
 *   sudo modprobe rdma_rxe
 *   sudo rdma link add rxe_lo type rxe netdev lo
 *
 * build+run:
 *   cc -O2 -Wall -Isrc -o /tmp/ibrxe tests/ib_rxe_test.c src/ib_net.c \
 *      -lcriterion -libverbs -lpthread && NCCL_IB_HCA=rxe_lo NCCL_IB_GID_INDEX=0 /tmp/ibrxe
 *
 * We MODEL the bug first: the NCCL-like burst (post a stream of sends faster
 * than recvs are posted) must deliver every message intact. The current backend
 * has no pre-posted recv pool, so this is expected to FAIL until fixed.
 */
#include <criterion/criterion.h>
#include <arpa/inet.h>
#include <string.h>
#include <unistd.h>
#include "net_abi.h"

ncclDebugLogger_t rrLog = NULL;

/* establish a connected send/recv comm pair in-process over rxe_lo (loopback) */
static void establish(void **lc, void **sc, void **rc) {
	rrSockAddr la; memset(&la, 0, sizeof(la)); la.ip = inet_addr("127.0.0.1");
	cr_assert_eq(rrIbListen(0, &la, lc), ncclSuccess, "listen");
	cr_assert_not_null(*lc);
	*sc = NULL; *rc = NULL;
	ncclNetDeviceHandle_v9_t *d1 = NULL, *d2 = NULL;
	for (int i = 0; i < 200000 && !(*sc && *rc); i++) {
		if (!*sc) rrIbConnect(0, &la, sc);
		if (!*rc) rrIbAccept(*lc, rc);
		usleep(50);
	}
	cr_assert(*sc && *rc, "handshake (connect+accept) completed");
}

static void *reg(void *comm, void *buf, size_t n) {
	void *mh = NULL;
	cr_assert_eq(rrIbRegMr(comm, buf, n, NCCL_PTR_HOST, &mh), ncclSuccess, "regMr");
	return mh;
}
static int wait_done(void *req) {
	int done = 0, sz = 0;
	for (long i = 0; i < 50000000L && !done; i++) rrIbTest(req, &done, &sz);
	return done ? sz : -1;
}

Test(ib_rxe, basic_one_message) {
	void *lc, *sc, *rc; establish(&lc, &sc, &rc);
	static char tx[4096], rx[4096];
	for (int i = 0; i < 4096; i++) tx[i] = (char)(i * 7 + 1);
	memset(rx, 0, sizeof(rx));
	void *smh = reg(sc, tx, sizeof(tx)), *rmh = reg(rc, rx, sizeof(rx));
	void *rb = rx; size_t rs = sizeof(rx); int tg = 0;
	void *rreq = NULL, *sreq = NULL;
	rrIbIrecv(rc, 1, &rb, &rs, &tg, &rmh, &rreq);
	rrIbIsend(sc, tx, sizeof(tx), 0, smh, &sreq);
	cr_assert_eq(wait_done(sreq), (int)sizeof(tx), "send completed");
	cr_assert_eq(wait_done(rreq), (int)sizeof(tx), "recv completed full size");
	cr_assert_arr_eq(tx, rx, sizeof(tx), "payload intact");
}

/* MODEL THE BUG: NCCL streams many chunks; the receiver may not have a recv
 * posted when a send arrives. A correct RC backend pre-posts a recv pool so no
 * send RNR-stalls. Without it, the QP errors (WC FLUSH) under the burst. */
Test(ib_rxe, burst_no_preposted_recv) {
	void *lc, *sc, *rc; establish(&lc, &sc, &rc);
	enum { N = 16, SZ = 65536 };
	static char tx[N][SZ], rx[N][SZ];
	for (int m = 0; m < N; m++) { memset(tx[m], m + 1, SZ); memset(rx[m], 0, SZ); }
	void *smh = reg(sc, tx, sizeof(tx)), *rmh = reg(rc, rx, sizeof(rx));

	/* sender fires the whole burst first; recvs are posted lazily afterwards */
	void *sreq[N], *rreq[N];
	for (int m = 0; m < N; m++) { sreq[m] = NULL; rrIbIsend(sc, tx[m], SZ, m, smh, &sreq[m]); }
	for (int m = 0; m < N; m++) {
		void *rb = rx[m]; size_t rs = SZ; int tg = m; void *mh = rmh;
		rreq[m] = NULL; rrIbIrecv(rc, 1, &rb, &rs, &tg, &mh, &rreq[m]);
	}
	for (int m = 0; m < N; m++) {
		cr_expect_geq(wait_done(sreq[m]), 0, "send %d completed (no QP error)", m);
		cr_expect_eq(wait_done(rreq[m]), SZ, "recv %d completed full size", m);
		cr_expect_arr_eq(tx[m], rx[m], SZ, "burst message %d intact", m);
	}
}

/* MODEL THE CRASH: a send whose matching recv is posted only AFTER the RNR
 * retry window (rnr_retry=7 x min_rnr_timer ~= a few ms) expires. With no
 * pre-posted recv pool the QP exhausts RNR retries and goes to ERROR -> the
 * send WC is a real error and the late recv flushes (WC status=5), exactly the
 * DSV4 signature. A correct backend pre-posts recvs so the send never RNRs. */
Test(ib_rxe, recv_posted_after_rnr_window) {
	void *lc, *sc, *rc; establish(&lc, &sc, &rc);
	const size_t SZ = 65536;
	static char tx[65536], rx[65536];
	memset(tx, 0xab, SZ); memset(rx, 0, SZ);
	void *smh = reg(sc, tx, SZ), *rmh = reg(rc, rx, SZ);

	void *sreq = NULL; rrIbIsend(sc, tx, SZ, 0, smh, &sreq);
	usleep(50 * 1000); /* 50 ms >> the RNR retry window */
	void *rb = rx; size_t rs = SZ; int tg = 0; void *mh = rmh;
	void *rreq = NULL; rrIbIrecv(rc, 1, &rb, &rs, &tg, &mh, &rreq);

	cr_expect_geq(wait_done(sreq), 0, "send must complete even when recv is late (no RNR-exhaust QP error)");
	cr_expect_eq(wait_done(rreq), (int)SZ, "late recv must still deliver");
	cr_expect_eq(memcmp(tx, rx, SZ), 0, "payload intact across the RNR window");
}

/* MODEL THE CONCURRENCY: NCCL runs several QPs/comms per rank at once (2
 * channels x ring+tree). Open K comm pairs on the SAME device and move traffic
 * on all of them; every message must arrive intact. The single-pair tests never
 * exercised multiple simultaneous QPs/CQs (or the rrIbInit/g_dev sharing). */
Test(ib_rxe, concurrent_comms) {
	enum { K = 6, SZ = 65536 };
	void *lc[K], *sc[K], *rc[K], *smh[K], *rmh[K], *sreq[K], *rreq[K];
	static char tx[K][SZ], rx[K][SZ];
	for (int k = 0; k < K; k++) {
		establish(&lc[k], &sc[k], &rc[k]);
		memset(tx[k], k + 1, SZ); memset(rx[k], 0, SZ);
		smh[k] = reg(sc[k], tx[k], SZ); rmh[k] = reg(rc[k], rx[k], SZ);
	}
	/* post recv then send on every comm, interleaved, before draining any */
	for (int k = 0; k < K; k++) {
		void *rb = rx[k]; size_t rs = SZ; int tg = 0; void *mh = rmh[k];
		rreq[k] = NULL; rrIbIrecv(rc[k], 1, &rb, &rs, &tg, &mh, &rreq[k]);
		sreq[k] = NULL; rrIbIsend(sc[k], tx[k], SZ, 0, smh[k], &sreq[k]);
	}
	for (int k = 0; k < K; k++) {
		cr_expect_geq(wait_done(sreq[k]), 0, "comm %d send completed", k);
		cr_expect_eq(wait_done(rreq[k]), SZ, "comm %d recv full size", k);
		cr_expect_eq(memcmp(tx[k], rx[k], SZ), 0, "comm %d payload intact", k);
	}
}

/* a single large message (the DSV4 first RECV was ~40 MB) */
Test(ib_rxe, large_message) {
	void *lc, *sc, *rc; establish(&lc, &sc, &rc);
	const size_t SZ = 16u << 20; /* 16 MiB */
	char *tx = malloc(SZ), *rx = malloc(SZ);
	for (size_t i = 0; i < SZ; i++) tx[i] = (char)i;
	memset(rx, 0, SZ);
	void *smh = reg(sc, tx, SZ), *rmh = reg(rc, rx, SZ);
	void *rb = rx; size_t rs = SZ; int tg = 0;
	void *rreq = NULL, *sreq = NULL;
	rrIbIrecv(rc, 1, &rb, &rs, &tg, &rmh, &rreq);
	rrIbIsend(sc, tx, SZ, 0, smh, &sreq);
	cr_expect_geq(wait_done(sreq), 0, "large send completed");
	cr_expect_eq(wait_done(rreq), (int)SZ, "large recv completed full size");
	cr_expect_eq(memcmp(tx, rx, SZ), 0, "large payload intact");
	free(tx); free(rx);
}
