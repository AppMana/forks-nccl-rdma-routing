/*************************************************************************
 * ib_net.c - standard libibverbs RC backend for the rdmaroute NCCL net plugin.
 *
 * Device-agnostic RoCEv2: works on any RDMA device (soft-RoCE rxe, or a TB
 * rail) -- route.c picks WHICH device per connection by GID /64. Bootstrap
 * (device-GID + QP-info exchange) rides an out-of-band TCP socket; data moves
 * with IBV_WR_SEND_WITH_IMM over an RC QP with GID-based RoCEv2 addressing.
 *
 * Tag handling: the 32-bit NCCL tag rides the SEND work request's immediate
 * data; the receiver matches a completed recv WR to the posted irecv by tag
 * (single in-order stream per comm in this first cut).
 *************************************************************************/
#define _GNU_SOURCE
#include "net_abi.h"
#include "route.h"
#include "cts.h"

/* wr_id sentinel for the receiver's credit-grant RDMA_WRITEs, so rrIbTest can
 * skip them when matching data send/recv completions (it is never a req ptr). */
#define RR_GRANT_WRID 0xC0FFEE01ULL

#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#include <pthread.h>

#include <infiniband/verbs.h>

/* direct stderr diagnostics (RR_INFO goes to NCCL's logger which test harnesses
 * leave NULL); enable with NCCL_DEBUG set (INFO/TRACE/etc.) */
#define RR_DBG(fmt, ...) do { if (getenv("NCCL_DEBUG")) { \
  fprintf(stderr, "[ib] " fmt "\n", ##__VA_ARGS__); fflush(stderr); } } while (0)

/* ---- env-config ---- */
/* Read only the standard NCCL_* vars the tb-chain webhook already injects (no
 * mux-private env). NCCL_IB_HCA is a prefix/list (e.g. "usb4_rdma", optionally
 * "name:port" or comma-separated) -- we take the device-name token and match by
 * PREFIX below, exactly like NCCL's net_ib. */
static const char* ib_hca(void) {
  const char* e = getenv("NCCL_IB_HCA"); if (e && *e) return e;
  return "";  /* unset -> match all RDMA devices (device-agnostic) */
}
static int ib_gididx(void) {
  const char* e = getenv("NCCL_IB_GID_INDEX"); if (e && *e) return rr_atoi(e);
  return -1; /* -1 => scan (honours NCCL_IB_ADDR_FAMILY) */
}

#define RR_IB_PORT   1
#define RR_IB_MTU    IBV_MTU_4096
#define RR_IB_CQ_SZ  256
#define RR_IB_MAX_WR 256

/* ---- QP info exchanged over OOB TCP ---- */
typedef struct {
  uint32_t qp_num;
  uint32_t psn;
  uint16_t lid;       /* 0 for RoCE */
  uint8_t  gid[16];
  uint64_t credit_addr; /* sender's CTS inbox: receiver RDMA-writes cumulative grants here */
  uint32_t credit_rkey;
} rrIbQpInfo;

/* ---- per-process device context (lazily opened, shared) ---- */
typedef struct {
  struct ibv_context* ctx;
  struct ibv_pd*      pd;
  int                 portNum;
  int                 gidIndex;
  union ibv_gid       gid;
} rrIbDev;

/* All RDMA devices we route across (the rails; soft-RoCE rxe in the harness).
 * rdmaroute_select() picks one per connection by GID-/64 reachability. */
#define RR_MAX_DEV 16
static rrIbDev g_devs[RR_MAX_DEV];
static int      g_ndev   = 0;   /* rails (g_devs[0..g_nrail-1]) + fallback */
static int      g_nrail  = 0;   /* number of point-to-point rail devices */
static int      g_fallback = -1;/* reaches-all device index, or -1 (NCCL_RDMAROUTE_FALLBACK_HCA) */
static int      g_inited = 0;

/* NCCL_ROUTING_CONF_FILE explicit-reachability fallback (optional). */
static struct rdmaroute_conf g_conf;
static int                   g_have_conf = 0;

/* device-GID list exchanged over OOB before QP creation, so each side can run
 * rdmaroute_select(localGids, peerGids) and create its QP on the reachable dev. */
typedef struct {
  uint32_t ndev;
  uint8_t  gids[RR_MAX_DEV][16];
} rrIbDevList;

/* ---- request ---- */
enum { IREQ_FREE = 0, IREQ_SEND, IREQ_RECV };
struct rrIbComm_s;
typedef struct {
  int      used;
  int      done;
  uint32_t bytes;
  int      tag;
  uint64_t wr_id;
  struct rrIbComm_s* _comm;
} rrIbReq;

/* ---- comm ---- */
typedef struct rrIbComm_s {
  int                 oobFd;     /* out-of-band TCP for bootstrap */
  int                 sel;       /* g_devs[] index chosen by rdmaroute_select */
  struct ibv_cq*      cq;
  struct ibv_qp*      qp;
  rrIbReq            reqs[NCCL_NET_MAX_REQUESTS];
  int                 connected;
  /* non-blocking handshake: connect/accept must NOT block (NCCL contract), so
   * the blocking OOB connect + QP exchange runs in a thread and connect/accept
   * return NULL until hsState != 0. */
  pthread_t           hsTh;
  volatile int        hsState;   /* 0 running, 1 ok, -1 error */
  int                 isConn;    /* 1 connector, 0 acceptor */
  rrSockAddr         peer;      /* connector: where to connect the OOB */
  /* CTS credit flow control (anti-RNR). Sender (connector) gates SENDs on a
   * credit; receiver (acceptor) RDMA-writes its cumulative posted-recv count to
   * the sender's inbox. Cumulative + idempotent: a dropped grant self-heals on
   * the next recv. */
  rr_cts_tx           tx;            /* sender: credits available to SEND */
  volatile uint32_t*  creditInbox;   /* sender: receiver writes cumulative grants here */
  struct ibv_mr*      creditInboxMr;
  uint32_t            grantsApplied; /* sender: cumulative grants already fed to tx */
  uint32_t*           grantSrc;      /* receiver: local cumulative grant value to write */
  struct ibv_mr*      grantSrcMr;
  uint32_t            recvsTotal;    /* receiver: cumulative recvs posted */
  uint64_t            peerCreditAddr;/* receiver: sender's inbox addr */
  uint32_t            peerCreditRkey;
} rrIbComm;

#define RR_IB_PENDING 64
typedef struct {
  int oobFd;                          /* listening OOB TCP socket */
  rrIbComm* pending[RR_IB_PENDING]; /* accepted comms still handshaking */
  int npending;
} rrIbListenComm;

/* connector-side registry: NCCL re-calls connect() with the same handle until
 * it returns non-NULL, so resume the same in-progress comm keyed by peer ip:port. */
static struct { uint32_t ip; uint16_t port; rrIbComm* comm; } g_conn_pending[RR_IB_PENDING];
static pthread_mutex_t g_conn_lock = PTHREAD_MUTEX_INITIALIZER;

/* ---- MR wrapper ---- */
typedef struct { struct ibv_mr* mr; } rrIbMr;

/* ========================================================================
 * device open
 * ====================================================================== */
/* Scan a device's GID table for a routable GID (honouring NCCL_IB_GID_INDEX /
 * NCCL_IB_ADDR_FAMILY), preferring non-link-local eui64. Returns index or -1. */
static int pick_gid(struct ibv_context* ctx, int portNum) {
  int forced = ib_gididx();
  if (forced >= 0) return forced;
  const char* af = getenv("NCCL_IB_ADDR_FAMILY");
  int want_v4 = af && (strstr(af, "AF_INET") && !strstr(af, "INET6"));
  int best = -1, best_score = -1;
  union ibv_gid g;
  for (int i = 0; i < 64; i++) {
    if (ibv_query_gid(ctx, portNum, i, &g)) continue;
    int zero = 1; for (int k = 0; k < 16; k++) if (g.raw[k]) { zero = 0; break; }
    if (zero) continue;
    static const uint8_t v4p[12] = {0,0,0,0,0,0,0,0,0,0,0xff,0xff};
    int is_v4 = memcmp(g.raw, v4p, 12) == 0;
    if (is_v4 != want_v4) continue;                              /* match family */
    int score = (g.raw[0] == 0xfe && g.raw[1] == 0x80) ? 1 : 2;  /* prefer non-link-local */
    if (score > best_score) { best_score = score; best = i; }
  }
  return best;
}

/* open device d, scan its GID, append to g_devs[]; returns 0 ok / -1. */
static int rr_open_dev(struct ibv_device* d) {
  if (g_ndev >= RR_MAX_DEV) return -1;
  struct ibv_context* ctx = ibv_open_device(d);
  if (!ctx) return -1;
  struct ibv_pd* pd = ibv_alloc_pd(ctx);
  if (!pd) { ibv_close_device(ctx); return -1; }
  int pick = pick_gid(ctx, RR_IB_PORT);
  union ibv_gid gid;
  if (pick < 0 || ibv_query_gid(ctx, RR_IB_PORT, pick, &gid)) {
    ibv_dealloc_pd(pd); ibv_close_device(ctx); return -1;
  }
  g_devs[g_ndev].ctx = ctx; g_devs[g_ndev].pd = pd;
  g_devs[g_ndev].portNum = RR_IB_PORT; g_devs[g_ndev].gidIndex = pick; g_devs[g_ndev].gid = gid;
  RR_INFO("rdmaroute/ib: dev[%d] = %s gid_index=%d", g_ndev, ibv_get_device_name(d), pick);
  g_ndev++;
  return 0;
}

/* open all devices whose name starts with the first token of `spec`; count added. */
static int rr_open_matching(struct ibv_device** list, int num, const char* spec) {
  if (!spec) return 0;
  char want[64]; size_t wl = 0;
  for (const char* p = spec; *p && *p != ':' && *p != ',' && wl < sizeof(want)-1; p++) want[wl++] = *p;
  want[wl] = '\0';
  int added = 0;
  for (int i = 0; i < num && g_ndev < RR_MAX_DEV; i++)
    if (strncmp(ibv_get_device_name(list[i]), want, wl) == 0 && rr_open_dev(list[i]) == 0) added++;
  return added;
}

/* Open the rails (NCCL_IB_HCA) + the reaches-all fallback
 * (NCCL_RDMAROUTE_FALLBACK_HCA), and load the optional reachability conf. Per
 * connection, rdmaroute_select() picks a rail; if none reaches, the fallback. */
ncclResult_t rrIbInit(void) {
  if (g_inited) return ncclSuccess;

  const char* cf = getenv("NCCL_ROUTING_CONF_FILE");
  if (cf && *cf) {
    FILE* f = fopen(cf, "rb");
    if (f) {
      char buf[8192]; size_t n = fread(buf, 1, sizeof(buf) - 1, f); fclose(f);
      buf[n] = '\0';
      if (rdmaroute_conf_parse(buf, &g_conf) == 0) g_have_conf = 1;
      else RR_WARN("rdmaroute: malformed NCCL_ROUTING_CONF_FILE %s", cf);
    } else RR_WARN("rdmaroute: cannot open NCCL_ROUTING_CONF_FILE %s", cf);
  }

  int num = 0;
  struct ibv_device** list = ibv_get_device_list(&num);
  if (!list || num == 0) { RR_WARN("rdmaroute/ib: no RDMA devices"); return ncclSystemError; }

  rr_open_matching(list, num, ib_hca());                   /* rails (NCCL_IB_HCA) */
  g_nrail = g_ndev;
  const char* fb = getenv("NCCL_RDMAROUTE_FALLBACK_HCA");  /* reaches-all (e.g. rxe on the LAN) */
  if (fb && *fb) {
    int before = g_ndev;
    if (rr_open_matching(list, num, fb) > 0) g_fallback = before;
  }
  ibv_free_device_list(list);

  if (g_ndev == 0) { RR_WARN("rdmaroute/ib: no device matched NCCL_IB_HCA"); return ncclSystemError; }
  RR_INFO("rdmaroute/ib: %d rail(s) + fallback dev=%d", g_nrail, g_fallback);
  g_inited = 1;
  return ncclSuccess;
}

/* ========================================================================
 * OOB TCP helpers (blocking, only used during bootstrap)
 * ====================================================================== */
static int oobReadAll(int fd, void* buf, size_t len) {
  size_t off = 0;
  while (off < len) {
    ssize_t n = recv(fd, (char*)buf + off, len - off, 0);
    if (n > 0) { off += (size_t)n; continue; }
    if (n < 0 && errno == EINTR) continue;
    return -1;
  }
  return 0;
}
static int oobWriteAll(int fd, const void* buf, size_t len) {
  size_t off = 0;
  while (off < len) {
    ssize_t n = send(fd, (const char*)buf + off, len - off, MSG_NOSIGNAL);
    if (n > 0) { off += (size_t)n; continue; }
    if (n < 0 && errno == EINTR) continue;
    return -1;
  }
  return 0;
}

/* ========================================================================
 * QP creation + state transitions
 * ====================================================================== */
static struct ibv_qp* createQp(rrIbDev* d, struct ibv_cq* cq) {
  struct ibv_qp_init_attr ia; memset(&ia, 0, sizeof(ia));
  ia.send_cq = cq;
  ia.recv_cq = cq;
  ia.cap.max_send_wr  = RR_IB_MAX_WR;
  ia.cap.max_recv_wr  = RR_IB_MAX_WR;
  ia.cap.max_send_sge = 1;
  ia.cap.max_recv_sge = 1;
  ia.qp_type = IBV_QPT_RC;
  return ibv_create_qp(d->pd, &ia);
}

static int qpToInit(rrIbDev* d, struct ibv_qp* qp) {
  struct ibv_qp_attr a; memset(&a, 0, sizeof(a));
  a.qp_state = IBV_QPS_INIT;
  a.pkey_index = 0;
  a.port_num = d->portNum;
  a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
  return ibv_modify_qp(qp, &a,
    IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
}

static int qpToRtr(rrIbDev* d, struct ibv_qp* qp, const rrIbQpInfo* remote) {
  struct ibv_qp_attr a; memset(&a, 0, sizeof(a));
  a.qp_state = IBV_QPS_RTR;
  a.path_mtu = RR_IB_MTU;
  a.dest_qp_num = remote->qp_num;
  a.rq_psn = remote->psn;
  a.max_dest_rd_atomic = 1;
  a.min_rnr_timer = 12;

  a.ah_attr.is_global = 1;          /* RoCE: always GRH */
  a.ah_attr.dlid = 0;               /* RoCE has no LID */
  a.ah_attr.sl = 0;
  a.ah_attr.src_path_bits = 0;
  a.ah_attr.port_num = d->portNum;
  memcpy(&a.ah_attr.grh.dgid, remote->gid, 16);
  a.ah_attr.grh.flow_label = 0;
  a.ah_attr.grh.sgid_index = d->gidIndex;
  a.ah_attr.grh.hop_limit = 1;
  a.ah_attr.grh.traffic_class = 0;

  return ibv_modify_qp(qp, &a,
    IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
    IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
}

static int qpToRts(struct ibv_qp* qp, uint32_t myPsn) {
  struct ibv_qp_attr a; memset(&a, 0, sizeof(a));
  a.qp_state = IBV_QPS_RTS;
  a.timeout = 14;
  a.retry_cnt = 7;
  a.rnr_retry = 7;
  a.sq_psn = myPsn;
  a.max_rd_atomic = 1;
  return ibv_modify_qp(qp, &a,
    IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
    IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
}

/* Full RC bring-up over an already-connected OOB fd.
 * `isConnector` decides who writes first (avoid deadlock; both sides do
 * write-then-read or read-then-write symmetrically). */
static ncclResult_t ibBringUp(rrIbComm* c, int isConnector) {
  /* --- reachability: exchange device-GID lists, then rdmaroute_select() --- */
  /* exchange RAIL GIDs only (the fallback reaches everyone; it's not a rail). */
  rrIbDevList localList; memset(&localList, 0, sizeof(localList));
  localList.ndev = (uint32_t)g_nrail;
  for (int i = 0; i < g_nrail; i++) memcpy(localList.gids[i], &g_devs[i].gid, 16);

  rrIbDevList remoteList; memset(&remoteList, 0, sizeof(remoteList));
  if (isConnector) {
    if (oobWriteAll(c->oobFd, &localList, sizeof(localList)) < 0) return ncclSystemError;
    if (oobReadAll(c->oobFd, &remoteList, sizeof(remoteList)) < 0) return ncclSystemError;
  } else {
    if (oobReadAll(c->oobFd, &remoteList, sizeof(remoteList)) < 0) return ncclSystemError;
    if (oobWriteAll(c->oobFd, &localList, sizeof(localList)) < 0) return ncclSystemError;
  }
  int peer_ndev = (int)remoteList.ndev; if (peer_ndev > RR_MAX_DEV) peer_ndev = RR_MAX_DEV;
  int sel = rdmaroute_select((const uint8_t*)localList.gids, g_nrail,
                             (const uint8_t*)remoteList.gids, peer_ndev,
                             g_have_conf ? &g_conf : NULL);
  if (sel < 0) {
    /* No rail reaches the peer's subnet -> ride the reaches-all fallback (a
     * different path). Everything stays reachable; only a missing fallback is
     * a true ENETUNREACH. */
    if (g_fallback < 0) {
      RR_DBG("%s: no rail + no fallback -> unreachable", isConnector?"conn":"acc");
      return ncclSystemError;
    }
    sel = g_fallback;
    RR_DBG("%s: no rail; routing over fallback dev[%d]", isConnector?"conn":"acc", sel);
  }
  c->sel = sel;
  rrIbDev* d = &g_devs[sel];
  RR_DBG("%s: rdmaroute_select -> dev[%d] gid_index=%d", isConnector?"conn":"acc", sel, d->gidIndex);

  c->cq = ibv_create_cq(d->ctx, RR_IB_CQ_SZ, NULL, NULL, 0);
  if (!c->cq) { RR_DBG("%s: create_cq failed errno=%d", isConnector?"conn":"acc", errno); return ncclSystemError; }
  c->qp = createQp(d, c->cq);
  if (!c->qp) { RR_DBG("%s: create_qp failed errno=%d", isConnector?"conn":"acc", errno); return ncclSystemError; }
  if (qpToInit(d, c->qp)) { RR_DBG("%s: qpToInit failed errno=%d", isConnector?"conn":"acc", errno); return ncclSystemError; }

  /* CTS credit channel. Sender (connector) exposes a 4-byte inbox the receiver
   * RDMA-writes its cumulative posted-recv count into; receiver keeps a local
   * source word it writes from. */
  rrIbQpInfo local; memset(&local, 0, sizeof(local));
  if (isConnector) {
    rr_cts_tx_init(&c->tx);
    c->creditInbox = calloc(1, sizeof(uint32_t));
    if (!c->creditInbox) return ncclSystemError;
    c->creditInboxMr = ibv_reg_mr(d->pd, (void*)c->creditInbox, sizeof(uint32_t),
                                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!c->creditInboxMr) { RR_DBG("conn: reg credit inbox failed errno=%d", errno); return ncclSystemError; }
    local.credit_addr = (uint64_t)(uintptr_t)c->creditInbox;
    local.credit_rkey = c->creditInboxMr->rkey;
  } else {
    c->recvsTotal = 0;
    c->grantSrc = calloc(1, sizeof(uint32_t));
    if (!c->grantSrc) return ncclSystemError;
    c->grantSrcMr = ibv_reg_mr(d->pd, c->grantSrc, sizeof(uint32_t), IBV_ACCESS_LOCAL_WRITE);
    if (!c->grantSrcMr) { RR_DBG("acc: reg grant src failed errno=%d", errno); return ncclSystemError; }
  }

  local.qp_num = c->qp->qp_num;
  local.psn = (uint32_t)(rand() & 0xffffff);
  local.lid = 0;
  memcpy(local.gid, &d->gid, 16);

  rrIbQpInfo remote; memset(&remote, 0, sizeof(remote));
  if (isConnector) {
    if (oobWriteAll(c->oobFd, &local, sizeof(local)) < 0) { RR_DBG("conn: oob write qpinfo failed errno=%d", errno); return ncclSystemError; }
    if (oobReadAll(c->oobFd, &remote, sizeof(remote)) < 0) { RR_DBG("conn: oob read qpinfo failed errno=%d", errno); return ncclSystemError; }
  } else {
    if (oobReadAll(c->oobFd, &remote, sizeof(remote)) < 0) { RR_DBG("acc: oob read qpinfo failed errno=%d", errno); return ncclSystemError; }
    if (oobWriteAll(c->oobFd, &local, sizeof(local)) < 0) { RR_DBG("acc: oob write qpinfo failed errno=%d", errno); return ncclSystemError; }
  }

  RR_DBG("%s exchanged: local_qp=%u remote_qp=%u gididx=%d",
          isConnector ? "conn" : "acc", local.qp_num, remote.qp_num, d->gidIndex);
  if (!isConnector) {            /* receiver: remember where to write grants */
    c->peerCreditAddr = remote.credit_addr;
    c->peerCreditRkey = remote.credit_rkey;
  }
  if (qpToRtr(d, c->qp, &remote)) { RR_DBG("RTR failed errno=%d", errno); return ncclSystemError; }
  RR_DBG("%s RTR ok", isConnector ? "conn" : "acc");
  if (qpToRts(c->qp, local.psn)) { RR_DBG("RTS failed errno=%d", errno); return ncclSystemError; }
  RR_DBG("%s RTS ok -> connected", isConnector ? "conn" : "acc");
  c->connected = 1;
  return ncclSuccess;
}

/* Runs the blocking bring-up off the caller's thread. For the connector it also
 * does the (blocking) OOB connect here, so connect()/accept() never block. */
static void* ibHsThread(void* arg) {
  rrIbComm* c = (rrIbComm*)arg;
  if (c->isConn) {
    RR_DBG("conn: OOB connect -> %08x:%u", c->peer.ip, ntohs(c->peer.port));
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { c->hsState = -1; return NULL; }
    struct sockaddr_in sa; memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET; sa.sin_addr.s_addr = c->peer.ip; sa.sin_port = c->peer.port;
    if (connect(fd, (struct sockaddr*)&sa, sizeof(sa)) < 0) { RR_DBG("conn: OOB connect failed errno=%d", errno); close(fd); c->hsState = -1; return NULL; }
    int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    c->oobFd = fd;
    RR_DBG("conn: OOB connected");
  } else {
    RR_DBG("acc: OOB accepted, bringing up");
  }
  c->hsState = (ibBringUp(c, c->isConn) == ncclSuccess) ? 1 : -1;
  RR_DBG("%s: handshake done state=%d", c->isConn ? "conn" : "acc", c->hsState);
  return NULL;
}

static void ibFreeComm(rrIbComm* c) {
  if (!c) return;
  if (c->qp) ibv_destroy_qp(c->qp);
  if (c->cq) ibv_destroy_cq(c->cq);
  if (c->oobFd >= 0) close(c->oobFd);
  free(c);
}

/* ========================================================================
 * listen / connect / accept  (connect/accept are non-blocking: the OOB QP
 * exchange runs in ibHsThread; the calls return NULL until the handshake
 * finishes, matching NCCL's contract and avoiding the connect/accept deadlock.)
 * ====================================================================== */
ncclResult_t rrIbListen(int dev, rrSockAddr* addr, void** listenComm) {
  (void)dev;
  if (rrIbInit() != ncclSuccess) return ncclSystemError;
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return ncclSystemError;
  int one = 1; setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  struct sockaddr_in sa; memset(&sa, 0, sizeof(sa));
  sa.sin_family = AF_INET; sa.sin_addr.s_addr = INADDR_ANY; sa.sin_port = 0;
  if (bind(fd, (struct sockaddr*)&sa, sizeof(sa)) < 0) { close(fd); return ncclSystemError; }
  if (listen(fd, 64) < 0) { close(fd); return ncclSystemError; }
  socklen_t sl = sizeof(sa);
  if (getsockname(fd, (struct sockaddr*)&sa, &sl) < 0) { close(fd); return ncclSystemError; }

  /* non-blocking so accept() can poll */
  int fl = fcntl(fd, F_GETFL, 0); fcntl(fd, F_SETFL, fl | O_NONBLOCK);

  addr->port = sa.sin_port; /* mux fills addr->ip */
  rrIbListenComm* lc = (rrIbListenComm*)calloc(1, sizeof(rrIbListenComm));
  if (!lc) { close(fd); return ncclSystemError; }
  lc->oobFd = fd;
  *listenComm = lc;
  return ncclSuccess;
}

ncclResult_t rrIbConnect(int dev, rrSockAddr* addr, void** sendComm) {
  (void)dev;
  *sendComm = NULL;
  if (rrIbInit() != ncclSuccess) return ncclSystemError;

  pthread_mutex_lock(&g_conn_lock);
  rrIbComm* c = NULL; int slot = -1, free_slot = -1;
  for (int i = 0; i < RR_IB_PENDING; i++) {
    if (g_conn_pending[i].comm) {
      if (g_conn_pending[i].ip == addr->ip && g_conn_pending[i].port == addr->port) { c = g_conn_pending[i].comm; slot = i; break; }
    } else if (free_slot < 0) free_slot = i;
  }
  if (!c) {
    if (free_slot < 0) { pthread_mutex_unlock(&g_conn_lock); return ncclSystemError; }
    c = (rrIbComm*)calloc(1, sizeof(rrIbComm));
    if (!c) { pthread_mutex_unlock(&g_conn_lock); return ncclSystemError; }
    c->oobFd = -1; c->isConn = 1; c->peer = *addr; c->hsState = 0;
    g_conn_pending[free_slot].ip = addr->ip; g_conn_pending[free_slot].port = addr->port; g_conn_pending[free_slot].comm = c;
    pthread_create(&c->hsTh, NULL, ibHsThread, c);
    pthread_mutex_unlock(&g_conn_lock);
    return ncclSuccess; /* handshaking; NCCL retries */
  }
  int st = c->hsState;
  pthread_mutex_unlock(&g_conn_lock);
  if (st == 0) return ncclSuccess; /* still handshaking */
  pthread_join(c->hsTh, NULL);
  pthread_mutex_lock(&g_conn_lock); g_conn_pending[slot].comm = NULL; pthread_mutex_unlock(&g_conn_lock);
  if (st < 0) { ibFreeComm(c); return ncclSystemError; }
  *sendComm = c;
  return ncclSuccess;
}

ncclResult_t rrIbAccept(void* listenComm, void** recvComm) {
  rrIbListenComm* lc = (rrIbListenComm*)listenComm;
  *recvComm = NULL;

  /* drain any newly arrived OOB connections; each starts a handshake thread */
  for (;;) {
    struct sockaddr_in sa; socklen_t sl = sizeof(sa);
    int fd = accept(lc->oobFd, (struct sockaddr*)&sa, &sl);
    if (fd < 0) break;
    int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    if (lc->npending >= RR_IB_PENDING) { close(fd); break; }
    rrIbComm* c = (rrIbComm*)calloc(1, sizeof(rrIbComm));
    if (!c) { close(fd); break; }
    c->oobFd = fd; c->isConn = 0; c->hsState = 0;
    lc->pending[lc->npending++] = c;
    pthread_create(&c->hsTh, NULL, ibHsThread, c);
  }

  /* return the first accepted comm whose handshake finished */
  for (int i = 0; i < lc->npending; i++) {
    rrIbComm* c = lc->pending[i];
    if (c->hsState == 0) continue;
    pthread_join(c->hsTh, NULL);
    lc->pending[i] = lc->pending[--lc->npending];
    if (c->hsState < 0) { ibFreeComm(c); return ncclSystemError; }
    *recvComm = c;
    return ncclSuccess;
  }
  return ncclSuccess;
}

/* ========================================================================
 * memory registration
 * ====================================================================== */
ncclResult_t rrIbRegMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  (void)type;
  rrIbComm* c = (rrIbComm*)comm;   /* MR lives on the comm's selected device's PD */
  rrIbMr* m = (rrIbMr*)calloc(1, sizeof(rrIbMr));
  if (!m) return ncclSystemError;
  m->mr = ibv_reg_mr(g_devs[c->sel].pd, data, size,
                     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  if (!m->mr) {
    fprintf(stderr, "RR_POSTERR regMr ibv_reg_mr errno=%d (%s) size=%zu type=%d (memlock?)\n",
            errno, strerror(errno), size, type); fflush(stderr);
    free(m); return ncclSystemError;
  }
  *mhandle = m;
  return ncclSuccess;
}

ncclResult_t rrIbDeregMr(void* comm, void* mhandle) {
  (void)comm;
  rrIbMr* m = (rrIbMr*)mhandle;
  if (m) { if (m->mr) ibv_dereg_mr(m->mr); free(m); }
  return ncclSuccess;
}

/* ========================================================================
 * isend / irecv / iflush
 * ====================================================================== */
static rrIbReq* ibAllocReq(rrIbComm* c, int kind) {
  for (int i = 0; i < NCCL_NET_MAX_REQUESTS; i++)
    if (c->reqs[i].used == IREQ_FREE) {
      memset(&c->reqs[i], 0, sizeof(rrIbReq));
      c->reqs[i].used = kind;
      c->reqs[i].wr_id = (uint64_t)(uintptr_t)&c->reqs[i];
      c->reqs[i]._comm = c;
      return &c->reqs[i];
    }
  return NULL;
}

ncclResult_t rrIbIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void** request) {
  rrIbComm* c = (rrIbComm*)sendComm;
  /* CTS: fold in any grants the receiver RDMA-wrote (cumulative, monotonic),
   * then require a credit before posting -- otherwise the SEND can outrun the
   * receiver's recv WR and the usb4_rdma QP raises RNR (status=13). No credit ->
   * *request=NULL backpressure; NCCL re-calls isend (same as a full SQ). */
  if (c->creditInbox) {
    uint32_t cum = *c->creditInbox;
    if (cum != c->grantsApplied) {
      rr_cts_grant(&c->tx, cum - c->grantsApplied);
      c->grantsApplied = cum;
    }
    if (!rr_cts_can_send(&c->tx)) { *request = NULL; return ncclSuccess; }
  }
  rrIbMr* m = (rrIbMr*)mhandle;
  rrIbReq* r = ibAllocReq(c, IREQ_SEND);
  if (!r) { *request = NULL; return ncclSuccess; }
  r->tag = tag;
  r->bytes = (uint32_t)size;

  struct ibv_sge sge; memset(&sge, 0, sizeof(sge));
  sge.addr = (uintptr_t)data;
  sge.length = (uint32_t)size;
  sge.lkey = m->mr->lkey;

  struct ibv_send_wr wr; memset(&wr, 0, sizeof(wr));
  wr.wr_id = r->wr_id;
  wr.sg_list = &sge;
  wr.num_sge = (size > 0) ? 1 : 0;
  wr.opcode = IBV_WR_SEND_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.imm_data = (uint32_t)tag;

  struct ibv_send_wr* bad = NULL;
  int rc = ibv_post_send(c->qp, &wr, &bad);
  if (rc) {
    /* EAGAIN/ENOMEM = send queue momentarily full: BACKPRESSURE, not fatal.
     * Free the request and return *request=NULL so NCCL re-calls isend later
     * (exactly how net_ib handles a full SQ). Other errors are fatal. */
    if (rc == EAGAIN || rc == ENOMEM || errno == EAGAIN || errno == ENOMEM) {
      r->used = IREQ_FREE; *request = NULL; return ncclSuccess;
    }
    fprintf(stderr, "RR_POSTERR isend ibv_post_send rc=%d errno=%d (%s) qp=%u size=%zu tag=%d\n",
            rc, errno, strerror(errno), c->qp ? c->qp->qp_num : 0, size, tag); fflush(stderr);
    r->used = IREQ_FREE; return ncclSystemError;
  }
  rr_cts_on_send(&c->tx);          /* consume the credit this SEND used */
  *request = r;
  return ncclSuccess;
}

ncclResult_t rrIbIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** request) {
  (void)sizes; (void)tags;
  rrIbComm* c = (rrIbComm*)recvComm;
  if (n < 1) { *request = NULL; return ncclSuccess; }
  rrIbReq* r = ibAllocReq(c, IREQ_RECV);
  if (!r) { *request = NULL; return ncclSuccess; }
  r->tag = tags ? tags[0] : 0;

  rrIbMr* m = (rrIbMr*)mhandles[0];
  struct ibv_sge sge; memset(&sge, 0, sizeof(sge));
  sge.addr = (uintptr_t)data[0];
  sge.length = (uint32_t)sizes[0];
  sge.lkey = m->mr->lkey;

  struct ibv_recv_wr wr; memset(&wr, 0, sizeof(wr));
  wr.wr_id = r->wr_id;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  struct ibv_recv_wr* bad = NULL;
  int rc = ibv_post_recv(c->qp, &wr, &bad);
  if (rc) {
    /* recv queue full = backpressure; retry (NCCL re-calls irecv on NULL).
     * Other errors are fatal+logged. */
    if (rc == EAGAIN || rc == ENOMEM || errno == EAGAIN || errno == ENOMEM) {
      r->used = IREQ_FREE; *request = NULL; return ncclSuccess;
    }
    fprintf(stderr, "RR_POSTERR irecv ibv_post_recv rc=%d errno=%d (%s) qp=%u size=%zu\n",
            rc, errno, strerror(errno), c->qp ? c->qp->qp_num : 0, sizes[0]); fflush(stderr);
    r->used = IREQ_FREE; return ncclSystemError;
  }
  /* CTS: a recv WR is now posted -- announce the new cumulative recv count to
   * the sender's inbox so it may send. Cumulative + idempotent, so a dropped or
   * failed write self-heals on the next recv: best-effort, ignore post errors. */
  if (c->peerCreditRkey) {
    c->recvsTotal++;
    *c->grantSrc = c->recvsTotal;
    struct ibv_sge gs; memset(&gs, 0, sizeof(gs));
    gs.addr = (uintptr_t)c->grantSrc; gs.length = sizeof(uint32_t); gs.lkey = c->grantSrcMr->lkey;
    struct ibv_send_wr gw; memset(&gw, 0, sizeof(gw));
    gw.wr_id = RR_GRANT_WRID;
    gw.sg_list = &gs; gw.num_sge = 1;
    gw.opcode = IBV_WR_RDMA_WRITE;
    gw.send_flags = IBV_SEND_SIGNALED;
    gw.wr.rdma.remote_addr = c->peerCreditAddr;
    gw.wr.rdma.rkey = c->peerCreditRkey;
    struct ibv_send_wr* gbad = NULL;
    ibv_post_send(c->qp, &gw, &gbad);
  }
  *request = r;
  return ncclSuccess;
}

ncclResult_t rrIbIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  (void)recvComm; (void)n; (void)data; (void)sizes; (void)mhandles;
  *request = NULL; /* host memory, GDR off: nothing to flush */
  return ncclSuccess;
}

/* ========================================================================
 * test - poll the CQ, mark matching requests done
 * ====================================================================== */
ncclResult_t rrIbTest(void* request, int* done, int* sizes) {
  rrIbReq* r = (rrIbReq*)request;
  *done = 0;
  if (!r || r->used == IREQ_FREE) { *done = 1; if (sizes) *sizes = 0; return ncclSuccess; }

  /* Recover the comm: requests live inside the comm's array; the wr_id is
   * the request pointer. We need the CQ, so derive comm from the request's
   * position. To keep it simple, store the comm pointer in the request. */
  rrIbComm* c = r->_comm;
  struct ibv_wc wc[16];
  int ne = ibv_poll_cq(c->cq, 16, wc);
  if (ne < 0) {
    fprintf(stderr, "RR_POSTERR poll ibv_poll_cq ne=%d errno=%d (%s) qp=%u\n",
            ne, errno, strerror(errno), c->qp ? c->qp->qp_num : 0); fflush(stderr);
    return ncclSystemError;
  }
  { static int dbgn = 0; if (ne > 0 && dbgn < 12) { RR_DBG("poll ne=%d wc0.status=%d opcode=%d len=%u", ne, wc[0].status, wc[0].opcode, wc[0].byte_len); dbgn++; } }
  for (int i = 0; i < ne; i++) {
    if (wc[i].wr_id == RR_GRANT_WRID) continue;   /* CTS grant write: not a data req */
    rrIbReq* cr = (rrIbReq*)(uintptr_t)wc[i].wr_id;
    if (wc[i].status != IBV_WC_SUCCESS) {
      /* Capture the FIRST error WC in full -- status 5 (WR_FLUSH) is only the
       * cascade after an earlier real error (4 LOC_PROT, 10 REM_ACCESS, 12
       * RETRY_EXC, 13 RNR_RETRY_EXC, ...). Dump to stderr unconditionally so it
       * survives NCCL log filtering; this is the observation the test models. */
      static int g_first_wc_err = 1;
      const char* side = !cr ? "?" : (cr->used == IREQ_RECV ? "RECV" : "SEND");
      if (g_first_wc_err) {
        g_first_wc_err = 0;
        fprintf(stderr, "RR_WCERR FIRST status=%d vendor_err=%u opcode=%d byte_len=%u "
                "qp=%u wr_id=0x%lx side=%s imm=0x%x\n",
                wc[i].status, wc[i].vendor_err, wc[i].opcode, wc[i].byte_len,
                wc[i].qp_num, (unsigned long)wc[i].wr_id, side, wc[i].imm_data);
        fflush(stderr);
      }
      fprintf(stderr, "RR_WCERR status=%d vendor_err=%u opcode=%d qp=%u side=%s\n",
              wc[i].status, wc[i].vendor_err, wc[i].opcode, wc[i].qp_num, side);
      fflush(stderr);
      RR_WARN("rdmaroute/ib: WC error status=%d vendor_err=%u opcode=%d qp=%u side=%s",
               wc[i].status, wc[i].vendor_err, wc[i].opcode, wc[i].qp_num, side);
      if (cr) { cr->done = 1; cr->bytes = 0; }
      continue;
    }
    if (cr) {
      cr->done = 1;
      if (cr->used == IREQ_RECV) {
        cr->bytes = wc[i].byte_len;
        cr->tag = (int)wc[i].imm_data; /* tag carried in immediate */
      }
    }
  }

  if (r->done) {
    *done = 1;
    if (sizes) *sizes = (int)r->bytes;
    r->used = IREQ_FREE;
  }
  return ncclSuccess;
}

ncclResult_t rrIbCloseSend(void* sendComm) {
  rrIbComm* c = (rrIbComm*)sendComm;
  if (c) {
    if (c->qp) ibv_destroy_qp(c->qp);
    if (c->cq) ibv_destroy_cq(c->cq);
    /* CTS credit channel teardown (after the QP, so no in-flight access). */
    if (c->creditInboxMr) ibv_dereg_mr(c->creditInboxMr);
    if (c->creditInbox) free((void*)c->creditInbox);
    if (c->grantSrcMr) ibv_dereg_mr(c->grantSrcMr);
    if (c->grantSrc) free(c->grantSrc);
    if (c->oobFd >= 0) close(c->oobFd);
    free(c);
  }
  return ncclSuccess;
}
ncclResult_t rrIbCloseRecv(void* recvComm) { return rrIbCloseSend(recvComm); }
ncclResult_t rrIbCloseListen(void* listenComm) {
  rrIbListenComm* lc = (rrIbListenComm*)listenComm;
  if (lc) { if (lc->oobFd >= 0) close(lc->oobFd); free(lc); }
  return ncclSuccess;
}
