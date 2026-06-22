// SPDX-License-Identifier: BSD-3-Clause
// Clear-to-send credit accounting (pure). The sender holds one credit per recv
// WR the receiver has posted and announced; it may post a SEND only while a
// credit is in hand, so the wire SEND can never outrun the receiver's recv and
// raise RNR (status=13 -- the DSV4 PP hang). The RDMA-write that carries grants
// across the wire lives in ib_net.c. Contract: tests/cts_test.cc.
#include "cts.h"

void rr_cts_tx_init(rr_cts_tx* tx) { tx->credits = 0; }

bool rr_cts_can_send(const rr_cts_tx* tx) { return tx->credits > 0; }

void rr_cts_on_send(rr_cts_tx* tx) {
	if (tx->credits) tx->credits--;        // consume; never underflow
}

void rr_cts_grant(rr_cts_tx* tx, uint32_t n) { tx->credits += n; }

void rr_cts_rx_init(rr_cts_rx* rx) { rx->pending_grant = 0; }

void rr_cts_recv_posted(rr_cts_rx* rx) { rx->pending_grant++; }

uint32_t rr_cts_take_grants(rr_cts_rx* rx) {
	uint32_t n = rx->pending_grant;
	rx->pending_grant = 0;
	return n;
}
