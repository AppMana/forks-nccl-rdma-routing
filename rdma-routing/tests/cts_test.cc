// SPDX-License-Identifier: BSD-3-Clause
// Contract for the clear-to-send credit accounting that rrIbIsend must obey so
// it never outruns the receiver's posted recvs (the DSV4 PP RNR/status=13 hang).
// Written test-first: RED against the no-CTS stub in cts.c (which always says
// "clear to send"), GREEN once the real credit logic lands. Pure logic, no RDMA.
#include <gtest/gtest.h>

extern "C" {
#include "cts.h"
}

// The core invariant: with no recv announced, the sender must NOT send. The
// current plugin sends anyway -> the SEND outruns the recv -> RNR. RED today.
TEST(Cts, FreshSenderHasNoCreditAndCannotSend) {
  rr_cts_tx tx;
  rr_cts_tx_init(&tx);
  EXPECT_FALSE(rr_cts_can_send(&tx));
}

// One grant authorises exactly one SEND, then backpressure returns.
TEST(Cts, OneGrantAuthorisesExactlyOneSend) {
  rr_cts_tx tx;
  rr_cts_tx_init(&tx);
  rr_cts_grant(&tx, 1);
  EXPECT_TRUE(rr_cts_can_send(&tx));
  rr_cts_on_send(&tx);
  EXPECT_FALSE(rr_cts_can_send(&tx));
}

// Credits accumulate across grants and are consumed one-per-send; the send
// after the last credit is blocked.
TEST(Cts, CreditsAccumulateAndAreConsumedOncePerSend) {
  rr_cts_tx tx;
  rr_cts_tx_init(&tx);
  rr_cts_grant(&tx, 2);
  rr_cts_grant(&tx, 2);                       // 4 outstanding
  for (int i = 0; i < 4; i++) {
    ASSERT_TRUE(rr_cts_can_send(&tx)) << "credit " << i;
    rr_cts_on_send(&tx);
  }
  EXPECT_FALSE(rr_cts_can_send(&tx));         // 5th send must block
}

// Receiver side: each posted recv becomes one grant to announce to the sender;
// draining hands them off and resets the pending count.
TEST(Cts, ReceiverProducesOneGrantPerPostedRecv) {
  rr_cts_rx rx;
  rr_cts_rx_init(&rx);
  rr_cts_recv_posted(&rx);
  rr_cts_recv_posted(&rx);
  rr_cts_recv_posted(&rx);
  EXPECT_EQ(rr_cts_take_grants(&rx), 3u);
  EXPECT_EQ(rr_cts_take_grants(&rx), 0u);     // drained: nothing left to announce
}

// End-to-end: the receiver's posted recvs, carried as grants, bound the sender
// EXACTLY -- it may send that many and no more. This is the anti-RNR guarantee.
TEST(Cts, ReceiverGrantsBoundSenderExactly) {
  rr_cts_rx rx;
  rr_cts_rx_init(&rx);
  rr_cts_tx tx;
  rr_cts_tx_init(&tx);

  for (int i = 0; i < 3; i++) rr_cts_recv_posted(&rx);
  rr_cts_grant(&tx, rr_cts_take_grants(&rx));  // 3 grants cross the wire

  int sent = 0;
  while (sent < 100 && rr_cts_can_send(&tx)) { // bounded so the stub can't hang
    rr_cts_on_send(&tx);
    sent++;
  }
  EXPECT_EQ(sent, 3);                           // never outruns posted recvs
}
