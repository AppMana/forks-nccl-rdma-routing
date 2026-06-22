// SPDX-License-Identifier: BSD-3-Clause
// Clear-to-send (CTS) credit accounting for one rdmaroute RC connection.
//
// Why: rrIbIsend currently posts a SEND the moment NCCL hands it data. On
// pipelined traffic (DSV4 PP), the wire SEND outruns the receiver's posted
// recv WR, the usb4_rdma RC QP raises RNR, and -- because the driver does not
// honour infinite rnr_retry -- the send completes status=13 (RNR_RETRY_EXC),
// the QP errors, and the PP pipeline hangs. net_ib avoids this with a CTS FIFO:
// the receiver tells the sender how many recvs are posted, and the sender only
// sends with a credit in hand.
//
// This is the PURE credit accounting (the anti-RNR invariant). The RDMA-write
// that carries grants across the wire lives in ib_net.c and is not modelled
// here. Device-agnostic.
#ifndef RR_CTS_H
#define RR_CTS_H
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Sender side: credits == number of recv WRs the receiver has posted and
// announced that we have not yet consumed with a SEND.
typedef struct { uint32_t credits; } rr_cts_tx;

// Receiver side: recv WRs posted but not yet announced to the sender.
typedef struct { uint32_t pending_grant; } rr_cts_rx;

void     rr_cts_tx_init(rr_cts_tx* tx);
bool     rr_cts_can_send(const rr_cts_tx* tx);   // a credit is available
void     rr_cts_on_send(rr_cts_tx* tx);          // consume one credit (post SEND)
void     rr_cts_grant(rr_cts_tx* tx, uint32_t n);// receiver granted n credits

void     rr_cts_rx_init(rr_cts_rx* rx);
void     rr_cts_recv_posted(rr_cts_rx* rx);      // a recv WR was posted
uint32_t rr_cts_take_grants(rr_cts_rx* rx);      // drain pending grants to wire

#ifdef __cplusplus
}
#endif
#endif
