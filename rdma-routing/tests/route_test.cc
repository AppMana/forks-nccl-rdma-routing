// SPDX-License-Identifier: BSD-3-Clause
// Unit tests for the rdmaroute router: GID-/64 subnet sensing + the
// NCCL_ROUTING_CONF_FILE explicit-reachability fallback. Pure logic, no RDMA
// hardware. Device-agnostic (usb4_rdma / soft-RoCE are identical to it).
// Written test-first: the conf/select tests fail against the route.c stubs and
// pass once the real logic lands.
#include <gtest/gtest.h>
#include <cstring>
#include <cstdint>

extern "C" {
#include "route.h"
}

// ---------------- per-neighbour HCA selection (sensing) ----------------

// Build a RoCEv2 ULA GID: fd00:0000:0000:<link>::<eui>. The /64 subnet prefix
// (first 8 bytes) is per-LINK; the EUI (last 8) is per-node. Two HCAs on the
// same physical TB link therefore share the first 8 bytes.
static void mkgid(uint8_t* g, uint16_t link, uint64_t eui) {
  memset(g, 0, 16);
  g[0] = 0xfd;                 // ULA fd00::/8
  g[6] = (link >> 8) & 0xff;   // per-link discriminator inside the /64
  g[7] = link & 0xff;
  for (int i = 0; i < 8; i++) g[8 + i] = (uint8_t)(eui >> (8 * (7 - i)));
}

static void mk_linklocal(uint8_t* g, uint64_t eui) {
  memset(g, 0, 16);
  g[0] = 0xfe; g[1] = 0x80;    // fe80::/64 link-local (same prefix on every link)
  for (int i = 0; i < 8; i++) g[8 + i] = (uint8_t)(eui >> (8 * (7 - i)));
}

TEST(HcaSelect, PicksHcaOnSameLinkAsRemote) {
  uint8_t local[2 * 16];
  mkgid(local + 0,  0x0001, 0xaaaaaaaaaaaaaaaaULL); // HCA0 on link 1
  mkgid(local + 16, 0x0002, 0xbbbbbbbbbbbbbbbbULL); // HCA1 on link 2
  uint8_t remote[16];
  mkgid(remote, 0x0002, 0xccccccccccccccccULL);     // remote peer on link 2
  EXPECT_EQ(rdmaroute_pick_local_hca(local, 2, remote), 1);
}

TEST(HcaSelect, PicksFirstHcaWhenRemoteOnItsLink) {
  uint8_t local[2 * 16];
  mkgid(local + 0,  0x0001, 0xaaaaaaaaaaaaaaaaULL);
  mkgid(local + 16, 0x0002, 0xbbbbbbbbbbbbbbbbULL);
  uint8_t remote[16];
  mkgid(remote, 0x0001, 0xddddddddddddddddULL);     // remote on link 1
  EXPECT_EQ(rdmaroute_pick_local_hca(local, 2, remote), 0);
}

TEST(HcaSelect, NoSubnetMatchReturnsMinus1) {
  uint8_t local[16]; mkgid(local, 0x0001, 0xaaaaaaaaaaaaaaaaULL);
  uint8_t remote[16]; mkgid(remote, 0x0099, 0xccccccccccccccccULL);
  EXPECT_EQ(rdmaroute_pick_local_hca(local, 1, remote), -1);
}

TEST(HcaSelect, LinkLocalNeverMatches) {
  // fe80::/64 is identical on every link, so it must NOT be used to pick.
  uint8_t local[16];  mk_linklocal(local, 0xaaaaaaaaaaaaaaaaULL);
  uint8_t remote[16]; mk_linklocal(remote, 0xccccccccccccccccULL);
  EXPECT_EQ(rdmaroute_pick_local_hca(local, 1, remote), -1);
}

TEST(HcaSelect, AllZeroGidNeverMatches) {
  uint8_t local[16];  memset(local, 0, 16);
  uint8_t remote[16]; memset(remote, 0, 16);
  EXPECT_EQ(rdmaroute_pick_local_hca(local, 1, remote), -1);
}

// ---------- init-time map builder: match across two GID lists ----------

TEST(MatchHca, FindsSharedLinkPairMidChain) {
  // me: HCA0 on link1, HCA1 on link2.  neighbour: HCAs on link2 + link3.
  // shared link = 2 -> my HCA1.
  uint8_t local[2 * 16];
  mkgid(local + 0,  0x0001, 0xaaaaaaaaaaaaaaaaULL);
  mkgid(local + 16, 0x0002, 0xbbbbbbbbbbbbbbbbULL);
  uint8_t remote[2 * 16];
  mkgid(remote + 0,  0x0002, 0xccccccccccccccccULL);
  mkgid(remote + 16, 0x0003, 0xddddddddddddddddULL);
  EXPECT_EQ(rdmaroute_match_local_hca(local, 2, remote, 2), 1);
}

TEST(MatchHca, EndpointSingleHcaPairs) {
  uint8_t local[16];  mkgid(local, 0x0005, 0xaaaaaaaaaaaaaaaaULL);
  uint8_t remote[2 * 16];
  mkgid(remote + 0,  0x0005, 0xccccccccccccccccULL);
  mkgid(remote + 16, 0x0006, 0xddddddddddddddddULL);
  EXPECT_EQ(rdmaroute_match_local_hca(local, 1, remote, 2), 0);
}

TEST(MatchHca, NoSharedSubnetReturnsMinus1) {
  uint8_t local[2 * 16];
  mkgid(local + 0,  0x0001, 0xaaaaaaaaaaaaaaaaULL);
  mkgid(local + 16, 0x0002, 0xbbbbbbbbbbbbbbbbULL);
  uint8_t remote[16]; mkgid(remote, 0x0099, 0xccccccccccccccccULL);
  EXPECT_EQ(rdmaroute_match_local_hca(local, 2, remote, 1), -1);
}

TEST(MatchHca, LinkLocalDoesNotPair) {
  uint8_t local[16];  mk_linklocal(local, 0xaaaaaaaaaaaaaaaaULL);
  uint8_t remote[16]; mk_linklocal(remote, 0xccccccccccccccccULL);
  EXPECT_EQ(rdmaroute_match_local_hca(local, 1, remote, 1), -1);
}

// ---------- NCCL_ROUTING_CONF_FILE: parse (inputs) ----------
// mkgid(_, link, 0) yields a GID whose first 8 bytes ARE the fd00:0:0:<link>::/64
// prefix, so it doubles as the expected parsed prefix.

TEST(Conf, ParsesBasicRule) {
  struct rdmaroute_conf c;
  ASSERT_EQ(rdmaroute_conf_parse("reach fd00:0:0:1::/64 fd00:0:0:9::/64\n", &c), 0);
  ASSERT_EQ(c.n, 1);
  EXPECT_EQ(c.rules[0].local_plen, 64);
  EXPECT_EQ(c.rules[0].remote_plen, 64);
  uint8_t lp[16], rp[16];
  mkgid(lp, 0x0001, 0); mkgid(rp, 0x0009, 0);
  EXPECT_EQ(memcmp(c.rules[0].local_prefix, lp, 8), 0);
  EXPECT_EQ(memcmp(c.rules[0].remote_prefix, rp, 8), 0);
}

TEST(Conf, ExpandsMultipleRemotesSkipsCommentsAndBlanks) {
  struct rdmaroute_conf c;
  const char* txt =
      "# explicit reachability\n"
      "\n"
      "reach fd00:0:0:1::/64 fd00:0:0:9::/64 fd00:0:0:a::/64\n";
  ASSERT_EQ(rdmaroute_conf_parse(txt, &c), 0);
  EXPECT_EQ(c.n, 2);  // one rule per remote, sharing the line's local subnet
}

TEST(Conf, MalformedReturnsError) {
  struct rdmaroute_conf c;
  EXPECT_LT(rdmaroute_conf_parse("reach not-an-address\n", &c), 0);
}

// ---------- rdmaroute_select: outputs (sense -> conf -> fallback) ----------

TEST(Select, SensesSameSubnetWithoutConf) {
  uint8_t local[2 * 16];
  mkgid(local + 0,  0x0001, 0xaaaaaaaaaaaaaaaaULL);
  mkgid(local + 16, 0x0002, 0xbbbbbbbbbbbbbbbbULL);
  uint8_t remote[16]; mkgid(remote, 0x0002, 0xccccccccccccccccULL);
  EXPECT_EQ(rdmaroute_select(local, 2, remote, 1, nullptr), 1);
}

TEST(Select, FallbackWhenNoSubnetAndNoConf) {
  uint8_t local[16]; mkgid(local, 0x0001, 0xaaaaaaaaaaaaaaaaULL);
  uint8_t remote[16]; mkgid(remote, 0x0009, 0xccccccccccccccccULL);
  EXPECT_EQ(rdmaroute_select(local, 1, remote, 1, nullptr), -1);  // -> reaches-all fallback
}

TEST(Select, ConfExtendsReachabilityBeyondSubnet) {
  // dev0 on subnet 1; peer on subnet 9 (no /64 match). conf says 1 reaches 9.
  struct rdmaroute_conf c;
  ASSERT_EQ(rdmaroute_conf_parse("reach fd00:0:0:1::/64 fd00:0:0:9::/64\n", &c), 0);
  uint8_t local[16]; mkgid(local, 0x0001, 0xaaaaaaaaaaaaaaaaULL);
  uint8_t remote[16]; mkgid(remote, 0x0009, 0xccccccccccccccccULL);
  EXPECT_EQ(rdmaroute_select(local, 1, remote, 1, &c), 0);
}

TEST(Select, ConfPicksTheCorrectLocalDevice) {
  // dev0 subnet1, dev1 subnet2; peer subnet9; conf permits subnet2 -> subnet9.
  struct rdmaroute_conf c;
  ASSERT_EQ(rdmaroute_conf_parse("reach fd00:0:0:2::/64 fd00:0:0:9::/64\n", &c), 0);
  uint8_t local[2 * 16];
  mkgid(local + 0,  0x0001, 0xaaaaaaaaaaaaaaaaULL);
  mkgid(local + 16, 0x0002, 0xbbbbbbbbbbbbbbbbULL);
  uint8_t remote[16]; mkgid(remote, 0x0009, 0xccccccccccccccccULL);
  EXPECT_EQ(rdmaroute_select(local, 2, remote, 1, &c), 1);
}
