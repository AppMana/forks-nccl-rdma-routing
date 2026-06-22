// SPDX-License-Identifier: BSD-3-Clause
// Does NCCL 2.30.7's net_ib subnet matching (gidSameSubnet) correctly
// distinguish our per-link IPv6 /64 GIDs (Thunderbolt rail chain)? This is the
// foundation ncclIbFindDevBySubnet ("override dev X with dev Y") stands on. If
// this is green, the matching is correct and the remaining gap is config
// (NCCL_IB_SUBNET_AWARE_ROUTING / NCCL_IB_MERGE_NICS), not a logic bug; if red,
// it pinpoints the one thing to fix. Exercises the REAL function (subnet_match.h
// is the single definition net_ib/connect.cc also includes).
#include <gtest/gtest.h>
#include <string.h>
#include <infiniband/verbs.h>   // union ibv_gid (subnet_match.h leaves this to the includer)
#include "subnet_match.h"

// IPv6 GID: /64 prefix in raw[0..7], interface id in raw[8..15].
static union ibv_gid gid6(const uint8_t prefix8[8], uint64_t ifaceId) {
  union ibv_gid g;
  memset(&g, 0, sizeof(g));
  memcpy(g.raw, prefix8, 8);
  for (int i = 0; i < 8; i++) g.raw[8 + i] = (uint8_t)(ifaceId >> (8 * (7 - i)));
  return g;
}
// Our per-link ULA: 0xfd + a per-link hash discriminator in the /64.
static union ibv_gid railGid(uint8_t linkTag, uint64_t ifaceId) {
  uint8_t p[8] = {0xfd, linkTag, linkTag, linkTag, 0x00, 0x00, 0x00, 0x00};
  return gid6(p, ifaceId);
}

// The ULA prefix must classify as IPv6 (not mis-detected as v4-mapped), or the
// /64 comparison path never runs.
TEST(Subnet, UlaClassifiesAsIpv6) {
  union ibv_gid ula = railGid(0xA1, 1);
  EXPECT_EQ(getGidAddrFamily(&ula), AF_INET6);
}

// Both ends of the SAME Thunderbolt link share the /64 -> reachable.
TEST(Subnet, SameLinkBothEndsShareThe64) {
  union ibv_gid localA = railGid(0xA1, 1);   // my rail to neighbour A (::1)
  union ibv_gid peerA  = railGid(0xA1, 2);   // neighbour A's rail   (::2, same /64)
  EXPECT_TRUE(gidSameSubnet(&localA, &peerA, 64));
}

// A mid-chain node's two rails are on DIFFERENT /64s; each reaches exactly its
// own cabled neighbour, not the other. This is the property selection needs to
// pick the right rail per peer.
TEST(Subnet, MidChainRailsReachOnlyTheirOwnNeighbour) {
  union ibv_gid railUp   = railGid(0xA1, 1);   // upstream link
  union ibv_gid railDown = railGid(0xB2, 1);   // downstream link (different /64)
  union ibv_gid peerUp   = railGid(0xA1, 2);
  union ibv_gid peerDown = railGid(0xB2, 2);

  EXPECT_TRUE (gidSameSubnet(&railUp,   &peerUp,   64));
  EXPECT_FALSE(gidSameSubnet(&railUp,   &peerDown, 64));
  EXPECT_TRUE (gidSameSubnet(&railDown, &peerDown, 64));
  EXPECT_FALSE(gidSameSubnet(&railDown, &peerUp,   64));
}

// --- NCCL_IB_SUBNET_PREFER_HCA: prefer the rail over the reaches-all fallback ---
// The flat fallback (rxe_lan) reaches every peer, so without a preference an
// adjacent peer wrongly stays on it. The prefer-list picks the rail when it
// reaches, and only falls back when no rail does.
TEST(PreferHca, AdjacentPicksRailOverFallback) {
  const char* names[] = {"usb4_rdma5", "usb4_rdma15", "rxe_lan"};
  int reach[] = {1, 0, 1};  // my rail-to-this-peer reaches; other rail doesn't; rxe reaches (flat)
  EXPECT_EQ(ibPreferReachableDev(names, reach, 3, "usb4_rdma"), 0);  // pick the rail, not rxe
}
TEST(PreferHca, NonAdjacentFallsThroughToCaller) {
  const char* names[] = {"usb4_rdma5", "usb4_rdma15", "rxe_lan"};
  int reach[] = {0, 0, 1};  // no rail reaches; only the flat fallback
  EXPECT_EQ(ibPreferReachableDev(names, reach, 3, "usb4_rdma"), -1);  // no preferred match -> caller's fallback
}
TEST(PreferHca, SecondPriorityWhenFirstUnreachable) {
  const char* names[] = {"usb4_rdma5", "usb4_rdma15", "rxe_lan"};
  int reach[] = {0, 0, 1};
  EXPECT_EQ(ibPreferReachableDev(names, reach, 3, "usb4_rdma,rxe_lan"), 2);  // rxe is 2nd in the list
}
TEST(PreferHca, EmptyOrNullListNoPreference) {
  const char* names[] = {"usb4_rdma5", "rxe_lan"};
  int reach[] = {1, 1};
  EXPECT_EQ(ibPreferReachableDev(names, reach, 2, ""), -1);
  EXPECT_EQ(ibPreferReachableDev(names, reach, 2, nullptr), -1);
}
TEST(PreferHca, FirstReachableOfHighestPriorityPrefix) {
  const char* names[] = {"rxe_lan", "usb4_rdma5", "usb4_rdma15"};  // fallback listed first by index
  int reach[] = {1, 0, 1};  // rxe reaches; rail15 reaches
  EXPECT_EQ(ibPreferReachableDev(names, reach, 3, "usb4_rdma,rxe_lan"), 2);  // rail beats rxe despite index order
}

// The exact config syntax: usb4_rdma\d*,rxe_lan\d* (raw string => the literal
// backslash reaches the matcher, which translates \d -> [0-9]).
TEST(PreferHca, RegexDigitSyntaxFromConfig) {
  const char* names[] = {"usb4_rdma5", "usb4_rdma15", "rxe_lan"};
  int reach[] = {1, 0, 1};
  EXPECT_EQ(ibPreferReachableDev(names, reach, 3, R"(usb4_rdma\d*,rxe_lan\d*)"), 0);
}
TEST(PreferHca, RegexDigitNonAdjacentTakesFallback) {
  const char* names[] = {"usb4_rdma5", "usb4_rdma15", "rxe_lan"};
  int reach[] = {0, 0, 1};
  EXPECT_EQ(ibPreferReachableDev(names, reach, 3, R"(usb4_rdma\d*,rxe_lan\d*)"), 2);  // rxe (2nd pattern)
}
TEST(PreferHca, DevPathPrefixStripped) {
  const char* names[] = {"usb4_rdma5", "rxe_lan"};
  int reach[] = {1, 1};
  EXPECT_EQ(ibPreferReachableDev(names, reach, 2, R"(/dev/usb4_rdma\d*)"), 0);
}

// Two links that share the first 32 bits but differ in bits 33-64 must STILL be
// different subnets -- i.e. the IPv6 path compares the full /64, not an IPv4
// /<=32 prefix. (Our FNV-hash discriminator can collide in the high bytes.)
TEST(Subnet, FullSixtyFourBitsCompared) {
  uint8_t pA[8] = {0xfd, 0xAB, 0xCD, 0xEF, 0x11, 0x11, 0x11, 0x11};
  uint8_t pB[8] = {0xfd, 0xAB, 0xCD, 0xEF, 0x22, 0x22, 0x22, 0x22}; // same first 32 bits
  union ibv_gid a = gid6(pA, 1), b = gid6(pB, 2);
  EXPECT_FALSE(gidSameSubnet(&a, &b, 64));   // differ in bits 33-64 -> different /64
  EXPECT_FALSE(gidSameSubnet(&a, &b, 24));   // and the IPv4 prefixLen must be ignored
}
