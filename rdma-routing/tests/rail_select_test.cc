// SPDX-License-Identifier: BSD-3-Clause
// Self-inferring IB device selection (no external file, no DNS, no route lookup):
//
//   a) NCCL_NODE_TO_NODE_TOPO_FILE (node_topo_test.cc) stays the optional
//      override -- unchanged, tested there.
//   b) RAIL match: a local device matching a non-last prefer-list pattern
//      (usb4_rdma\d*) is selected iff ANY of the peer's advertised GIDs shares
//      its /64 (per-link ULA -- both cabled ends share it, deterministic).
//   c) FALLBACK: the device matching the LAST prefer-list pattern (rxe_lan\d*)
//      is selected UNCONDITIONALLY when no rail matched. No subnet match: the
//      two nodes' rxe_lan prefixes may legitimately differ (SLAAC races GID
//      index ordering at boot), which is exactly the bug that hung the
//      ring-closing hop of the 10-rank all-reduce.
//   d) Neither -> -1, caller keeps NCCL's default device.
//
// Plus the wire layer that feeds b): each device advertises ALL its valid GIDs
// (capped), not just the one at the selected GID index, and a peer sending the
// OLD single-GID metadata must still work (mixed old/new fleet).
#include <gtest/gtest.h>
#include <string.h>
#include <infiniband/verbs.h>   // union ibv_gid (subnet_match.h leaves this to the includer)
#include "subnet_match.h"

namespace {

constexpr const char* kPreferList = R"(usb4_rdma\d*,rxe_lan\d*)";

// IPv6 GID: /64 prefix tagged by linkTag, interface id in raw[8..15].
void mkGid(uint8_t g[16], uint8_t linkTag, uint64_t iface) {
  memset(g, 0, 16);
  g[0] = 0xfd; g[1] = linkTag; g[2] = linkTag; g[3] = linkTag;
  for (int i = 0; i < 8; i++) g[8 + i] = (uint8_t)(iface >> (8 * (7 - i)));
}
// IPv4-mapped GID ::ffff:a.b.c.d
void mkV4Gid(uint8_t g[16], uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
  memset(g, 0, 16);
  g[10] = 0xff; g[11] = 0xff;
  g[12] = a; g[13] = b; g[14] = c; g[15] = d;
}
void mkLinkLocal(uint8_t g[16], uint64_t iface) {
  memset(g, 0, 16);
  g[0] = 0xfe; g[1] = 0x80;
  for (int i = 0; i < 8; i++) g[8 + i] = (uint8_t)(iface >> (8 * (7 - i)));
}

// A mid-chain node: rail to neighbour A (link a1), rail to neighbour C (link
// c3), and the flat fallback rxe_lan whose /64 differs from every peer's.
struct Node {
  const char* names[3] = {"usb4_rdma5", "usb4_rdma15", "rxe_lan"};
  uint8_t gids[3][16];
  int gidDev[3] = {0, 1, 2};
  Node() {
    mkGid(gids[0], 0xa1, 1);   // rail /64 shared with neighbour A
    mkGid(gids[1], 0xc3, 1);   // rail /64 shared with neighbour C
    mkGid(gids[2], 0x5a, 1);   // rxe_lan, local-only /64 (SLAAC-raced in reality)
  }
};

// ---------------------------------------------------------------------------
// prefer-list split: all patterns but the LAST are rails; the last is the
// unconditional fallback.
// ---------------------------------------------------------------------------
TEST(PreferSplit, RailsAreAllButLastPattern) {
  EXPECT_TRUE(ibDevIsRail("usb4_rdma5", kPreferList));
  EXPECT_TRUE(ibDevIsRail("usb4_rdma15", kPreferList));
  EXPECT_FALSE(ibDevIsRail("rxe_lan", kPreferList));
}
TEST(PreferSplit, FallbackIsTheLastPattern) {
  EXPECT_TRUE(ibDevIsFallback("rxe_lan", kPreferList));
  EXPECT_TRUE(ibDevIsFallback("rxe_lan0", kPreferList));
  EXPECT_FALSE(ibDevIsFallback("usb4_rdma5", kPreferList));
}
TEST(PreferSplit, UnrelatedDeviceIsNeither) {
  EXPECT_FALSE(ibDevIsRail("mlx5_0", kPreferList));
  EXPECT_FALSE(ibDevIsFallback("mlx5_0", kPreferList));
}

// ---------------------------------------------------------------------------
// device selection
// ---------------------------------------------------------------------------
TEST(RailSelect, AdjacentPeerPicksTheSharedRail) {
  Node local;
  uint8_t peer[2][16];
  mkGid(peer[0], 0xa1, 2);          // neighbour A's end of link a1
  mkGid(peer[1], 0x77, 2);          // neighbour A's rxe_lan (foreign /64)
  int how = -1;
  int dev = ibRailSelectDev(local.names, 3, local.gids, local.gidDev, 3,
                            peer, 2, 24, kPreferList, &how);
  EXPECT_EQ(dev, 0) << "peer shares rail0's /64 -> that rail, not rxe";
  EXPECT_EQ(how, IB_RAIL_SELECT_RAIL);
}

TEST(RailSelect, TwoRailsPickTheCorrectOne) {
  Node local;
  uint8_t peer[1][16];
  mkGid(peer[0], 0xc3, 2);          // neighbour C's end of link c3
  int how = -1;
  int dev = ibRailSelectDev(local.names, 3, local.gids, local.gidDev, 3,
                            peer, 1, 24, kPreferList, &how);
  EXPECT_EQ(dev, 1) << "must pick the rail on the peer's link, not the first rail";
  EXPECT_EQ(how, IB_RAIL_SELECT_RAIL);
}

// THE bug's shape: ring-closing hop. Peer is non-adjacent (none of its GIDs
// shares any local rail /64) and its rxe_lan is on a DIFFERENT /64 than ours.
// Old code required a subnet match even for the fallback -> found nothing ->
// kept a usb4 rail toward a non-neighbor -> the all-reduce hung.
TEST(RailSelect, NonAdjacentPeerPicksFallbackUnconditionally) {
  Node local;
  uint8_t peer[2][16];
  mkGid(peer[0], 0xd4, 2);          // peer's rail on a link we're not on
  mkGid(peer[1], 0x99, 2);          // peer's rxe_lan: DIFFERENT /64 than local rxe (0x5a)
  int how = -1;
  int dev = ibRailSelectDev(local.names, 3, local.gids, local.gidDev, 3,
                            peer, 2, 24, kPreferList, &how);
  EXPECT_EQ(dev, 2) << "no rail matched -> rxe_lan, even though the /64s differ";
  EXPECT_EQ(how, IB_RAIL_SELECT_FALLBACK);
}

// A peer that advertises several GIDs (v4-mapped + link ULA): ANY match selects
// the rail. This is why multi-GID advertisement matters.
TEST(RailSelect, RailChosenByAnyOfMultiplePeerGids) {
  Node local;
  uint8_t peer[3][16];
  mkV4Gid(peer[0], 10, 2, 0, 31);   // v4-mapped (rxe_lan's IPv4 face)
  mkGid(peer[1], 0x99, 2);          // foreign /64
  mkGid(peer[2], 0xa1, 2);          // the rail ULA, advertised last
  int how = -1;
  int dev = ibRailSelectDev(local.names, 3, local.gids, local.gidDev, 3,
                            peer, 3, 24, kPreferList, &how);
  EXPECT_EQ(dev, 0);
  EXPECT_EQ(how, IB_RAIL_SELECT_RAIL);
}

// Malformed / empty peer advertisement must never strand the connection: no
// valid peer GID -> no rail can match -> unconditional fallback.
TEST(RailSelect, NoPeerGidsStillFallsBack) {
  Node local;
  int how = -1;
  int dev = ibRailSelectDev(local.names, 3, local.gids, local.gidDev, 3,
                            NULL, 0, 24, kPreferList, &how);
  EXPECT_EQ(dev, 2);
  EXPECT_EQ(how, IB_RAIL_SELECT_FALLBACK);
}
TEST(RailSelect, InvalidPeerGidsIgnoredForRailMatch) {
  Node local;
  uint8_t peer[2][16];
  memset(peer[0], 0, 16);           // zero GID: invalid
  mkLinkLocal(peer[1], 2);          // link-local: excluded
  int how = -1;
  int dev = ibRailSelectDev(local.names, 3, local.gids, local.gidDev, 3,
                            peer, 2, 24, kPreferList, &how);
  EXPECT_EQ(dev, 2);
  EXPECT_EQ(how, IB_RAIL_SELECT_FALLBACK);
}

// No fallback device exists and no rail matched -> -1: caller keeps NCCL's
// default; we do not invent a new failure mode.
TEST(RailSelect, NoFallbackDeviceReturnsMinusOne) {
  const char* names[2] = {"usb4_rdma5", "usb4_rdma15"};
  uint8_t gids[2][16];
  mkGid(gids[0], 0xa1, 1);
  mkGid(gids[1], 0xc3, 1);
  int gidDev[2] = {0, 1};
  uint8_t peer[1][16];
  mkGid(peer[0], 0xd4, 2);
  int how = -1;
  int dev = ibRailSelectDev(names, 2, gids, gidDev, 2, peer, 1, 24, kPreferList, &how);
  EXPECT_EQ(dev, -1);
}

// Empty / NULL prefer list -> -1 (caller keeps today's exact behaviour).
TEST(RailSelect, NoPreferListKeepsCallerDefault) {
  Node local;
  uint8_t peer[1][16];
  mkGid(peer[0], 0xa1, 2);
  int how = -1;
  EXPECT_EQ(ibRailSelectDev(local.names, 3, local.gids, local.gidDev, 3,
                            peer, 1, 24, "", &how), -1);
  EXPECT_EQ(ibRailSelectDev(local.names, 3, local.gids, local.gidDev, 3,
                            peer, 1, 24, NULL, &how), -1);
}

// A single-pattern list has no rails: its one pattern is the fallback, applied
// unconditionally (documented semantics of the rail/fallback split).
TEST(RailSelect, SinglePatternListIsPureFallback) {
  Node local;
  uint8_t peer[1][16];
  mkGid(peer[0], 0xa1, 2);          // would match rail0 -- but rails are empty
  int how = -1;
  int dev = ibRailSelectDev(local.names, 3, local.gids, local.gidDev, 3,
                            peer, 1, 24, R"(rxe_lan\d*)", &how);
  EXPECT_EQ(dev, 2);
  EXPECT_EQ(how, IB_RAIL_SELECT_FALLBACK);
}

// The local rail may itself carry several GIDs (flat list, one entry per
// (dev,gid)); a match on ANY of them selects the rail.
TEST(RailSelect, LocalRailWithSecondaryGidStillMatches) {
  const char* names[2] = {"usb4_rdma5", "rxe_lan"};
  uint8_t gids[3][16];
  mkV4Gid(gids[0], 10, 2, 1, 5);    // rail's stray v4-mapped GID (first in table)
  mkGid(gids[1], 0xa1, 1);          // rail's ULA
  mkGid(gids[2], 0x5a, 1);          // rxe
  int gidDev[3] = {0, 0, 1};
  uint8_t peer[1][16];
  mkGid(peer[0], 0xa1, 2);
  int how = -1;
  int dev = ibRailSelectDev(names, 2, gids, gidDev, 3, peer, 1, 24, kPreferList, &how);
  EXPECT_EQ(dev, 0);
  EXPECT_EQ(how, IB_RAIL_SELECT_RAIL);
}

// ---------------------------------------------------------------------------
// wire format: advertise ALL valid GIDs per device (capped), old peers degrade
// ---------------------------------------------------------------------------
TEST(WireGids, PackAdvertisesPrimaryPlusExtra) {
  uint8_t primary[16], table[2][16], extra[16];
  mkGid(primary, 0xa1, 1);
  memcpy(table[0], primary, 16);
  mkGid(table[1], 0xa2, 1);
  uint8_t magic = 0, count = 0;
  int n = ibDevInfoPackGids(primary, table, 2, &magic, &count, extra);
  EXPECT_EQ(n, 2);
  EXPECT_EQ(magic, NCCL_IB_DEV_GIDS_MAGIC);
  EXPECT_EQ(count, 2);
  EXPECT_EQ(memcmp(extra, table[1], 16), 0);
}

TEST(WireGids, PackSingleGidDeviceCountsOne) {
  uint8_t primary[16], table[1][16], extra[16];
  mkGid(primary, 0xa1, 1);
  memcpy(table[0], primary, 16);   // table holds only the primary
  uint8_t magic = 0, count = 0;
  int n = ibDevInfoPackGids(primary, table, 1, &magic, &count, extra);
  EXPECT_EQ(n, 1);
  EXPECT_EQ(magic, NCCL_IB_DEV_GIDS_MAGIC);
  EXPECT_EQ(count, 1);
  uint8_t zero[16] = {0};
  EXPECT_EQ(memcmp(extra, zero, 16), 0) << "unused extra slot must stay zero on the wire";
}

// The extra slot is the ONE additional GID we can afford: prefer a native IPv6
// GID (the rail ULAs live there) over another v4-mapped one.
TEST(WireGids, PackPrefersNativeIpv6Extra) {
  uint8_t primary[16], table[3][16], extra[16];
  mkV4Gid(primary, 10, 2, 0, 31);      // primary got polluted with the v4 face
  memcpy(table[0], primary, 16);
  mkV4Gid(table[1], 10, 2, 1, 31);     // another v4-mapped
  mkGid(table[2], 0xa1, 1);            // the ULA -- must win the extra slot
  uint8_t magic = 0, count = 0;
  ibDevInfoPackGids(primary, table, 3, &magic, &count, extra);
  EXPECT_EQ(count, 2);
  EXPECT_EQ(memcmp(extra, table[2], 16), 0);
}

TEST(WireGids, RoundTripsTwoGidsPerDevice) {
  uint8_t primary[16], table[2][16], extra[16];
  mkGid(primary, 0xa1, 1);
  memcpy(table[0], primary, 16);
  mkGid(table[1], 0xa2, 1);
  uint8_t magic = 0, count = 0;
  ibDevInfoPackGids(primary, table, 2, &magic, &count, extra);

  uint8_t out[NCCL_IB_DEV_ADV_GIDS][16];
  int n = ibDevInfoUnpackGids(primary, magic, count, extra, out, NCCL_IB_DEV_ADV_GIDS);
  ASSERT_EQ(n, 2);
  EXPECT_EQ(memcmp(out[0], primary, 16), 0);
  EXPECT_EQ(memcmp(out[1], table[1], 16), 0);
}

// Wire-compat: OLD peers send zeros where magic/count/extra now live (their
// metadata struct is memset and the extra slot was a never-sent field). The
// unpacker must degrade to exactly the single advertised GID.
TEST(WireGids, OldSingleGidPeerDegradesToPrimary) {
  uint8_t primary[16], zero[16] = {0};
  mkGid(primary, 0xa1, 1);
  uint8_t out[NCCL_IB_DEV_ADV_GIDS][16];
  int n = ibDevInfoUnpackGids(primary, /*magic=*/0, /*count=*/0, zero, out, NCCL_IB_DEV_ADV_GIDS);
  ASSERT_EQ(n, 1);
  EXPECT_EQ(memcmp(out[0], primary, 16), 0);
}

// Malformed metadata must clamp, never hang or overrun: absurd count, or a
// magic-tagged but invalid extra GID, degrade to the primary alone.
TEST(WireGids, MalformedCountOrExtraClampsSafely) {
  uint8_t primary[16], garbage[16], zero[16] = {0};
  mkGid(primary, 0xa1, 1);
  mkLinkLocal(garbage, 7);          // link-local: not a valid advertised GID
  uint8_t out[NCCL_IB_DEV_ADV_GIDS][16];

  int n = ibDevInfoUnpackGids(primary, NCCL_IB_DEV_GIDS_MAGIC, /*count=*/200, garbage, out,
                              NCCL_IB_DEV_ADV_GIDS);
  EXPECT_EQ(n, 1) << "invalid extra dropped, count clamped";

  n = ibDevInfoUnpackGids(primary, NCCL_IB_DEV_GIDS_MAGIC, /*count=*/2, zero, out,
                          NCCL_IB_DEV_ADV_GIDS);
  EXPECT_EQ(n, 1) << "zero extra dropped even when count claims 2";
}

// ---------------------------------------------------------------------------
// listen-handle advertisement: 3 slots, rails first, breadth-first across
// devices (every device lands its first GID before any device lands a second).
// ---------------------------------------------------------------------------
TEST(AdvertiseBfs, RailsFirstThenBreadthFirst) {
  // dev0 = rxe (2 GIDs), dev1 = rail (1 GID), dev2 = rail (2 GIDs)
  uint8_t gids[5][16];
  int gidDev[5] = {0, 0, 1, 2, 2};
  mkGid(gids[0], 0x5a, 1);   // rxe gid A
  mkGid(gids[1], 0x5b, 1);   // rxe gid B
  mkGid(gids[2], 0xa1, 1);   // rail1 ULA
  mkGid(gids[3], 0xc3, 1);   // rail2 ULA
  mkGid(gids[4], 0xc4, 1);   // rail2 secondary
  int devIsRail[3] = {0, 1, 1};
  uint8_t out[3][16];
  int n = ibCollectAdvertiseGidsBfs(gids, gidDev, 5, devIsRail, 3, out, 3);
  ASSERT_EQ(n, 3);
  EXPECT_EQ(memcmp(out[0], gids[2], 16), 0) << "rail1's first GID leads";
  EXPECT_EQ(memcmp(out[1], gids[3], 16), 0) << "rail2's first GID next";
  EXPECT_EQ(memcmp(out[2], gids[0], 16), 0) << "then the non-rail's first GID";
}

TEST(AdvertiseBfs, SecondGidsOnlyAfterEveryFirst) {
  // 2 rails with 2 GIDs each, cap 4: both firsts, then both seconds.
  uint8_t gids[4][16];
  int gidDev[4] = {0, 0, 1, 1};
  mkGid(gids[0], 0xa1, 1);
  mkGid(gids[1], 0xa2, 1);
  mkGid(gids[2], 0xc3, 1);
  mkGid(gids[3], 0xc4, 1);
  int devIsRail[2] = {1, 1};
  uint8_t out[4][16];
  int n = ibCollectAdvertiseGidsBfs(gids, gidDev, 4, devIsRail, 2, out, 4);
  ASSERT_EQ(n, 4);
  EXPECT_EQ(memcmp(out[0], gids[0], 16), 0);
  EXPECT_EQ(memcmp(out[1], gids[2], 16), 0);
  EXPECT_EQ(memcmp(out[2], gids[1], 16), 0);
  EXPECT_EQ(memcmp(out[3], gids[3], 16), 0);
}

TEST(AdvertiseBfs, CapTruncates) {
  uint8_t gids[4][16];
  int gidDev[4] = {0, 1, 2, 3};
  for (int i = 0; i < 4; i++) mkGid(gids[i], (uint8_t)(0x10 + i), 1);
  int devIsRail[4] = {1, 1, 1, 1};
  uint8_t out[2][16];
  int n = ibCollectAdvertiseGidsBfs(gids, gidDev, 4, devIsRail, 4, out, 2);
  EXPECT_EQ(n, 2);
}

// ---------------------------------------------------------------------------
// validGid (relocated verbatim from connect.cc so selection and tests share the
// single definition): zero and link-local are not advertisable/matchable.
// ---------------------------------------------------------------------------
TEST(ValidGid, ZeroAndLinkLocalExcludedUlaAccepted) {
  union ibv_gid zero, ll, ula;
  memset(&zero, 0, sizeof(zero));
  mkLinkLocal(ll.raw, 2);
  mkGid(ula.raw, 0xa1, 1);
  EXPECT_FALSE(validGid(&zero));
  EXPECT_FALSE(validGid(&ll));
  EXPECT_TRUE(validGid(&ula));
}

}  // namespace
