// SPDX-License-Identifier: BSD-3-Clause
// NCCL_NODE_TO_NODE_TOPO_FILE: explicit inter-node topology graph for net_ib
// device selection, replacing gidSameSubnet /64 inference (which is fragile:
// an interface carries many IPv6 addresses across prefixes, rxe_lan's GID index
// ordering races with SLAAC at boot, and only ONE GID per device goes on the
// wire -- so a ring-closing hop between two nodes that advertised rxe_lan on
// DIFFERENT /64s finds no reachable device, NCCL falls back to a usb4 rail for
// a non-adjacent peer, and the collective hangs forever).
//
// The graph is keyed by OOB node IP (the address NCCL bootstraps on, same
// values as VLLM_RAY_WORKER_IP_ORDER). Selection: among the LOCAL node's
// interfaces, pick the minimum-weight one whose `neighbors` contains the
// remote node's IP. Not found -> -1 (caller keeps NCCL's default / falls back
// to gidSameSubnet). Exercises the REAL header net_ib/connect.cc includes.
#include <gtest/gtest.h>
#include <string.h>
#include <string>
#include <memory>
#include <sys/socket.h>
#include "node_topo.h"

namespace {

// 3-node Thunderbolt chain A - B - C (a LINE, not a mesh):
//   A(10.2.0.1) --usb4_rdma5-- B(10.2.0.2) --usb4_rdma15-- C(10.2.0.3)
// Every node's rxe_lan (flat soft-RoCE over the LAN) reaches everyone, at a
// much higher cost. A is NOT cabled to C: the only correct device for the
// ring-closing hop A->C is rxe_lan.
const char* kChain3 = R"json({
  "10.2.0.1": {
    "usb4_rdma5": { "neighbors": ["10.2.0.2"], "weight": 10 },
    "rxe_lan":    { "neighbors": ["10.2.0.2", "10.2.0.3"], "weight": 100 }
  },
  "10.2.0.2": {
    "usb4_rdma5":  { "neighbors": ["10.2.0.1"], "weight": 10 },
    "usb4_rdma15": { "neighbors": ["10.2.0.3"], "weight": 10 },
    "rxe_lan":     { "neighbors": ["10.2.0.1", "10.2.0.3"], "weight": 100 }
  },
  "10.2.0.3": {
    "usb4_rdma15": { "neighbors": ["10.2.0.2"], "weight": 10 },
    "rxe_lan":     { "neighbors": ["10.2.0.1", "10.2.0.2"], "weight": 100 }
  }
})json";

struct TopoHolder {
  std::unique_ptr<ncclNodeTopo> topo{new ncclNodeTopo};
  char err[NCCL_NODE_TOPO_ERRMSG_MAX];
  int parse(const char* json) {
    err[0] = '\0';
    return ncclNodeTopoParse(json, topo.get(), err, sizeof(err));
  }
};

// Resolve the selected iface index to its device name (what connect.cc maps to
// a local ibverbs device).
std::string selName(const ncclNodeTopo* topo, const char* local, const char* remote) {
  int nodeIdx = -1, ifaceIdx = ncclNodeTopoSelectIface(topo, local, remote, &nodeIdx);
  if (ifaceIdx < 0) return "";
  return topo->nodes[nodeIdx].ifaces[ifaceIdx].devName;
}

// ---------------- parsing ----------------

TEST(NodeTopoParse, ValidChainGraph) {
  TopoHolder h;
  ASSERT_EQ(h.parse(kChain3), 0) << h.err;
  EXPECT_EQ(h.topo->nNodes, 3);

  ncclNodeTopoAddr a;
  ASSERT_EQ(ncclNodeTopoAddrFromString("10.2.0.2", &a), 0);
  int b = ncclNodeTopoFindNode(h.topo.get(), &a);
  ASSERT_GE(b, 0);
  EXPECT_EQ(h.topo->nodes[b].nIfaces, 3);

  // B's rxe_lan lists both other nodes; its rails list exactly one each.
  for (int i = 0; i < h.topo->nodes[b].nIfaces; i++) {
    const ncclNodeTopoIface* ifc = &h.topo->nodes[b].ifaces[i];
    if (strcmp(ifc->devName, "rxe_lan") == 0) {
      EXPECT_EQ(ifc->weight, 100);
      EXPECT_EQ(ifc->nNeighbors, 2);
    } else {
      EXPECT_EQ(ifc->weight, 10);
      EXPECT_EQ(ifc->nNeighbors, 1);
    }
  }
}

TEST(NodeTopoParse, UnknownKeysInsideIfaceAreSkipped) {
  TopoHolder h;
  const char* json = R"({
    "10.2.0.1": {
      "rxe_lan": { "comment": "flat fallback", "neighbors": ["10.2.0.2"], "weight": 100, "extra": [1, {"x": null}, true] }
    },
    "10.2.0.2": { "rxe_lan": { "neighbors": ["10.2.0.1"], "weight": 100 } }
  })";
  ASSERT_EQ(h.parse(json), 0) << h.err;
  EXPECT_EQ(h.topo->nNodes, 2);
  EXPECT_EQ(h.topo->nodes[0].ifaces[0].weight, 100);
}

TEST(NodeTopoParse, EmptyInputIsCleanError) {
  TopoHolder h;
  EXPECT_EQ(h.parse(""), -1);
  EXPECT_NE(h.err[0], '\0') << "error message must say what went wrong";
  EXPECT_EQ(h.parse("   \n\t "), -1);
}

TEST(NodeTopoParse, MalformedJsonIsCleanError) {
  TopoHolder h;
  EXPECT_EQ(h.parse("{"), -1);
  EXPECT_EQ(h.parse("not json at all"), -1);
  EXPECT_EQ(h.parse(R"({"10.2.0.1": {"rxe_lan": {"neighbors": ["10.2.0.2")"), -1);  // truncated
  EXPECT_EQ(h.parse(R"([ "10.2.0.1" ])"), -1);  // top level must be an object
  EXPECT_EQ(h.parse(R"({"10.2.0.1": {"rxe_lan": {"neighbors": ["10.2.0.2"], "weight": 100}}} trailing)"), -1);
}

TEST(NodeTopoParse, MissingRequiredKeysIsCleanError) {
  TopoHolder h;
  // no "weight"
  EXPECT_EQ(h.parse(R"({"10.2.0.1": {"rxe_lan": {"neighbors": ["10.2.0.2"]}}})"), -1);
  EXPECT_NE(h.err[0], '\0');
  // no "neighbors"
  EXPECT_EQ(h.parse(R"({"10.2.0.1": {"rxe_lan": {"weight": 100}}})"), -1);
  // wrong types
  EXPECT_EQ(h.parse(R"({"10.2.0.1": {"rxe_lan": {"neighbors": "10.2.0.2", "weight": 100}}})"), -1);
  EXPECT_EQ(h.parse(R"({"10.2.0.1": {"rxe_lan": {"neighbors": [42], "weight": 100}}})"), -1);
  EXPECT_EQ(h.parse(R"({"10.2.0.1": {"rxe_lan": {"neighbors": ["10.2.0.2"], "weight": "cheap"}}})"), -1);
}

TEST(NodeTopoParse, BadAddressIsCleanError) {
  TopoHolder h;
  EXPECT_EQ(h.parse(R"({"not-an-ip": {"rxe_lan": {"neighbors": ["10.2.0.2"], "weight": 100}}})"), -1);
  EXPECT_EQ(h.parse(R"({"10.2.0.1": {"rxe_lan": {"neighbors": ["999.2.0.2"], "weight": 100}}})"), -1);
}

TEST(NodeTopoLoadFile, MissingFileIsCleanError) {
  TopoHolder h;
  EXPECT_EQ(ncclNodeTopoLoadFile("/nonexistent/topo.json", h.topo.get(), h.err, sizeof(h.err)), -1);
  EXPECT_NE(h.err[0], '\0');
}

TEST(NodeTopoLoadFile, ExampleFixtureParses) {
  TopoHolder h;
  ASSERT_EQ(ncclNodeTopoLoadFile(NODE_TOPO_FIXTURE, h.topo.get(), h.err, sizeof(h.err)), 0) << h.err;
  EXPECT_EQ(h.topo->nNodes, 3);
  EXPECT_EQ(selName(h.topo.get(), "10.2.0.1", "10.2.0.3"), "rxe_lan");
}

// ---------------- selection ----------------

TEST(NodeTopoSelect, AdjacentPicksTheRail) {
  TopoHolder h;
  ASSERT_EQ(h.parse(kChain3), 0) << h.err;
  // A->B: both usb4_rdma5 (10) and rxe_lan (100) reach B; min weight wins.
  EXPECT_EQ(selName(h.topo.get(), "10.2.0.1", "10.2.0.2"), "usb4_rdma5");
  // B->C must pick the rail TOWARD C, not the rail toward A.
  EXPECT_EQ(selName(h.topo.get(), "10.2.0.2", "10.2.0.3"), "usb4_rdma15");
  EXPECT_EQ(selName(h.topo.get(), "10.2.0.2", "10.2.0.1"), "usb4_rdma5");
}

TEST(NodeTopoSelect, NonAdjacentPicksTheFallbackNeverARail) {
  TopoHolder h;
  ASSERT_EQ(h.parse(kChain3), 0) << h.err;
  // The failure signature this feature exists to kill: the ring-closing hop
  // A->C is NOT cabled; only rxe_lan lists C. A usb4 rail here hangs forever.
  EXPECT_EQ(selName(h.topo.get(), "10.2.0.1", "10.2.0.3"), "rxe_lan");
  EXPECT_EQ(selName(h.topo.get(), "10.2.0.3", "10.2.0.1"), "rxe_lan");
}

TEST(NodeTopoSelect, RemoteNotInGraphIsMinusOne) {
  TopoHolder h;
  ASSERT_EQ(h.parse(kChain3), 0) << h.err;
  int nodeIdx = -1;
  EXPECT_EQ(ncclNodeTopoSelectIface(h.topo.get(), "10.2.0.1", "10.2.0.99", &nodeIdx), -1);
}

TEST(NodeTopoSelect, LocalNotInGraphIsMinusOne) {
  TopoHolder h;
  ASSERT_EQ(h.parse(kChain3), 0) << h.err;
  int nodeIdx = -1;
  EXPECT_EQ(ncclNodeTopoSelectIface(h.topo.get(), "10.2.0.99", "10.2.0.2", &nodeIdx), -1);
}

TEST(NodeTopoSelect, LowestWeightWinsRegardlessOfDeclarationOrder) {
  TopoHolder h;
  // Expensive iface declared FIRST; selection must still pick the cheap one.
  const char* json = R"({
    "10.2.0.1": {
      "rxe_lan":    { "neighbors": ["10.2.0.2"], "weight": 100 },
      "usb4_rdma5": { "neighbors": ["10.2.0.2"], "weight": 10 }
    },
    "10.2.0.2": { "rxe_lan": { "neighbors": ["10.2.0.1"], "weight": 100 } }
  })";
  ASSERT_EQ(h.parse(json), 0) << h.err;
  EXPECT_EQ(selName(h.topo.get(), "10.2.0.1", "10.2.0.2"), "usb4_rdma5");
}

// The REAL bug's shape: two nodes whose rxe_lan GIDs sat on DIFFERENT
// advertised /64s (SLAAC race decided which single GID went on the wire), so
// gidSameSubnet found no reachable device for the ring-closing hop and NCCL
// fell back to a usb4 rail toward a non-adjacent peer -> hang. With the graph,
// the OOB node IPs decide -- the GID prefixes never enter into it. Here the
// OOB addresses themselves sit on entirely different IPv6 /64s and routing
// still works, because reachability is DECLARED, not inferred.
TEST(NodeTopoSelect, DifferentSlash64sStillRouteViaDeclaredGraph) {
  TopoHolder h;
  const char* json = R"({
    "fd51:aaaa:bbbb::1": {
      "usb4_rdma5": { "neighbors": ["fd52:cccc:dddd::2"], "weight": 10 },
      "rxe_lan":    { "neighbors": ["fd52:cccc:dddd::2", "fd53:eeee:ffff::3"], "weight": 100 }
    },
    "fd52:cccc:dddd::2": {
      "usb4_rdma5":  { "neighbors": ["fd51:aaaa:bbbb::1"], "weight": 10 },
      "usb4_rdma15": { "neighbors": ["fd53:eeee:ffff::3"], "weight": 10 },
      "rxe_lan":     { "neighbors": ["fd51:aaaa:bbbb::1", "fd53:eeee:ffff::3"], "weight": 100 }
    },
    "fd53:eeee:ffff::3": {
      "usb4_rdma15": { "neighbors": ["fd52:cccc:dddd::2"], "weight": 10 },
      "rxe_lan":     { "neighbors": ["fd51:aaaa:bbbb::1", "fd52:cccc:dddd::2"], "weight": 100 }
    }
  })";
  ASSERT_EQ(h.parse(json), 0) << h.err;
  // Non-adjacent (ring-closing) hop across different /64s: MUST be rxe_lan.
  EXPECT_EQ(selName(h.topo.get(), "fd51:aaaa:bbbb::1", "fd53:eeee:ffff::3"), "rxe_lan");
  // Adjacent hop across different /64s: MUST be the rail toward that neighbor.
  EXPECT_EQ(selName(h.topo.get(), "fd51:aaaa:bbbb::1", "fd52:cccc:dddd::2"), "usb4_rdma5");
}

// ---------------- address normalization ----------------

TEST(NodeTopoAddr, V4MappedV6EqualsPlainV4) {
  // The OOB socket may report the peer as ::ffff:10.2.0.2 (dual-stack accept)
  // while the graph is keyed "10.2.0.2". These must be the same node.
  ncclNodeTopoAddr a, b;
  ASSERT_EQ(ncclNodeTopoAddrFromString("::ffff:10.2.0.2", &a), 0);
  ASSERT_EQ(ncclNodeTopoAddrFromString("10.2.0.2", &b), 0);
  EXPECT_TRUE(ncclNodeTopoAddrEq(&a, &b));
  EXPECT_EQ(a.family, AF_INET);
}

TEST(NodeTopoAddr, DistinctAddressesDiffer) {
  ncclNodeTopoAddr a, b;
  ASSERT_EQ(ncclNodeTopoAddrFromString("10.2.0.2", &a), 0);
  ASSERT_EQ(ncclNodeTopoAddrFromString("10.2.0.3", &b), 0);
  EXPECT_FALSE(ncclNodeTopoAddrEq(&a, &b));
  ncclNodeTopoAddr c, d;
  ASSERT_EQ(ncclNodeTopoAddrFromString("fd51::1", &c), 0);
  ASSERT_EQ(ncclNodeTopoAddrFromString("fd51::2", &d), 0);
  EXPECT_FALSE(ncclNodeTopoAddrEq(&c, &d));
  EXPECT_FALSE(ncclNodeTopoAddrEq(&a, &c));  // cross-family never equal
}

TEST(NodeTopoAddr, GarbageRejected) {
  ncclNodeTopoAddr a;
  EXPECT_EQ(ncclNodeTopoAddrFromString("", &a), -1);
  EXPECT_EQ(ncclNodeTopoAddrFromString("hostname.example", &a), -1);
  EXPECT_EQ(ncclNodeTopoAddrFromString("10.2.0", &a), -1);
}

// ---------------- ranked lookup (what connect.cc maps onto local devices) ----

TEST(NodeTopoRank, RanksAllReachingIfacesByAscendingWeight) {
  TopoHolder h;
  ASSERT_EQ(h.parse(kChain3), 0) << h.err;
  ncclNodeTopoAddr local, remote;
  ASSERT_EQ(ncclNodeTopoAddrFromString("10.2.0.1", &local), 0);
  ASSERT_EQ(ncclNodeTopoAddrFromString("10.2.0.2", &remote), 0);
  const ncclNodeTopoIface* ranked[NCCL_NODE_TOPO_MAX_IFACES];
  int n = ncclNodeTopoRankIfaces(h.topo.get(), &local, &remote, ranked, NCCL_NODE_TOPO_MAX_IFACES);
  ASSERT_EQ(n, 2);  // both the rail and the fallback reach B
  EXPECT_STREQ(ranked[0]->devName, "usb4_rdma5");
  EXPECT_STREQ(ranked[1]->devName, "rxe_lan");

  ASSERT_EQ(ncclNodeTopoAddrFromString("10.2.0.3", &remote), 0);
  n = ncclNodeTopoRankIfaces(h.topo.get(), &local, &remote, ranked, NCCL_NODE_TOPO_MAX_IFACES);
  ASSERT_EQ(n, 1);  // only the fallback reaches C
  EXPECT_STREQ(ranked[0]->devName, "rxe_lan");
}

}  // namespace
