/*************************************************************************
 * node_topo.h - explicit inter-node topology graph for net_ib device
 * selection (NCCL_NODE_TO_NODE_TOPO_FILE).
 *
 * NCCL's native NCCL_TOPO_FILE/NCCL_GRAPH_FILE are INTRA-node only; base NCCL
 * assumes a full-mesh network between nodes. On a Thunderbolt chain the
 * inter-node network is a LINE: a usb4_rdma rail reaches ONLY the directly
 * cabled neighbour; everything else must use the flat fallback (rxe_lan).
 * gidSameSubnet inference of that reachability is fragile (many IPv6
 * addresses per interface, rxe GID-index/SLAAC races, one GID on the wire),
 * so this module lets the operator DECLARE the graph instead:
 *
 *   { "<nodeIP>": { "<ifaceDevName>": { "neighbors": ["<nodeIP>", ...],
 *                                       "weight": <int> }, ... }, ... }
 *
 * node key  = OOB IPv4/IPv6 address the node bootstraps on (same values as
 *             VLLM_RAY_WORKER_IP_ORDER).
 * iface key = local ibverbs device name (e.g. "usb4_rdma5", "rxe_lan").
 * neighbors = node IPs directly reachable on that device.
 * weight    = edge cost (rail cheap e.g. 10, fallback expensive e.g. 100).
 *
 * Selection is a weighted DIRECT-EDGE lookup: among the local node's
 * interfaces, the minimum-weight one whose neighbors contain the remote node.
 * Deliberately NOT shortest-path: an intermediate chain node does not forward
 * RDMA, so a multi-hop "cheaper" rail path would hang exactly like the bug
 * this replaces. Not found -> -1, the caller keeps NCCL's default behaviour.
 *
 * Header-only and hardware-free (no verbs, no NCCL headers) so it is
 * unit-testable in isolation: rdma-routing/tests/node_topo_test.cc.
 * The JSON parser is hand-rolled for exactly this schema (objects, arrays of
 * strings, integers; unknown keys skipped) -- no third-party dependency.
 * Parsing is strict and fail-closed: a missing "neighbors"/"weight", a bad
 * address, or trailing garbage is a clean error, never a partial graph.
 *************************************************************************/
#ifndef NCCL_NET_IB_NODE_TOPO_H_
#define NCCL_NET_IB_NODE_TOPO_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/socket.h>
#include <arpa/inet.h>

#define NCCL_NODE_TOPO_MAX_NODES 32
#define NCCL_NODE_TOPO_MAX_IFACES 8
#define NCCL_NODE_TOPO_MAX_NEIGHBORS NCCL_NODE_TOPO_MAX_NODES
#define NCCL_NODE_TOPO_DEVNAME_MAX 64
#define NCCL_NODE_TOPO_ERRMSG_MAX 256

// A parsed node address: AF_INET (4 significant bytes) or AF_INET6 (16).
// IPv4-mapped IPv6 (::ffff:a.b.c.d) is canonicalized to AF_INET so a
// dual-stack accept()ed peer matches a graph keyed with plain IPv4.
typedef struct ncclNodeTopoAddr {
  int family;
  uint8_t bytes[16];
} ncclNodeTopoAddr;

typedef struct ncclNodeTopoIface {
  char devName[NCCL_NODE_TOPO_DEVNAME_MAX];
  int weight;
  int nNeighbors;
  ncclNodeTopoAddr neighbors[NCCL_NODE_TOPO_MAX_NEIGHBORS];
} ncclNodeTopoIface;

typedef struct ncclNodeTopoNode {
  ncclNodeTopoAddr addr;
  int nIfaces;
  ncclNodeTopoIface ifaces[NCCL_NODE_TOPO_MAX_IFACES];
} ncclNodeTopoNode;

typedef struct ncclNodeTopo {
  int nNodes;
  ncclNodeTopoNode nodes[NCCL_NODE_TOPO_MAX_NODES];
} ncclNodeTopo;

// ---------------------------------------------------------------- addresses

// Parse a numeric IPv4/IPv6 address string. IPv4-mapped IPv6 canonicalizes to
// AF_INET. Returns 0 on success, -1 on anything inet_pton rejects (hostnames
// are deliberately NOT resolved: the graph is keyed by literal OOB IPs).
static inline int ncclNodeTopoAddrFromString(const char* s, ncclNodeTopoAddr* out) {
  if (s == NULL || out == NULL) return -1;
  memset(out, 0, sizeof(*out));
  struct in_addr a4;
  if (inet_pton(AF_INET, s, &a4) == 1) {
    out->family = AF_INET;
    memcpy(out->bytes, &a4, 4);
    return 0;
  }
  struct in6_addr a6;
  if (inet_pton(AF_INET6, s, &a6) == 1) {
    if (IN6_IS_ADDR_V4MAPPED(&a6)) {
      out->family = AF_INET;
      memcpy(out->bytes, a6.s6_addr + 12, 4);
    } else {
      out->family = AF_INET6;
      memcpy(out->bytes, a6.s6_addr, 16);
    }
    return 0;
  }
  return -1;
}

static inline bool ncclNodeTopoAddrEq(const ncclNodeTopoAddr* a, const ncclNodeTopoAddr* b) {
  if (a == NULL || b == NULL || a->family != b->family) return false;
  return memcmp(a->bytes, b->bytes, a->family == AF_INET ? 4 : 16) == 0;
}

// ------------------------------------------------------------- JSON parsing
// Minimal recursive-descent parser for the topo schema. `p` walks the input;
// errors record a message + byte offset and unwind via return code.

typedef struct ncclNtParser {
  const char* buf;   // whole input (for error offsets)
  const char* p;     // cursor
  char* err;
  size_t errSz;
} ncclNtParser;

static inline int ncclNtFail(ncclNtParser* ps, const char* msg) {
  if (ps->err && ps->errSz)
    snprintf(ps->err, ps->errSz, "node topo JSON: %s (at byte %ld)", msg, (long)(ps->p - ps->buf));
  return -1;
}

static inline void ncclNtSkipWs(ncclNtParser* ps) {
  while (*ps->p == ' ' || *ps->p == '\t' || *ps->p == '\n' || *ps->p == '\r') ps->p++;
}

// Parse a JSON string into out (no \u escapes; addresses and device names
// never need them -- a \u is a clean error, not a crash).
static inline int ncclNtParseString(ncclNtParser* ps, char* out, size_t outSz) {
  ncclNtSkipWs(ps);
  if (*ps->p != '"') return ncclNtFail(ps, "expected '\"'");
  ps->p++;
  size_t o = 0;
  while (*ps->p && *ps->p != '"') {
    char c = *ps->p;
    if (c == '\\') {
      ps->p++;
      switch (*ps->p) {
        case '"': c = '"'; break;
        case '\\': c = '\\'; break;
        case '/': c = '/'; break;
        case 'n': c = '\n'; break;
        case 't': c = '\t'; break;
        case 'r': c = '\r'; break;
        case 'b': c = '\b'; break;
        case 'f': c = '\f'; break;
        default: return ncclNtFail(ps, "unsupported string escape");
      }
    }
    if (o + 1 >= outSz) return ncclNtFail(ps, "string too long");
    out[o++] = c;
    ps->p++;
  }
  if (*ps->p != '"') return ncclNtFail(ps, "unterminated string");
  ps->p++;
  out[o] = '\0';
  return 0;
}

static inline int ncclNtParseInt(ncclNtParser* ps, int* out) {
  ncclNtSkipWs(ps);
  char* end = NULL;
  long v = strtol(ps->p, &end, 10);
  if (end == ps->p) return ncclNtFail(ps, "expected an integer");
  if (*end == '.' || *end == 'e' || *end == 'E') return ncclNtFail(ps, "expected an integer, got a float");
  ps->p = end;
  *out = (int)v;
  return 0;
}

// Skip any well-formed JSON value (for unknown keys we tolerate but ignore).
static inline int ncclNtSkipValue(ncclNtParser* ps, int depth) {
  if (depth > 32) return ncclNtFail(ps, "nesting too deep");
  ncclNtSkipWs(ps);
  char scratch[NCCL_NODE_TOPO_DEVNAME_MAX];
  switch (*ps->p) {
    case '"': {
      // Skip a string of any length without buffering it.
      ps->p++;
      while (*ps->p && *ps->p != '"') {
        if (*ps->p == '\\' && ps->p[1]) ps->p++;
        ps->p++;
      }
      if (*ps->p != '"') return ncclNtFail(ps, "unterminated string");
      ps->p++;
      return 0;
    }
    case '{': {
      ps->p++;
      ncclNtSkipWs(ps);
      if (*ps->p == '}') { ps->p++; return 0; }
      for (;;) {
        if (ncclNtParseString(ps, scratch, sizeof(scratch)) != 0) return -1;
        ncclNtSkipWs(ps);
        if (*ps->p != ':') return ncclNtFail(ps, "expected ':'");
        ps->p++;
        if (ncclNtSkipValue(ps, depth + 1) != 0) return -1;
        ncclNtSkipWs(ps);
        if (*ps->p == ',') { ps->p++; continue; }
        if (*ps->p == '}') { ps->p++; return 0; }
        return ncclNtFail(ps, "expected ',' or '}'");
      }
    }
    case '[': {
      ps->p++;
      ncclNtSkipWs(ps);
      if (*ps->p == ']') { ps->p++; return 0; }
      for (;;) {
        if (ncclNtSkipValue(ps, depth + 1) != 0) return -1;
        ncclNtSkipWs(ps);
        if (*ps->p == ',') { ps->p++; continue; }
        if (*ps->p == ']') { ps->p++; return 0; }
        return ncclNtFail(ps, "expected ',' or ']'");
      }
    }
    default: {
      if (strncmp(ps->p, "true", 4) == 0) { ps->p += 4; return 0; }
      if (strncmp(ps->p, "false", 5) == 0) { ps->p += 5; return 0; }
      if (strncmp(ps->p, "null", 4) == 0) { ps->p += 4; return 0; }
      char* end = NULL;
      strtod(ps->p, &end);
      if (end != ps->p) { ps->p = end; return 0; }
      return ncclNtFail(ps, "unexpected token");
    }
  }
}

// "neighbors": [ "<ip>", ... ]
static inline int ncclNtParseNeighbors(ncclNtParser* ps, ncclNodeTopoIface* ifc) {
  ncclNtSkipWs(ps);
  if (*ps->p != '[') return ncclNtFail(ps, "\"neighbors\" must be an array of address strings");
  ps->p++;
  ncclNtSkipWs(ps);
  if (*ps->p == ']') { ps->p++; return 0; }
  for (;;) {
    char addrStr[64];
    ncclNtSkipWs(ps);
    if (*ps->p != '"') return ncclNtFail(ps, "\"neighbors\" entries must be address strings");
    if (ncclNtParseString(ps, addrStr, sizeof(addrStr)) != 0) return -1;
    if (ifc->nNeighbors >= NCCL_NODE_TOPO_MAX_NEIGHBORS) return ncclNtFail(ps, "too many neighbors");
    if (ncclNodeTopoAddrFromString(addrStr, &ifc->neighbors[ifc->nNeighbors]) != 0)
      return ncclNtFail(ps, "neighbor is not a valid IPv4/IPv6 address");
    ifc->nNeighbors++;
    ncclNtSkipWs(ps);
    if (*ps->p == ',') { ps->p++; continue; }
    if (*ps->p == ']') { ps->p++; return 0; }
    return ncclNtFail(ps, "expected ',' or ']'");
  }
}

// { "neighbors": [...], "weight": <int>, <unknown keys skipped> }
static inline int ncclNtParseIface(ncclNtParser* ps, ncclNodeTopoIface* ifc) {
  ncclNtSkipWs(ps);
  if (*ps->p != '{') return ncclNtFail(ps, "interface value must be an object");
  ps->p++;
  bool haveNeighbors = false, haveWeight = false;
  ncclNtSkipWs(ps);
  if (*ps->p == '}') { ps->p++; return ncclNtFail(ps, "interface missing \"neighbors\" and \"weight\""); }
  for (;;) {
    char key[NCCL_NODE_TOPO_DEVNAME_MAX];
    if (ncclNtParseString(ps, key, sizeof(key)) != 0) return -1;
    ncclNtSkipWs(ps);
    if (*ps->p != ':') return ncclNtFail(ps, "expected ':'");
    ps->p++;
    if (strcmp(key, "neighbors") == 0) {
      if (ncclNtParseNeighbors(ps, ifc) != 0) return -1;
      haveNeighbors = true;
    } else if (strcmp(key, "weight") == 0) {
      if (ncclNtParseInt(ps, &ifc->weight) != 0) return -1;
      haveWeight = true;
    } else {
      if (ncclNtSkipValue(ps, 0) != 0) return -1;
    }
    ncclNtSkipWs(ps);
    if (*ps->p == ',') { ps->p++; continue; }
    if (*ps->p == '}') { ps->p++; break; }
    return ncclNtFail(ps, "expected ',' or '}'");
  }
  if (!haveNeighbors) return ncclNtFail(ps, "interface missing required \"neighbors\"");
  if (!haveWeight) return ncclNtFail(ps, "interface missing required \"weight\"");
  return 0;
}

// { "<ifaceDevName>": { ... }, ... }
static inline int ncclNtParseNode(ncclNtParser* ps, ncclNodeTopoNode* node) {
  ncclNtSkipWs(ps);
  if (*ps->p != '{') return ncclNtFail(ps, "node value must be an object of interfaces");
  ps->p++;
  ncclNtSkipWs(ps);
  if (*ps->p == '}') { ps->p++; return 0; }
  for (;;) {
    if (node->nIfaces >= NCCL_NODE_TOPO_MAX_IFACES) return ncclNtFail(ps, "too many interfaces on a node");
    ncclNodeTopoIface* ifc = &node->ifaces[node->nIfaces];
    if (ncclNtParseString(ps, ifc->devName, sizeof(ifc->devName)) != 0) return -1;
    if (ifc->devName[0] == '\0') return ncclNtFail(ps, "empty interface name");
    ncclNtSkipWs(ps);
    if (*ps->p != ':') return ncclNtFail(ps, "expected ':'");
    ps->p++;
    if (ncclNtParseIface(ps, ifc) != 0) return -1;
    node->nIfaces++;
    ncclNtSkipWs(ps);
    if (*ps->p == ',') { ps->p++; continue; }
    if (*ps->p == '}') { ps->p++; return 0; }
    return ncclNtFail(ps, "expected ',' or '}'");
  }
}

// Parse the whole document into `topo`. Returns 0, or -1 with a message in
// err[] (never a partial/garbage graph: topo is only valid on 0).
static inline int ncclNodeTopoParse(const char* json, ncclNodeTopo* topo, char* err, size_t errSz) {
  if (err && errSz) err[0] = '\0';
  if (json == NULL || topo == NULL) {
    if (err && errSz) snprintf(err, errSz, "node topo JSON: null input");
    return -1;
  }
  memset(topo, 0, sizeof(*topo));
  ncclNtParser ps = {json, json, err, errSz};
  ncclNtSkipWs(&ps);
  if (*ps.p == '\0') return ncclNtFail(&ps, "empty document");
  if (*ps.p != '{') return ncclNtFail(&ps, "top level must be an object keyed by node IP");
  ps.p++;
  ncclNtSkipWs(&ps);
  if (*ps.p != '}') {
    for (;;) {
      if (topo->nNodes >= NCCL_NODE_TOPO_MAX_NODES) return ncclNtFail(&ps, "too many nodes");
      ncclNodeTopoNode* node = &topo->nodes[topo->nNodes];
      char addrStr[64];
      if (ncclNtParseString(&ps, addrStr, sizeof(addrStr)) != 0) return -1;
      if (ncclNodeTopoAddrFromString(addrStr, &node->addr) != 0)
        return ncclNtFail(&ps, "node key is not a valid IPv4/IPv6 address");
      ncclNtSkipWs(&ps);
      if (*ps.p != ':') return ncclNtFail(&ps, "expected ':'");
      ps.p++;
      if (ncclNtParseNode(&ps, node) != 0) return -1;
      topo->nNodes++;
      ncclNtSkipWs(&ps);
      if (*ps.p == ',') { ps.p++; continue; }
      if (*ps.p == '}') break;
      return ncclNtFail(&ps, "expected ',' or '}'");
    }
  }
  ps.p++;
  ncclNtSkipWs(&ps);
  if (*ps.p != '\0') return ncclNtFail(&ps, "trailing characters after the topology object");
  return 0;
}

static inline int ncclNodeTopoLoadFile(const char* path, ncclNodeTopo* topo, char* err, size_t errSz) {
  if (err && errSz) err[0] = '\0';
  FILE* f = fopen(path, "rb");
  if (f == NULL) {
    if (err && errSz) snprintf(err, errSz, "node topo: cannot open '%s'", path ? path : "(null)");
    return -1;
  }
  int ret = -1;
  char* buf = NULL;
  long sz;
  if (fseek(f, 0, SEEK_END) != 0 || (sz = ftell(f)) < 0 || fseek(f, 0, SEEK_SET) != 0) {
    if (err && errSz) snprintf(err, errSz, "node topo: cannot stat '%s'", path);
    goto out;
  }
  if (sz > 4L * 1024 * 1024) {
    if (err && errSz) snprintf(err, errSz, "node topo: '%s' is unreasonably large (%ld bytes)", path, sz);
    goto out;
  }
  buf = (char*)malloc((size_t)sz + 1);
  if (buf == NULL) goto out;
  if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
    if (err && errSz) snprintf(err, errSz, "node topo: short read on '%s'", path);
    goto out;
  }
  buf[sz] = '\0';
  ret = ncclNodeTopoParse(buf, topo, err, errSz);
out:
  free(buf);
  fclose(f);
  return ret;
}

// ---------------------------------------------------------------- selection

static inline int ncclNodeTopoFindNode(const ncclNodeTopo* topo, const ncclNodeTopoAddr* addr) {
  if (topo == NULL || addr == NULL) return -1;
  for (int n = 0; n < topo->nNodes; n++)
    if (ncclNodeTopoAddrEq(&topo->nodes[n].addr, addr)) return n;
  return -1;
}

static inline bool ncclNtIfaceReaches(const ncclNodeTopoIface* ifc, const ncclNodeTopoAddr* remote) {
  for (int i = 0; i < ifc->nNeighbors; i++)
    if (ncclNodeTopoAddrEq(&ifc->neighbors[i], remote)) return true;
  return false;
}

// General weighted direct-edge lookup: write into out[] every local interface
// whose declared neighbors contain `remote`, sorted by ascending weight (ties
// keep declaration order). Returns the count. out[0] is the interface
// selection must use; later entries are what the caller may fall back to when
// out[0] does not map to an actual local ibverbs device.
static inline int ncclNodeTopoRankIfaces(const ncclNodeTopo* topo, const ncclNodeTopoAddr* local,
                                         const ncclNodeTopoAddr* remote, const ncclNodeTopoIface** out, int maxOut) {
  if (out == NULL || maxOut <= 0) return 0;
  int nodeIdx = ncclNodeTopoFindNode(topo, local);
  if (nodeIdx < 0 || remote == NULL) return 0;
  const ncclNodeTopoNode* node = &topo->nodes[nodeIdx];
  int n = 0;
  for (int i = 0; i < node->nIfaces; i++) {
    const ncclNodeTopoIface* ifc = &node->ifaces[i];
    if (!ncclNtIfaceReaches(ifc, remote)) continue;
    // Insertion sort by weight, stable in declaration order.
    int pos = n;
    while (pos > 0 && out[pos - 1]->weight > ifc->weight) pos--;
    if (pos >= maxOut) continue;
    if (n < maxOut) n++;
    for (int j = n - 1; j > pos; j--) out[j] = out[j - 1];
    out[pos] = ifc;
  }
  return n;
}

// Convenience string-keyed form: returns the index (within the local node's
// ifaces[]) of the minimum-weight interface whose neighbors contain
// remoteAddr, and the local node's index via *nodeIdx. Returns -1 when the
// local node or the remote edge is not in the graph -- the caller keeps
// NCCL's default device (and falls back to gidSameSubnet inference).
static inline int ncclNodeTopoSelectIface(const ncclNodeTopo* topo, const char* localAddr, const char* remoteAddr,
                                          int* nodeIdx) {
  if (nodeIdx) *nodeIdx = -1;
  ncclNodeTopoAddr local, remote;
  if (ncclNodeTopoAddrFromString(localAddr, &local) != 0) return -1;
  if (ncclNodeTopoAddrFromString(remoteAddr, &remote) != 0) return -1;
  int n = ncclNodeTopoFindNode(topo, &local);
  if (n < 0) return -1;
  const ncclNodeTopoIface* ranked[NCCL_NODE_TOPO_MAX_IFACES];
  if (ncclNodeTopoRankIfaces(topo, &local, &remote, ranked, NCCL_NODE_TOPO_MAX_IFACES) < 1) return -1;
  if (nodeIdx) *nodeIdx = n;
  return (int)(ranked[0] - topo->nodes[n].ifaces);
}

#endif // NCCL_NET_IB_NODE_TOPO_H_
