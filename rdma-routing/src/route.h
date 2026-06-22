/*************************************************************************
 * rdmaroute/route.h - reachability-aware RDMA device selection for the NCCL
 * net plugin. DEVICE-AGNOSTIC: it decides purely on GID /64 subnets and knows
 * nothing about the underlying fabric (usb4_rdma, soft-RoCE rxe, or any RoCE
 * NIC are identical to it). Two layers, both unit-testable without hardware:
 *
 *  1. SENSING (primary, zero-config): a local device reaches a peer iff it
 *     shares a routable /64 with one of the peer's GIDs. Inferred from the GID
 *     lists nodes exchange at init -- no chain order, no topology env, no TBV_*.
 *  2. EXPLICIT (fallback, NCCL_ROUTING_CONF_FILE): for topologies where
 *     same-/64 is not the right reachability predicate (routable multi-subnet),
 *     an admin states "local subnet X reaches remote subnet Y". Generic.
 *
 * When neither says a specific device reaches the peer, the caller uses the
 * reaches-all fallback device (e.g. rxe on the switched LAN).
 *************************************************************************/
#ifndef RDMAROUTE_H
#define RDMAROUTE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- sensing: GID /64 subnet matching (no config) ---- */

/* Given n local HCA GIDs (16 raw bytes each, contiguous) and a remote GID,
 * return the index of the local HCA whose routable /64 prefix (first 8 bytes)
 * matches the remote GID, or -1 if none. Link-local (fe80::/64) and all-zero
 * GIDs are never a match (they don't identify a link). */
int rdmaroute_pick_local_hca(const uint8_t* local_gids, int n, const uint8_t* remote_gid);

/* Given our n_local HCA GIDs and a peer's n_remote HCA GIDs, return the index
 * of OUR HCA that shares a routable /64 with one of the peer's GIDs, or -1.
 * Generalises rdmaroute_pick_local_hca across two GID lists. */
int rdmaroute_match_local_hca(const uint8_t* local_gids, int n_local,
                              const uint8_t* remote_gids, int n_remote);

/* ---- explicit reachability: NCCL_ROUTING_CONF_FILE ---- */

#define RDMAROUTE_CONF_MAX_RULES 128

/* One rule: a local /plen subnet may reach a remote /plen subnet. */
struct rdmaroute_rule {
  uint8_t local_prefix[16];
  uint8_t remote_prefix[16];
  int local_plen;
  int remote_plen;
};

struct rdmaroute_conf {
  int n;
  struct rdmaroute_rule rules[RDMAROUTE_CONF_MAX_RULES];
};

/* Parse NCCL_ROUTING_CONF_FILE text into conf. One rule per line:
 *
 *     reach <local-prefix>/<plen> <remote-prefix>/<plen> [<remote-prefix>/<plen> ...]
 *
 * '#' begins a comment; blank lines are ignored. Each remote on a line becomes
 * its own rule sharing the line's local subnet. Prefixes are IPv6 (incl.
 * v4-mapped). conf is zeroed first. Returns 0 on success, <0 on malformed input
 * or > RDMAROUTE_CONF_MAX_RULES rules. */
int rdmaroute_conf_parse(const char* text, struct rdmaroute_conf* conf);

/* Choose the local device to reach a peer described by remote_gids:
 *   >= 0 : index of a local device that reaches the peer -- by sensed /64 match
 *          first, else permitted by an explicit conf rule (local subnet of the
 *          device's GID reaches the remote subnet of one of the peer's GIDs).
 *   -1   : no specific device reaches the peer -> use the reaches-all fallback.
 * conf may be NULL (sensing only). */
int rdmaroute_select(const uint8_t* local_gids, int n_local,
                     const uint8_t* remote_gids, int n_remote,
                     const struct rdmaroute_conf* conf);

#ifdef __cplusplus
}
#endif

#endif /* RDMAROUTE_H */
