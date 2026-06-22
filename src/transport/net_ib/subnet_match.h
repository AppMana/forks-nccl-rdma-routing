/*************************************************************************
 * subnet_match.h - GID address-family + subnet comparison for net_ib's
 * reachability-aware device selection (ncclIbFindDevBySubnet).
 *
 * Factored out of connect.cc verbatim (no logic change) so the comparison is
 * unit-testable in isolation -- in particular against per-link IPv6 /64 GIDs
 * (Thunderbolt rail chains), where each rail reaches only its cabled neighbour.
 * There is exactly ONE definition: connect.cc includes this header.
 *************************************************************************/
#ifndef NCCL_NET_IB_SUBNET_MATCH_H_
#define NCCL_NET_IB_SUBNET_MATCH_H_

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

// Returns AF_INET for an IPv4-mapped (or IPv4-mapped-multicast) GID, else AF_INET6.
static inline sa_family_t getGidAddrFamily(union ibv_gid* gid) {
  const struct in6_addr* a = (struct in6_addr*)gid->raw;
  bool isIpV4Mapped = ((a->s6_addr32[0] | a->s6_addr32[1]) | (a->s6_addr32[2] ^ htonl(0x0000ffff))) == 0UL;
  bool isIpV4MappedMulticast =
    (a->s6_addr32[0] == htonl(0xff0e0000) && ((a->s6_addr32[1] | (a->s6_addr32[2] ^ htonl(0x0000ffff))) == 0UL));
  return (isIpV4Mapped || isIpV4MappedMulticast) ? AF_INET : AF_INET6;
}

// True iff local and remote GIDs share a subnet. IPv4-mapped: compare the IPv4
// /prefixLen (bytes 12-15). IPv6: compare the /64 subnet prefix (first 64 bits).
static inline bool gidSameSubnet(union ibv_gid* local, union ibv_gid* remote, int prefixLen) {
  sa_family_t localFam = getGidAddrFamily(local);
  sa_family_t remoteFam = getGidAddrFamily(remote);
  if (localFam != remoteFam) return false;
  if (localFam == AF_INET) {
    // IPv4-mapped: compare using configured prefix length.
    // IPv4 address is in bytes 12-15 of the raw GID.
    uint32_t localIp, remoteIp;
    memcpy(&localIp, local->raw + 12, 4);
    memcpy(&remoteIp, remote->raw + 12, 4);
    uint32_t mask = htonl(~((1U << (32 - prefixLen)) - 1));
    return (localIp & mask) == (remoteIp & mask);
  } else {
    // IPv6: compare subnet prefix (first 64 bits)
    return local->global.subnet_prefix == remote->global.subnet_prefix;
  }
}

#endif // NCCL_NET_IB_SUBNET_MATCH_H_
