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
#include <regex.h>
#include <stddef.h>

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

// Among the merged devices, return the index of the REACHABLE device that best
// matches the comma-separated HCA-name-prefix priority list `preferList` (e.g.
// "usb4_rdma,rxe_lan" -> rails before the reaches-all fallback). Returns -1 when
// no reachable device matches any prefix, so the caller keeps its existing
// default/search behaviour (e.g. the fallback reaches a non-adjacent peer). This
// is what NCCL_IB_SUBNET_PREFER_HCA configures: which HCA to prefer when more
// than one can reach the peer (the flat fallback always "reaches", so without a
// preference an adjacent peer wrongly stays on it instead of its rail).
// Translate a user pattern token into a POSIX ERE: strip a leading "/dev/"
// (devices are matched by their ibverbs name, e.g. "usb4_rdma5"), and accept the
// PCRE shorthand "\d" -> "[0-9]" so NCCL_IB_SUBNET_AWARE_ROUTING=prefer_hca[
// usb4_rdma\d*,rxe_lan\d*] works as written.
static inline void rrTranslatePattern(const char* in, char* out, size_t outsz) {
  if (strncmp(in, "/dev/", 5) == 0) in += 5;
  size_t o = 0;
  for (const char* p = in; *p && o + 1 < outsz; p++) {
    if (p[0] == '\\' && p[1] == 'd') {
      const char* sub = "[0-9]";
      for (const char* s = sub; *s && o + 1 < outsz; s++) out[o++] = *s;
      p++;
    } else {
      out[o++] = *p;
    }
  }
  out[o] = '\0';
}

static inline int ibPreferReachableDev(const char** devNames, const int* reachable,
                                       int nDevs, const char* preferList) {
  if (!preferList || !*preferList) return -1;      // no preference -> caller's default
  const char* p = preferList;
  while (*p) {
    char tok[128]; int n = 0;                       // next comma-separated pattern
    while (*p && *p != ',' && n < (int)sizeof(tok) - 1) tok[n++] = *p++;
    tok[n] = '\0';
    if (*p == ',') p++;
    if (n == 0) continue;
    char ere[160];
    rrTranslatePattern(tok, ere, sizeof(ere));
    regex_t re;
    if (regcomp(&re, ere, REG_EXTENDED | REG_NOSUB) != 0) continue;
    int found = -1;
    for (int d = 0; d < nDevs; d++)                 // first reachable dev matching this pattern
      if (reachable[d] && devNames[d] && regexec(&re, devNames[d], 0, NULL, 0) == 0) { found = d; break; }
    regfree(&re);
    if (found >= 0) return found;                   // highest-priority pattern with a reachable dev wins
  }
  return -1;                                        // nothing reachable matched any pattern
}

#endif // NCCL_NET_IB_SUBNET_MATCH_H_
