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
// NOTE: do NOT include <infiniband/verbs.h> here. net_ib uses NCCL's vendored
// verbs definitions (src/include/ibvcore.h, unless NCCL_BUILD_RDMA_CORE); pulling
// in the system header alongside it conflicts on `struct verbs_context`. The
// includer provides `union ibv_gid` -- ibvcore.h via ibvwrap.h in connect.cc, or
// the system <infiniband/verbs.h> in the standalone unit test.
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

// --- GID validity (relocated verbatim from connect.cc: single definition,
// shared by selection, advertisement, and the unit tests) ---
static inline bool configuredGid(union ibv_gid* gid) {
  const struct in6_addr* a = (struct in6_addr*)gid->raw;
  int trailer = (a->s6_addr32[1] | a->s6_addr32[2] | a->s6_addr32[3]);
  if (((a->s6_addr32[0] | trailer) == 0UL) || ((a->s6_addr32[0] == htonl(0xfe800000)) && (trailer == 0UL))) {
    return false;
  }
  return true;
}

static inline bool linkLocalGid(union ibv_gid* gid) {
  const struct in6_addr* a = (struct in6_addr*)gid->raw;
  if (a->s6_addr32[0] == htonl(0xfe800000) && a->s6_addr32[1] == 0UL) {
    return true;
  }
  return false;
}

static inline bool validGid(union ibv_gid* gid) {
  return (configuredGid(gid) && !linkLocalGid(gid));
}

// ---------------------------------------------------------------------------
// Self-inferring rail/fallback device selection (the DEFAULT path underneath
// the optional NCCL_NODE_TO_NODE_TOPO_FILE override).
//
// The prefer-list (NCCL_IB_SUBNET_AWARE_ROUTING=prefer_hca[usb4_rdma\d*,
// rxe_lan\d*]) is split: every pattern except the LAST names RAILS (selected
// only when a peer-advertised GID shares the rail's /64 -- the per-link ULA is
// deterministic, both cabled ends share it); the LAST pattern names the
// FALLBACK, selected UNCONDITIONALLY when no rail matched (its prefixes are
// SLAAC-raced and legitimately differ between nodes, so subnet-matching it is
// exactly the fragility that hung the ring-closing hop).
// ---------------------------------------------------------------------------

#define IB_RAIL_SELECT_RAIL 0
#define IB_RAIL_SELECT_FALLBACK 1

// How the peer's per-device GIDs go on the wire: the primary gid (unchanged
// field) plus ONE extra GID in the previously send-unused remoteGid slot,
// tagged by a magic byte in a previously zero padding byte. Old peers send
// zeros there -> degrade to single-GID.
#define NCCL_IB_DEV_GIDS_MAGIC 0xB6
#define NCCL_IB_DEV_ADV_GIDS 2

// Extract the idx-th non-empty comma-separated pattern of the prefer-list.
// Returns 1 if it exists (written to tok), 0 otherwise.
static inline int rrPreferListPattern(const char* preferList, int idx, char* tok, size_t toksz) {
  if (preferList == NULL) return 0;
  const char* p = preferList;
  int i = 0;
  while (*p) {
    size_t n = 0;
    while (*p && *p != ',') {
      if (n + 1 < toksz) tok[n++] = *p;
      p++;
    }
    tok[n] = '\0';
    if (*p == ',') p++;
    if (n == 0) continue;
    if (i == idx) return 1;
    i++;
  }
  return 0;
}

static inline int rrPreferListCount(const char* preferList) {
  char tok[128];
  int c = 0;
  while (rrPreferListPattern(preferList, c, tok, sizeof(tok))) c++;
  return c;
}

// One pattern (user syntax: \d shorthand, optional /dev/ prefix) against one
// ibverbs device name. POSIX ERE search, same semantics as ibPreferReachableDev.
static inline int rrPatternMatchesDev(const char* pattern, const char* devName) {
  if (pattern == NULL || devName == NULL) return 0;
  char ere[160];
  rrTranslatePattern(pattern, ere, sizeof(ere));
  regex_t re;
  if (regcomp(&re, ere, REG_EXTENDED | REG_NOSUB) != 0) return 0;
  int m = (regexec(&re, devName, 0, NULL, 0) == 0);
  regfree(&re);
  return m;
}

// Rail = matches any pattern but the last; fallback = matches the last.
static inline int ibDevIsRail(const char* devName, const char* preferList) {
  int npat = rrPreferListCount(preferList);
  char tok[128];
  for (int i = 0; i + 1 < npat; i++) {
    if (rrPreferListPattern(preferList, i, tok, sizeof(tok)) && rrPatternMatchesDev(tok, devName)) return 1;
  }
  return 0;
}
static inline int ibDevIsFallback(const char* devName, const char* preferList) {
  int npat = rrPreferListCount(preferList);
  if (npat == 0) return 0;
  char tok[128];
  return rrPreferListPattern(preferList, npat - 1, tok, sizeof(tok)) && rrPatternMatchesDev(tok, devName);
}

// Select a device for a peer that advertised peerGids (flat list):
//   rail whose /64 contains ANY valid peer GID (patterns in priority order)
//   -> else the first device matching the fallback pattern, unconditionally
//   -> else -1 (caller keeps its default).
// localGids is a flat (gid, device-index) list covering ALL valid GIDs of every
// local device (a device may appear once per GID). *how reports RAIL/FALLBACK.
static inline int ibRailSelectDev(const char** devNames, int nDevs,
                                  const uint8_t (*localGids)[16], const int* localGidDev, int nLocalGids,
                                  const uint8_t (*peerGids)[16], int nPeerGids,
                                  int prefixLen, const char* preferList, int* how) {
  if (how) *how = -1;
  int npat = rrPreferListCount(preferList);
  if (npat == 0) return -1;                       // no prefer-list -> caller keeps its default path
  char tok[128];
  // RAILS, in priority order: select on a shared /64 with ANY valid peer GID.
  for (int pi = 0; pi + 1 < npat; pi++) {
    if (!rrPreferListPattern(preferList, pi, tok, sizeof(tok))) continue;
    for (int d = 0; d < nDevs; d++) {
      if (devNames[d] == NULL || !rrPatternMatchesDev(tok, devNames[d])) continue;
      for (int l = 0; l < nLocalGids; l++) {
        if (localGidDev[l] != d) continue;
        union ibv_gid lg;
        memcpy(lg.raw, localGids[l], 16);
        if (!validGid(&lg)) continue;
        for (int p = 0; p < nPeerGids; p++) {
          union ibv_gid pg;
          memcpy(pg.raw, peerGids[p], 16);
          if (!validGid(&pg)) continue;
          if (gidSameSubnet(&lg, &pg, prefixLen)) {
            if (how) *how = IB_RAIL_SELECT_RAIL;
            return d;
          }
        }
      }
    }
  }
  // FALLBACK (last pattern): unconditional -- no subnet match, ever.
  if (rrPreferListPattern(preferList, npat - 1, tok, sizeof(tok))) {
    for (int d = 0; d < nDevs; d++) {
      if (devNames[d] != NULL && rrPatternMatchesDev(tok, devNames[d])) {
        if (how) *how = IB_RAIL_SELECT_FALLBACK;
        return d;
      }
    }
  }
  return -1;                                      // neither -> caller keeps NCCL's default
}

// Pack one device's advertised GIDs: primary (already on the wire in .gid) plus
// the best extra from the device's full valid-GID table. Returns the advertised
// count (1 or 2); writes magic/count and the extra slot (zeroed when unused).
static inline int ibDevInfoPackGids(const uint8_t primary[16], const uint8_t (*table)[16], int nTable,
                                    uint8_t* magicOut, uint8_t* countOut, uint8_t extraOut[16]) {
  *magicOut = NCCL_IB_DEV_GIDS_MAGIC;
  memset(extraOut, 0, 16);
  // The extra slot fits ONE more GID: prefer a native IPv6 GID (the per-link
  // rail ULAs live there) over another v4-mapped one, first-in-table-order.
  int found = -1;
  for (int pass = 0; pass < 2 && found < 0; pass++) {
    for (int i = 0; i < nTable; i++) {
      if (memcmp(table[i], primary, 16) == 0) continue;
      union ibv_gid g;
      memcpy(g.raw, table[i], 16);
      if (!validGid(&g)) continue;
      if (pass == 0 && getGidAddrFamily(&g) != AF_INET6) continue;
      found = i;
      break;
    }
  }
  if (found >= 0) {
    memcpy(extraOut, table[found], 16);
    *countOut = 2;
    return 2;
  }
  *countOut = 1;
  return 1;
}

// Unpack a peer device's advertised GIDs; old single-GID peers (magic != ours,
// i.e. the bytes their memset left as zero) yield just the primary. Malformed
// counts/extras clamp to the primary -- never fail, never overrun.
static inline int ibDevInfoUnpackGids(const uint8_t primary[16], uint8_t magic, uint8_t count,
                                      const uint8_t extra[16], uint8_t (*out)[16], int maxOut) {
  int n = 0;
  if (maxOut <= 0) return 0;
  memcpy(out[n++], primary, 16);                  // legacy field: always present
  if (magic == NCCL_IB_DEV_GIDS_MAGIC && count >= 2 && n < maxOut) {
    union ibv_gid g;
    memcpy(g.raw, extra, 16);
    // Malformed extras (zero, link-local, duplicate of primary) are dropped and
    // an absurd count clamps here: nothing past the one extra slot is ever read.
    if (validGid(&g) && memcmp(extra, primary, 16) != 0) memcpy(out[n++], extra, 16);
  }
  return n;
}

// Flatten per-device GID lists into the listen handle's advertisement slots:
// rail devices before non-rail, breadth-first across devices (every device
// lands its first GID before any device lands its second). Returns count.
static inline int ibCollectAdvertiseGidsBfs(const uint8_t (*gids)[16], const int* gidDev, int nGids,
                                            const int* devIsRail, int nDevs, uint8_t (*out)[16], int maxOut) {
  int n = 0;
  for (int r = 0; n < maxOut; r++) {              // round r emits each device's r-th GID
    int emitted = 0;
    for (int pass = 0; pass < 2 && n < maxOut; pass++) {   // rails first, then the rest
      for (int d = 0; d < nDevs && n < maxOut; d++) {
        if ((devIsRail[d] != 0) != (pass == 0)) continue;
        int seen = 0, idx = -1;
        for (int g = 0; g < nGids; g++) {
          if (gidDev[g] != d) continue;
          if (seen == r) { idx = g; break; }
          seen++;
        }
        if (idx < 0) continue;
        memcpy(out[n++], gids[idx], 16);
        emitted = 1;
      }
    }
    if (!emitted) break;                          // every device exhausted
  }
  return n;
}

// Collect the GIDs a node advertises in its listen handle / connect metadata, so
// a peer can match whichever of OUR devices shares its /64 (a rail), not just the
// PCI-affinity default. On a Thunderbolt rail chain the default is the flat
// fallback (rxe), so advertising only it means the peer can NEVER match a rail
// and every connection collapses onto rxe. Writes up to maxOut GIDs to out[],
// returns the count. devValid[d] != 0 means device d has a usable RoCE GID.
static inline int ibCollectAdvertiseGids(const uint8_t devGids[][16], const int* devValid,
                                         int nDevs, int defaultDev, uint8_t out[][16], int maxOut) {
  (void)defaultDev;  // advertise ALL devices, not just the affinity default
  int n = 0;
  for (int d = 0; d < nDevs && n < maxOut; d++) {
    if (devValid[d]) memcpy(out[n++], devGids[d], 16);
  }
  return n;
}

#endif // NCCL_NET_IB_SUBNET_MATCH_H_
