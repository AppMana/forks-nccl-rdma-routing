/*************************************************************************
 * rdmaroute/route.c - reachability-aware RDMA device selection. See route.h.
 * Pure logic, no NCCL/RDMA deps, so it unit-tests standalone. Device-agnostic.
 *************************************************************************/
#include "route.h"

#include <string.h>
#include <stdlib.h>
#include <arpa/inet.h>

/* A GID identifies a link only if it is a real, routable, non-link-local
 * address. fe80::/64 is identical on every link; all-zero is unset. */
static bool gid_is_routable(const uint8_t* g) {
  int nonzero = 0;
  for (int i = 0; i < 16; i++) nonzero |= g[i];
  if (!nonzero) return false;                 /* all-zero */
  if (g[0] == 0xfe && g[1] == 0x80) return false; /* link-local fe80::/64 */
  return true;
}

int rdmaroute_pick_local_hca(const uint8_t* local_gids, int n, const uint8_t* remote_gid) {
  if (!local_gids || !remote_gid || n <= 0) return -1;
  if (!gid_is_routable(remote_gid)) return -1;
  for (int i = 0; i < n; i++) {
    const uint8_t* g = local_gids + (size_t)i * 16;
    if (!gid_is_routable(g)) continue;
    /* same /64 subnet prefix (first 8 bytes) == same point-to-point TB link */
    if (memcmp(g, remote_gid, 8) == 0) return i;
  }
  return -1;
}

int rdmaroute_match_local_hca(const uint8_t* local_gids, int n_local,
                            const uint8_t* remote_gids, int n_remote) {
  if (!local_gids || !remote_gids || n_local <= 0 || n_remote <= 0) return -1;
  for (int i = 0; i < n_local; i++) {
    const uint8_t* lg = local_gids + (size_t)i * 16;
    if (!gid_is_routable(lg)) continue;
    for (int j = 0; j < n_remote; j++) {
      const uint8_t* rg = remote_gids + (size_t)j * 16;
      if (!gid_is_routable(rg)) continue;
      if (memcmp(lg, rg, 8) == 0) return i;   /* shared /64 == same link */
    }
  }
  return -1;
}

/* ---- NCCL_ROUTING_CONF_FILE: explicit reachability ---- */

/* Parse "<ipv6>/<plen>" into prefix[16] + *plen. Returns 0 ok, <0 on error. */
static int parse_prefix(const char* tok, uint8_t prefix[16], int* plen) {
  char buf[128];
  size_t n = strlen(tok);
  if (n == 0 || n >= sizeof(buf)) return -1;
  memcpy(buf, tok, n + 1);
  char* slash = strchr(buf, '/');
  if (!slash) return -1;                 /* require an explicit prefix length */
  *slash = '\0';
  char* endp = NULL;
  long pl = strtol(slash + 1, &endp, 10);
  if (endp == slash + 1 || *endp != '\0' || pl < 0 || pl > 128) return -1;
  if (inet_pton(AF_INET6, buf, prefix) != 1) return -1;
  *plen = (int)pl;
  return 0;
}

/* True if addr matches the first plen bits of prefix. */
static bool prefix_match(const uint8_t* a, const uint8_t* p, int plen) {
  int full = plen / 8, rem = plen % 8;
  if (full && memcmp(a, p, (size_t)full) != 0) return false;
  if (rem) {
    uint8_t mask = (uint8_t)(0xff << (8 - rem));
    if ((a[full] & mask) != (p[full] & mask)) return false;
  }
  return true;
}

int rdmaroute_conf_parse(const char* text, struct rdmaroute_conf* conf) {
  if (!text || !conf) return -1;
  memset(conf, 0, sizeof(*conf));
  const char* p = text;
  while (*p) {
    const char* eol = strchr(p, '\n');
    size_t len = eol ? (size_t)(eol - p) : strlen(p);
    char line[512];
    if (len >= sizeof(line)) return -1;
    memcpy(line, p, len);
    line[len] = '\0';
    p = eol ? eol + 1 : p + len;

    char* hash = strchr(line, '#');       /* strip comment */
    if (hash) *hash = '\0';

    char* save = NULL;
    char* tok = strtok_r(line, " \t\r", &save);
    if (!tok) continue;                    /* blank / comment-only */
    if (strcmp(tok, "reach") != 0) return -1;

    char* ltok = strtok_r(NULL, " \t\r", &save);
    if (!ltok) return -1;
    uint8_t lpfx[16]; int lplen;
    if (parse_prefix(ltok, lpfx, &lplen) != 0) return -1;

    int remotes = 0;
    char* rtok;
    while ((rtok = strtok_r(NULL, " \t\r", &save)) != NULL) {
      uint8_t rpfx[16]; int rplen;
      if (parse_prefix(rtok, rpfx, &rplen) != 0) return -1;
      if (conf->n >= RDMAROUTE_CONF_MAX_RULES) return -1;
      struct rdmaroute_rule* rule = &conf->rules[conf->n++];
      memcpy(rule->local_prefix, lpfx, 16);
      rule->local_plen = lplen;
      memcpy(rule->remote_prefix, rpfx, 16);
      rule->remote_plen = rplen;
      remotes++;
    }
    if (remotes == 0) return -1;           /* "reach <local>" with no remote */
  }
  return 0;
}

int rdmaroute_select(const uint8_t* local_gids, int n_local,
                     const uint8_t* remote_gids, int n_remote,
                     const struct rdmaroute_conf* conf) {
  if (!local_gids || !remote_gids || n_local <= 0 || n_remote <= 0) return -1;

  /* 1. sensing: a local device sharing a routable /64 with the peer. */
  int sensed = rdmaroute_match_local_hca(local_gids, n_local, remote_gids, n_remote);
  if (sensed >= 0) return sensed;

  /* 2. explicit conf: a rule permitting this local subnet -> the peer subnet. */
  if (conf) {
    for (int i = 0; i < n_local; i++) {
      const uint8_t* lg = local_gids + (size_t)i * 16;
      if (!gid_is_routable(lg)) continue;
      for (int j = 0; j < n_remote; j++) {
        const uint8_t* rg = remote_gids + (size_t)j * 16;
        if (!gid_is_routable(rg)) continue;
        for (int r = 0; r < conf->n; r++) {
          const struct rdmaroute_rule* rule = &conf->rules[r];
          if (prefix_match(lg, rule->local_prefix, rule->local_plen) &&
              prefix_match(rg, rule->remote_prefix, rule->remote_plen))
            return i;
        }
      }
    }
  }

  /* 3. nothing reaches -> caller uses the reaches-all fallback. */
  return -1;
}
