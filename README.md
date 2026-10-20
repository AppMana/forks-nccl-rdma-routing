# forks-nccl-rdma-routing

A fork of [NVIDIA/nccl](https://github.com/NVIDIA/nccl) (**v2.30.7**) that makes
NCCL's `net_ib` transport **reachability-aware** when it selects which RDMA device
(NIC / rail) to use for an inter-node connection.

The transport itself — QP management, the clear-to-send credit FIFO, RNR handling,
multi-QP, registration cache — is **stock NCCL**. We change exactly one thing: how
net_ib chooses *which* device to connect a given peer over. We do **not**
reimplement the transport.

## Why

Stock NCCL assumes a **fully-connected** RDMA fabric — every NIC can reach every
peer — so it picks the NIC for a connection by **GPU↔NIC PCI affinity**
(`ncclTopoGetNetDev` → `ncclTopoGetLocalNet`). That assumption breaks on
**partially-connected** fabrics:

- **Thunderbolt rail chains.** Each node has 1–2 `usb4_rdma` rails; each rail
  reaches only its **one cabled neighbour**. A mid-chain node's two rails sit on
  **different per-link GID `/64` subnets** — rail A reaches the upstream neighbour,
  rail B the downstream one, and neither reaches anyone else.
- **Multi-plane / rail-optimised** datacentre networks where a NIC only reaches a
  subset of peers.

On these fabrics stock NCCL routinely selects a rail that **cannot reach** the
peer. The connection then fails at `ibv_modify_qp(INIT→RTR)` with `ENETUNREACH`,
or — with `NCCL_IB_MERGE_NICS=1` (the default) — stripes a single connection across
*both* rails when only one of them reaches the peer.

## The fix (one thing)

net_ib already exchanges GIDs during connection setup. This fork adds a single
capability: **select the local device whose GID `/64` subnet reaches the peer's
GID**, falling back to the PCI-affinity choice, and then to a reaches-all device
(e.g. soft-RoCE over the LAN), when no rail matches.

The reachability decision is a small, pure, unit-tested header suite inside `src/transport/net_ib/`:

- `src/transport/net_ib/subnet_match.h` — `/64` subnet sensing, rail vs. fallback classification (`ibRailSelectDev`), multi-GID wire packing (`0xB6`), and generation-scoped sparse GID table caching (`ibGetValidGidSnapshot`).
- `src/transport/net_ib/node_topo.h` — explicit inter-node topology graph (`NCCL_NODE_TO_NODE_TOPO_FILE`).
- `ibverbs/tbrxe/tbrxe.c` — userspace `libibverbs` provider (`libtbrxe-rdmav34.so`) binding the `thunderbolt_frame_rxe.ko` private RDMA driver ID (`0x55534234`, `"USB4"`).
- `rdma-routing/tests/` — unit tests for `subnet_match.h`, `rail_select_test.cc`, and `node_topo.h`.

## Install — Single Python Package (`nccl-usb4`)

Instead of `.deb` packages, `LD_PRELOAD`, or symlink surgery inside `site-packages/nvidia/nccl/lib/`, install the standalone **`nccl-usb4`** wheel:

```sh
pip install --extra-index-url https://nodes.appmana.com/simple/ "nccl-usb4==2.30.7"
```

`nccl-usb4` bundles:
1. `nccl_usb4/nccl/lib/libnccl.so.2` (routed NCCL `2.30.7`)
2. `nccl_usb4/ibverbs/libtbrxe-rdmav34.so` and `libusb4_rdma-rdmav34.so` (userspace `libibverbs` providers `0.2.73` for `thunderbolt_frame_rxe`)
3. `nccl_usb4.pth` — runs `nccl_usb4.activate()` at Python startup before `torch` or `vllm` is imported, registering the `tbrxe` provider with `libibverbs.so.1`, placing `nccl_usb4` at `sys.path[0]` so `torch._preload_cuda_deps()` loads `nccl_usb4/nccl/lib/libnccl.so.2` instead of `nvidia-nccl-cu*`, and setting `VLLM_NCCL_SO_PATH` and `NCCL_NET_PLUGIN=none`.

## Build from Source

```sh
make -j src.build NVCC_GENCODE="-gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_100,code=sm_100"
python3 python/build_wheel.py --nccl-so build/lib/libnccl.so.2.30.7
# -> dist/nccl_usb4-2.30.7-py3-none-manylinux_2_28_x86_64.whl
```

## Configuration

Reachability is either **declared explicitly** (`NCCL_NODE_TO_NODE_TOPO_FILE`,
preferred) or **sensed from GID subnets** automatically. Relevant NCCL env:

| Variable | Value | Why |
| --- | --- | --- |
| `NCCL_NODE_TO_NODE_TOPO_FILE` | `/etc/nccl/node-topo.json` | declared inter-node topology graph; takes precedence over GID sensing (see below) |
| `NCCL_IB_HCA` | `usb4_rdma,rxe_lan` | the rails **and** a reaches-all fallback (soft-RoCE on the LAN) for non-adjacent collective edges |
| `NCCL_IB_ADDR_FAMILY` | `AF_INET6` | use the per-link ULA GIDs, not v4-mapped |
| `NCCL_IB_MERGE_NICS` | `0` | do not fuse rails into one virtual device |
| `NCCL_IB_SUBNET_AWARE_ROUTING` | `prefer_hca[usb4_rdma\d*,rxe_lan\d*]` | enable subnet-aware routing **and** set HCA preference (see below) |

### `NCCL_NODE_TO_NODE_TOPO_FILE` — the declared inter-node graph

NCCL's native `NCCL_TOPO_FILE`/`NCCL_GRAPH_FILE` are *intra*-node only; base
NCCL assumes a full-mesh network between nodes. On the Thunderbolt chain the
inter-node network is a LINE, and inferring per-peer reachability from
advertised GID /64s is fragile: an interface carries many IPv6 addresses
across prefixes, `rxe_lan`'s GID index ordering races with SLAAC at boot, and
only ONE GID per device goes on the wire — so a ring-closing hop between two
nodes that advertised `rxe_lan` on different /64s finds no reachable device,
NCCL keeps a usb4 rail toward a non-adjacent peer, and the collective hangs
forever. The topo file replaces that inference with a declaration:

```json
{
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
}
```

- **node key** — the OOB IPv4/IPv6 address the node bootstraps on (the same
  values as `VLLM_RAY_WORKER_IP_ORDER`).
- **iface key** — the local ibverbs device name (`usb4_rdma5`, `rxe_lan`, …).
- **neighbors** — node IPs directly reachable on that device.
- **weight** — edge cost: rails cheap (10), the reaches-all fallback expensive
  (100).

Per connection, both the connector and the acceptor resolve the peer's OOB IP
(the connector from the listen handle's `connectAddr`, the acceptor from the
accepted socket's remote address) and pick the **minimum-weight local
interface whose `neighbors` contain that IP**. Selection is deliberately
direct-edge only, never multi-hop: an intermediate chain node does not forward
RDMA. Semantics:

| Situation | Behaviour |
| --- | --- |
| env unset | byte-for-byte the GID-sensing behaviour below |
| file missing / malformed / local node absent | `WARN` at init + GID-sensing fallback (never a crash or hang) |
| peer's IP not a declared neighbor of any local iface | `WARN` + keep NCCL's default device / GID-sensing fallback |
| graph names an iface with no matching local ibverbs device | `WARN` + GID-sensing fallback |

Implemented in `src/transport/net_ib/node_topo.h` (header-only, hand-rolled
schema parser, no third-party deps), loaded once at plugin init in
`init.cc`/`connect.cc`; unit-tested hardware-free in
`rdma-routing/tests/node_topo_test.cc` with the example graph at
`rdma-routing/tests/fixtures/node_topo_example.json`.

### `NCCL_IB_SUBNET_AWARE_ROUTING` — an expression, not just a flag

Stock 2.30.x treats this as `0`/`1`. This fork extends it: the value can carry an
HCA **preference**, because the reaches-all fallback (rxe over the LAN) matches
*every* peer's subnet — so without a preference, an adjacent peer wrongly stays on
the slow fallback instead of overriding to its fast rail.

| Value | Behaviour |
| --- | --- |
| `0` / unset | off |
| `1` | on, legacy keep-default |
| `prefer_hca[p1,p2,…]` | on, and when several devices reach a peer, prefer the first whose name matches pattern `p1`, then `p2`, … |

Patterns are POSIX ERE; `\d` (→ `[0-9]`) and a leading `/dev/` are accepted, so
`prefer_hca[usb4_rdma\d*,rxe_lan\d*]` means **rail for adjacent neighbours, rxe
only when no rail reaches** (the non-adjacent ring/tree edges). Implemented in
`ibPreferReachableDev` (`src/transport/net_ib/subnet_match.h`), unit-tested in
`rdma-routing/tests/subnet_match_test.cc`.

An optional `NCCL_ROUTING_CONF_FILE` provides explicit reachability for topologies
that subnet-sensing can't express.

## Layout

- `rdma-routing/` — the reachability module (`route.c`) + its test suite.
- `src/transport/net_ib.cc` — the patched device selection (the one change).

## Relationship to upstream

Tracks `v2.30.7-1`. The intent is a minimal, rebase-friendly diff: one routing
module plus one call site in net_ib, so the fork stays trivial to carry forward to
later NCCL releases.
