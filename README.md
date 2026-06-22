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

The reachability decision is a small, pure, unit-tested module:

- `rdma-routing/src/route.c` — `/64` subnet sensing + an optional explicit
  reachability config file. Pure logic, no RDMA, device-agnostic.
- `rdma-routing/tests/route_test.cc` — the contract (two ends of a link share a
  `/64`; different links differ; a peerless rail is unreachable).

The net_ib change is just the **call site** that consults it.

## Build

```sh
make -j src.build NVCC_GENCODE="-gencode=arch=compute_86,code=sm_86"
# -> build/lib/libnccl.so.2.30.7
```

Standard NCCL build (see upstream `BUILD.md`). Pick `NVCC_GENCODE` for your GPUs.

## Install — replacing the NCCL that ships with PyTorch

PyTorch bundles its own NCCL (the `nvidia-nccl-cuXX` wheel, loaded at runtime from
`<site-packages>/torch/lib/libnccl.so.2`). `torch.cuda.nccl.version()` reports the
*compiled-against* version regardless of what is actually loaded — **trust the NCCL
startup banner** (`NCCL version 2.30.7+…`) to confirm which library is live.

### Option A — `LD_PRELOAD` (no rebuild of anything)

```sh
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2.30.7 python your_app.py
```

### Option B — `.deb` that replaces torch's bundled NCCL

`packaging/` builds `libnccl-rdma-routing_2.30.7_amd64.deb`, which:

1. installs `libnccl.so.2.30.7` (+ `libnccl.so.2` symlink) into
   `/usr/lib/x86_64-linux-gnu/`, and
2. repoints PyTorch's bundle at it, replacing the wheel's
   `torch/lib/libnccl.so.2` with a symlink to the system library (the same move
   the vLLM-Ampere image's `REPLACE_TORCH_BUNDLED_NCCL` step performs).

```sh
apt-get install -y ./libnccl-rdma-routing_2.30.7_amd64.deb
```

## Configuration

Reachability is **sensed from GID subnets** automatically — no per-node config
needed. Relevant NCCL env:

| Variable | Value | Why |
| --- | --- | --- |
| `NCCL_IB_HCA` | `usb4_rdma` | the rails to route across |
| `NCCL_IB_ADDR_FAMILY` | `AF_INET6` | use the per-link ULA GIDs, not v4-mapped |
| `NCCL_IB_MERGE_NICS` | `0` | do not fuse rails into one virtual device |

An optional `NCCL_ROUTING_CONF_FILE` provides explicit reachability for topologies
that subnet-sensing can't express.

## Layout

- `rdma-routing/` — the reachability module (`route.c`) + its test suite.
- `src/transport/net_ib.cc` — the patched device selection (the one change).

## Relationship to upstream

Tracks `v2.30.7-1`. The intent is a minimal, rebase-friendly diff: one routing
module plus one call site in net_ib, so the fork stays trivial to carry forward to
later NCCL releases.
