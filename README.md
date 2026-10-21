# forks-nccl-rdma-routing (`nccl-usb4`)

Fork of [NVIDIA/nccl](https://github.com/NVIDIA/nccl) (**v2.32.3-1**) that adds reachability-aware RDMA device selection to `net_ib` and packages the routed `libnccl.so.2` together with the `libtbrxe` userspace `libibverbs` provider as a single Python wheel (`nccl-usb4==2.32.3`).

## Why This Fork Exists

1. **Per-peer RDMA rail selection on non-full-mesh fabrics**:
   Stock NCCL assumes every RDMA NIC (`ncclIbDevs`) can reach every peer on a shared fabric and assigns NICs to channels strictly by local GPU-to-NIC PCI affinity (`ncclTopoGetNetDev`). On point-to-point USB4/Thunderbolt chains (`thunderbolt_frame_rxe`), each `usb4_rdma*` HCA is a direct cable to a single adjacent node on its own `/64` IPv6 ULA subnet, while non-adjacent nodes are only reachable via soft-RoCE over Ethernet (`rxe_lan`). Stock NCCL either selects a `usb4_rdma*` rail that cannot reach a non-adjacent peer or selects the slow `rxe_lan` fallback for an adjacent peer because `rxe_lan`'s backing PCI NIC is closer in the PCI tree than the PCI-less virtual HCA.
2. **Why routing lives in `net_ib` rather than an external `ncclNet` plugin**:
   NCCL's topology engine (`src/graph/search.cc`) and collective/P2P channel setup choose the `dev` index before calling `ncclNet->connect(dev, ...)` / `listen(dev, ...)`. External `ncclNet` plugins also assume every IB device has a sysfs PCI path and fail on virtual `usb4_rdma*` HCAs.
3. **Why the userspace `tbrxe` provider is bundled**:
   On the host kernel (`AppMana/forks-thunderbolt`), `thunderbolt_frame_rxe.ko` implements the standard kernel `rxe` uverbs ABI over lossless USB4 XDomain frames (`thunderbolt_frame.ko`), but registers under private RDMA driver ID `0x55534234` (`"USB4"`) so unloading `rdma_rxe` never tears down USB4 RDMA rails. Because `libibverbs` matches providers by kernel driver ID and stamps that ID into every `uverbs` ioctl header, userspace needs `libtbrxe-rdmav<PABI>.so` (`ibverbs/tbrxe/tbrxe.c`), which compiles the `rdma-core` `rxe` provider bound exclusively to driver ID `0x55534234`.

## Changes Relative to Upstream `NVIDIA/nccl` (`v2.32.3-1`)

- **`src/transport/net_ib/subnet_match.h`, `connect.cc`, `common.{h,cc}`, `init.cc`**:
  - **Wire-compatible multi-GID advertisement**: Packs a second valid GID per device into `ncclIbDevInfo` using two former padding bytes (`advGidMagic = 0xB6`, `advGidCount`) and the send-unused `remoteGid` field. `sizeof(ncclIbDevInfo)` and all field offsets are unchanged (`static_assert`-checked in `common.h`).
  - **Generation-cached GID snapshots**: Caches valid GIDs per `ncclIbDev` (`ibGetValidGidSnapshot`) and invalidates the cache on `IBV_EVENT_GID_CHANGE`.
  - **`NCCL_IB_SUBNET_AWARE_ROUTING=prefer_hca[...]`**: Extends `NCCL_IB_SUBNET_AWARE_ROUTING` beyond `0`/`1` to accept an ordered HCA regex list (e.g. `prefer_hca[usb4_rdma\d*,rxe_lan\d*]`). During connection setup, `ibRailSelectDev` selects a matching rail whose `/64` subnet reaches the peer's advertised GIDs and falls back to the last pattern (`rxe_lan`) when no rail matches.
- **`src/transport/net_ib/node_topo.h`**:
  - **`NCCL_NODE_TO_NODE_TOPO_FILE`**: Optional JSON topology file mapping `(node_oob_ip, ib_dev) -> {neighbors, weight}` for deterministic minimum-weight direct-edge selection before GID subnet sensing.
- **`src/graph/search.cc`**:
  - Applies `prefer_hca[...]` priority in `ncclTopoGetNetDev` for P2P channels (`graph == NULL`) so P2P listeners and connectors prefer `usb4_rdma*` rails over `rxe_lan`.
- **`src/plugin/net.cc`**:
  - Skips external `ncclNet` plugins (such as NGC's default `NCCL_NET_PLUGIN=spcx`) when `NCCL_IB_SUBNET_AWARE_ROUTING` is set, when PCI-less RDMA devices (`usb4_rdma*`, `rxe_lan*`) exist in `/sys/class/infiniband/`, or when `spcx` is requested without an `mlx5_*` HCA, preventing NULL PCI-path dereferences in external plugins and preserving internal `net_ib` routing.
- **`ibverbs/tbrxe/`**:
  - Userspace `libibverbs` provider (`tbrxe.c`, `tbrxe.driver`, `CMakeLists.txt`) binding kernel driver ID `0x55534234` (`RDMA_DRIVER_USB4_RDMA`).
- **`python/` (`nccl-usb4`)**:
  - Standalone Python wheel builder (`python/build_wheel.py`) and runtime activation module (`python/nccl_usb4/__init__.py`).
- **`rdma-routing/tests/`**:
  - Standalone GTest suite (`subnet_match_test.cc`, `rail_select_test.cc`, `node_topo_test.cc` — 57 tests) that compiles and runs without GPUs or RDMA hardware.

## Single-Package Installation (`nccl-usb4==2.32.3`)

Containers and virtual environments do not need any custom `.deb` packages:

```sh
pip install --extra-index-url https://nodes.appmana.com/simple/ "nccl-usb4==2.32.3"
```

The `nccl-usb4` wheel contains:
1. `nccl_usb4/nccl/lib/libnccl.so.2` — routed NCCL `2.32.3`
2. `nccl_usb4/ibverbs/libtbrxe-rdmav59.so`, `nccl_usb4/ibverbs/libtbrxe-rdmav34.so`, and `nccl_usb4/ibverbs/tbrxe.driver` — userspace `libibverbs` providers (`0.2.73`) for `rdma-core` v63 (`rdmav59`, NGC containers) and v50 (`rdmav34`, Ubuntu 24.04)
3. `nccl_usb4.pth` — startup hook that invokes `nccl_usb4.activate()` at Python interpreter initialization:
   - Detects the runtime `libibverbs.so.1` provider ABI (`verbs_register_driver_59` or `verbs_register_driver_34`) and preloads the matching `libtbrxe-rdmav<PABI>.so` with `RTLD_GLOBAL`. When `/etc/libibverbs.d` and `/usr/lib/x86_64-linux-gnu` are writable (e.g. during `docker build`), installs `tbrxe.driver`, both `libtbrxe-rdmav*.so` providers, `/etc/appmana/usb4-rdma-provider-version`, and symlinks `/usr/lib/x86_64-linux-gnu/libnccl.so*` to the bundled `libnccl.so.2`.
   - Places `<site-packages>/nccl_usb4` at `sys.path[0]` and preloads `nccl_usb4/nccl/lib/libnccl.so.2` with `RTLD_GLOBAL` so `torch._preload_cuda_deps()` resolves `nccl_usb4` before `site-packages/nvidia/nccl/lib/libnccl.so.2`.
   - Sets `VLLM_NCCL_SO_PATH` (if unset) to the bundled `libnccl.so.2`.

## Building and Testing

```sh
# 1. Run the 57 hardware-free routing unit tests
cmake -S rdma-routing -B rdma-routing/build
cmake --build rdma-routing/build -j
ctest --test-dir rdma-routing/build --output-on-failure

# 2. Build libnccl.so.2.32.3
make -j16 src.build BUILDDIR="$PWD/build-2.32.3" \
  NVCC_GENCODE="-gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_121,code=sm_121 -gencode=arch=compute_121,code=compute_121"

# 3. Package the nccl-usb4 wheel and run Python verification tests
python3 python/build_wheel.py --nccl-so build-2.32.3/lib/libnccl.so.2.32.3
pip install --no-deps --force-reinstall dist/nccl_usb4-2.32.3-py3-none-manylinux_2_28_x86_64.whl
pytest python/tests/test_nccl_usb4.py
```

## Runtime Environment Variables

| Variable | Typical Value | Purpose |
| :--- | :--- | :--- |
| `NCCL_IB_HCA` | `usb4_rdma,rxe_lan` | Include both `usb4_rdma*` rails and the `rxe_lan` fallback |
| `NCCL_IB_ADDR_FAMILY` | `AF_INET6` | Match per-link IPv6 ULA `/64` GIDs |
| `NCCL_IB_MERGE_NICS` | `0` | Prevent NCCL from merging distinct rails into a single virtual NIC |
| `NCCL_IB_SUBNET_AWARE_ROUTING` | `prefer_hca[usb4_rdma\d*,rxe_lan\d*]` | Prefer a `/64`-matching `usb4_rdma*` rail; fall back to `rxe_lan` for non-adjacent peers |
| `NCCL_NODE_TO_NODE_TOPO_FILE` | `/etc/nccl/node-topo.json` | Optional explicit inter-node adjacency graph (takes precedence over GID sensing) |
| `NCCL_NET_PLUGIN` | `none` | Disable external net plugins (set automatically by `nccl_usb4.activate()`) |
