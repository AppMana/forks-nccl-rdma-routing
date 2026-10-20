"""Single-package enabler for routed NCCL 2.30.7 over USB4/Thunderbolt IB RDMA.

Installing ``nccl-usb4`` drops ``nccl_usb4.pth`` into ``site-packages``, which
calls :func:`activate` at interpreter startup before ``torch`` or ``vllm`` is
imported:

1. Loads ``libibverbs.so.1`` and the bundled ``tbrxe`` / ``usb4_rdma`` userspace
   providers (``libtbrxe-rdmav34.so`` and ``libusb4_rdma-rdmav34.so``) with
   ``RTLD_GLOBAL`` so their ELF constructors register the ``0x55534234``
   (``"USB4"``) driver ID directly with ``libibverbs`` in memory. When running
   with write access to ``/etc/libibverbs.d`` (e.g. inside containers), also
   populates ``/etc/libibverbs.d/tbrxe.driver`` and
   ``/etc/appmana/usb4-rdma-provider-version`` so standalone C verbs tools and
   preflight checks work without any ``.deb`` package.
2. Places ``<site-packages>/nccl_usb4`` at ``sys.path[0]`` (which contains
   ``nccl/lib/libnccl.so.2``) and preloads ``libnccl.so.2`` with
   ``RTLD_GLOBAL``. When ``import torch`` runs ``_preload_cuda_deps()``, its
   ``_get_cuda_dep_paths(path, "nccl", "libnccl.so.*[0-9]")`` loop matches
   ``sys.path[0]/nccl/lib/libnccl.so.2`` first and breaks before reaching
   ``site-packages/nvidia/nccl/lib/libnccl.so.2``, so stock ``nvidia-nccl-cu*``
   remains untouched on disk and is never loaded into the process.
3. Sets ``VLLM_NCCL_SO_PATH`` (if unset) to the bundled ``libnccl.so.2`` so
   vLLM's ``PyNcclCommunicator`` binds to the exact same library, and sets
   ``NCCL_NET_PLUGIN=none`` (if unset) so external plugins do not bypass
   built-in ``net_ib`` rail routing.
"""
from __future__ import annotations

import ctypes
import os
import platform
import sys
from pathlib import Path
from typing import Any

__version__ = "2.30.7"
NCCL_GIT_REF = "1eec1c2e173877870b7dca4738305a2430aae8d2"
USB4_RDMA_PROVIDER_VERSION = "0.2.73"

_ACTIVATED = False
_LOADED_NCCL_PATH: str | None = None
_LOADED_IBVERBS_PROVIDERS: list[str] = []


def package_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def nccl_lib_path() -> str:
    return os.path.join(package_root(), "nccl", "lib", "libnccl.so.2")


def ibverbs_provider_paths() -> list[str]:
    ibv_dir = os.path.join(package_root(), "ibverbs")
    names = ("libtbrxe-rdmav34.so", "libusb4_rdma-rdmav34.so")
    return [
        os.path.join(ibv_dir, name)
        for name in names
        if os.path.isfile(os.path.join(ibv_dir, name))
    ]


def _install_system_ibverbs_files_best_effort() -> None:
    """If running as root in a container without the deb, populate /etc/libibverbs.d
    and /usr/lib/x86_64-linux-gnu/libibverbs so non-Python C tools also see tbrxe."""
    sys_ibv_dir = Path("/usr/lib/x86_64-linux-gnu/libibverbs")
    etc_ibv_dir = Path("/etc/libibverbs.d")
    try:
        if sys_ibv_dir.is_dir() and os.access(sys_ibv_dir, os.W_OK):
            for src_str in ibverbs_provider_paths():
                src = Path(src_str)
                dst = sys_ibv_dir / src.name
                if not dst.exists():
                    try:
                        dst.symlink_to(src)
                    except OSError:
                        pass
        if etc_ibv_dir.is_dir() and os.access(etc_ibv_dir, os.W_OK):
            for drv_name in ("tbrxe", "usb4_rdma"):
                drv_file = etc_ibv_dir / f"{drv_name}.driver"
                if not drv_file.exists():
                    drv_file.write_text(f"driver {drv_name}\n", encoding="utf-8")
        stamp = Path("/etc/appmana/usb4-rdma-provider-version")
        if not stamp.exists() and os.access("/etc", os.W_OK):
            stamp.parent.mkdir(parents=True, exist_ok=True)
            stamp.write_text(f"{USB4_RDMA_PROVIDER_VERSION}\n", encoding="utf-8")
    except OSError:
        pass


def activate() -> bool:
    """Activate bundled USB4 RDMA libibverbs providers and routed libnccl.so.2."""
    global _ACTIVATED, _LOADED_NCCL_PATH
    if _ACTIVATED:
        return True
    if os.environ.get("NCCL_USB4_DISABLE", "").strip() in ("1", "true", "TRUE", "yes"):
        return False
    if sys.platform != "linux" or platform.machine() != "x86_64":
        return False

    pkg_dir = package_root()
    so_path = nccl_lib_path()
    if not os.path.isfile(so_path):
        return False

    _install_system_ibverbs_files_best_effort()

    # 1. Preload libibverbs.so.1 and our userspace RDMA providers so their
    #    ELF constructors call verbs_register_driver_34 in-process.
    try:
        ctypes.CDLL("libibverbs.so.1", mode=ctypes.RTLD_GLOBAL)
        for provider_path in ibverbs_provider_paths():
            try:
                ctypes.CDLL(provider_path, mode=ctypes.RTLD_GLOBAL)
                if provider_path not in _LOADED_IBVERBS_PROVIDERS:
                    _LOADED_IBVERBS_PROVIDERS.append(provider_path)
            except OSError:
                pass
    except OSError:
        pass

    # 2. Ensure pkg_dir is at sys.path[0] so torch._preload_cuda_deps() finds
    #    <pkg_dir>/nccl/lib/libnccl.so.2 before site-packages/nvidia/nccl/lib/libnccl.so.2.
    if not sys.path or sys.path[0] != pkg_dir:
        try:
            sys.path.remove(pkg_dir)
        except ValueError:
            pass
        sys.path.insert(0, pkg_dir)

    # 3. Preload our routed libnccl.so.2 with RTLD_GLOBAL so DT_SONAME
    #    ("libnccl.so.2") is satisfied in ld.so before libtorch_cuda.so loads.
    try:
        ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return False

    # 4. Point vLLM's PyNcclCommunicator at the exact same library and disable
    #    external net plugins that would bypass net_ib rail routing.
    os.environ.setdefault("VLLM_NCCL_SO_PATH", so_path)
    os.environ.setdefault("NCCL_NET_PLUGIN", "none")

    _LOADED_NCCL_PATH = so_path
    _ACTIVATED = True
    return True


def loaded_nccl_maps() -> list[str]:
    """Return all libnccl shared object paths currently mapped in /proc/self/maps."""
    if sys.platform != "linux":
        return []
    paths: set[str] = set()
    try:
        with open("/proc/self/maps", encoding="utf-8") as f:
            for line in f:
                if "libnccl" in line:
                    parts = line.strip().split()
                    if parts:
                        paths.add(parts[-1])
    except OSError:
        pass
    return sorted(paths)


def status() -> dict[str, Any]:
    """Return runtime diagnostic metadata for verification and telemetry."""
    return {
        "active": _ACTIVATED,
        "version": __version__,
        "nccl_git_ref": NCCL_GIT_REF,
        "usb4_rdma_provider_version": USB4_RDMA_PROVIDER_VERSION,
        "nccl_so_path": _LOADED_NCCL_PATH or nccl_lib_path(),
        "vllm_nccl_so_path": os.environ.get("VLLM_NCCL_SO_PATH"),
        "nccl_net_plugin": os.environ.get("NCCL_NET_PLUGIN"),
        "ibverbs_providers_loaded": list(_LOADED_IBVERBS_PROVIDERS),
        "loaded_nccl_maps": loaded_nccl_maps(),
    }
