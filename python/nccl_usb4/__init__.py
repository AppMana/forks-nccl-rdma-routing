"""Single-package enabler for routed NCCL 2.32.3 over USB4/Thunderbolt IB RDMA.

Installing ``nccl-usb4`` drops ``nccl_usb4.pth`` into ``site-packages``, which
calls :func:`activate` at interpreter startup before ``torch`` or ``vllm`` is
imported:

1. Loads ``libibverbs.so.1`` and the bundled ``tbrxe`` userspace provider
   (``libtbrxe-rdmav34.so``) with ``RTLD_GLOBAL`` so its ELF constructor
   registers the ``0x55534234`` (``"USB4"``) driver ID of ``thunderbolt_frame_rxe``
   directly with ``libibverbs`` in memory. When running with write access to
   ``/etc/libibverbs.d`` (e.g. inside containers), also populates
   ``/etc/libibverbs.d/tbrxe.driver``,
   ``/usr/lib/x86_64-linux-gnu/libibverbs/libtbrxe-rdmav34.so``, and
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
import shutil
import sys
from pathlib import Path
from typing import Any

__version__ = "2.32.3"
NCCL_VERSION_CODE = 23203
USB4_RDMA_PROVIDER_VERSION = "0.2.73"

_ACTIVATED = False
_LOADED_NCCL_PATH: str | None = None
_LOADED_IBVERBS_PROVIDERS: list[str] = []


def package_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def nccl_lib_path() -> str:
    return os.path.join(package_root(), "nccl", "lib", "libnccl.so.2")


get_nccl_lib_path = nccl_lib_path


def tbrxe_provider_path(pabi: int = 59) -> str:
    candidate = os.path.join(package_root(), "ibverbs", f"libtbrxe-rdmav{pabi}.so")
    if os.path.isfile(candidate):
        return candidate
    return os.path.join(package_root(), "ibverbs", "libtbrxe-rdmav34.so")


def ibverbs_provider_paths() -> list[str]:
    ibv_dir = Path(package_root()) / "ibverbs"
    if not ibv_dir.is_dir():
        return []
    return sorted(str(p) for p in ibv_dir.glob("libtbrxe-rdmav*.so") if p.is_file())


def _detect_ibverbs_pabi(ibv_lib: ctypes.CDLL) -> int | None:
    for pabi in (59, 34, 53, 49, 44, 39):
        if hasattr(ibv_lib, f"verbs_register_driver_{pabi}"):
            return pabi
    return None


def _install_system_ibverbs_files_best_effort() -> None:
    """If running with write access in a container, populate /etc/libibverbs.d,
    /usr/lib/x86_64-linux-gnu/libibverbs, and /usr/lib/x86_64-linux-gnu/libnccl.so*
    symlinks so non-Python C tools and ld.so also resolve tbrxe and NCCL 2.32.3."""
    sys_lib_dir = Path("/usr/lib/x86_64-linux-gnu")
    sys_ibv_dir = sys_lib_dir / "libibverbs"
    etc_ibv_dir = Path("/etc/libibverbs.d")
    try:
        if sys_ibv_dir.is_dir() and os.access(sys_ibv_dir, os.W_OK):
            for provider_str in ibverbs_provider_paths():
                src = Path(provider_str)
                dst = sys_ibv_dir / src.name
                if not dst.exists():
                    try:
                        shutil.copy2(src, dst)
                    except OSError:
                        pass
        if sys_lib_dir.is_dir() and os.access(sys_lib_dir, os.W_OK):
            so_target = Path(nccl_lib_path())
            if so_target.is_file():
                versioned = sys_lib_dir / f"libnccl.so.{__version__}"
                soname = sys_lib_dir / "libnccl.so.2"
                bare = sys_lib_dir / "libnccl.so"
                for old in sys_lib_dir.glob("libnccl.so.2.*"):
                    if old.name != versioned.name:
                        try:
                            old.unlink()
                        except OSError:
                            pass
                for link, target in (
                    (versioned, str(so_target)),
                    (soname, versioned.name),
                    (bare, soname.name),
                ):
                    try:
                        if link.is_symlink() or link.exists():
                            if link.resolve() == so_target.resolve() and link != versioned:
                                continue
                            link.unlink()
                        link.symlink_to(target)
                    except OSError:
                        pass
        if etc_ibv_dir.is_dir() and os.access(etc_ibv_dir, os.W_OK):
            drv_file = etc_ibv_dir / "tbrxe.driver"
            if not drv_file.exists():
                drv_file.write_text("driver tbrxe\n", encoding="utf-8")
            legacy_drv = etc_ibv_dir / "usb4_rdma.driver"
            if legacy_drv.exists():
                try:
                    legacy_drv.unlink()
                except OSError:
                    pass
        stamp = Path("/etc/appmana/usb4-rdma-provider-version")
        if os.access("/etc", os.W_OK):
            stamp.parent.mkdir(parents=True, exist_ok=True)
            stamp.write_text(f"{USB4_RDMA_PROVIDER_VERSION}\n", encoding="utf-8")
    except OSError:
        pass


def activate() -> dict[str, Any]:
    """Activate bundled tbrxe libibverbs provider and routed libnccl.so.2."""
    global _ACTIVATED, _LOADED_NCCL_PATH
    pkg_dir = package_root()
    if _ACTIVATED:
        if not sys.path or sys.path[0] != pkg_dir:
            try:
                sys.path.remove(pkg_dir)
            except ValueError:
                pass
            sys.path.insert(0, pkg_dir)
        return status()
    if os.environ.get("NCCL_USB4_DISABLE", "").strip() in ("1", "true", "TRUE", "yes"):
        return status()
    if sys.platform != "linux" or platform.machine() != "x86_64":
        return status()

    so_path = nccl_lib_path()
    if not os.path.isfile(so_path):
        return status()

    _install_system_ibverbs_files_best_effort()

    # 1. Preload libibverbs.so.1 and the matching tbrxe userspace RDMA provider
    #    (rdmav59 on NGC rdma-core 63.0, rdmav34 on Ubuntu 24.04 rdma-core 50.0)
    #    so its ELF constructor calls verbs_register_driver_<PABI> in-process.
    try:
        ibv_lib = ctypes.CDLL("libibverbs.so.1", mode=ctypes.RTLD_GLOBAL)
        pabi = _detect_ibverbs_pabi(ibv_lib)
        candidates = (
            [tbrxe_provider_path(pabi)] if pabi is not None else ibverbs_provider_paths()
        )
        for provider_path in candidates:
            if not os.path.isfile(provider_path):
                continue
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
        return status()

    # 4. Point vLLM's PyNcclCommunicator at the exact same library and disable
    #    external net plugins that would bypass net_ib rail routing.
    os.environ.setdefault("VLLM_NCCL_SO_PATH", so_path)
    os.environ.setdefault("NCCL_NET_PLUGIN", "none")

    _LOADED_NCCL_PATH = so_path
    _ACTIVATED = True
    return status()


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
        "nccl_version_code": NCCL_VERSION_CODE,
        "provider_version": USB4_RDMA_PROVIDER_VERSION,
        "usb4_rdma_provider_version": USB4_RDMA_PROVIDER_VERSION,
        "nccl_so_path": _LOADED_NCCL_PATH or nccl_lib_path(),
        "vllm_nccl_so_path": os.environ.get("VLLM_NCCL_SO_PATH"),
        "nccl_net_plugin": os.environ.get("NCCL_NET_PLUGIN"),
        "tbrxe_loaded": bool(_LOADED_IBVERBS_PROVIDERS),
        "ibverbs_providers_loaded": list(_LOADED_IBVERBS_PROVIDERS),
        "loaded_nccl_maps": loaded_nccl_maps(),
    }
