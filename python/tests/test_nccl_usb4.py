from __future__ import annotations

import ctypes
import subprocess
import sys

import nccl_usb4


def test_nccl_usb4_activates_and_exposes_routed_nccl_and_tbrxe() -> None:
    info = nccl_usb4.activate()
    assert info["active"] is True
    assert info["version"] == "2.32.3"
    assert info["nccl_version_code"] == 23203
    assert info["usb4_rdma_provider_version"] == "0.2.73"
    assert info["nccl_net_plugin"] == "none"
    assert info["tbrxe_loaded"] is True
    assert any("libtbrxe-rdmav" in p for p in info["ibverbs_providers_loaded"])
    assert len(nccl_usb4.ibverbs_provider_paths()) >= 2
    assert not any("libusb4_rdma" in p for p in info["ibverbs_providers_loaded"])

    so_path = info["nccl_so_path"]
    with open(so_path, "rb") as f:
        blob = f.read()
    assert b"rail routing:" in blob
    assert b"skipping external" in blob

    lib = ctypes.CDLL(so_path)
    ver = ctypes.c_int()
    assert lib.ncclGetVersion(ctypes.byref(ver)) == 0
    assert ver.value == 23203


def test_sys_path_preempts_torch_preload_cuda_deps() -> None:
    nccl_usb4.activate()
    pkg_idx = sys.path.index(nccl_usb4.package_root())
    site_indices = [i for i, p in enumerate(sys.path) if "site-packages" in p and p != nccl_usb4.package_root()]
    assert all(pkg_idx < i for i in site_indices)


def test_fresh_interpreter_maps_only_nccl_usb4_after_import_torch() -> None:
    code = (
        "import ctypes, torch, nccl_usb4; "
        "ctypes.CDLL(str(torch.__file__).rsplit('/', 1)[0] + '/lib/libtorch_cuda.so', mode=ctypes.RTLD_GLOBAL); "
        "st = nccl_usb4.status(); "
        "assert st['active'], st; "
        "assert st['tbrxe_loaded'], st; "
        "maps = st['loaded_nccl_maps']; "
        "assert len(maps) == 1 and 'nccl_usb4/nccl/lib/libnccl.so.2' in maps[0], maps; "
        "v = ctypes.c_int(); "
        "assert ctypes.CDLL('libnccl.so.2').ncclGetVersion(ctypes.byref(v)) == 0 and v.value == 23203, v.value"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
