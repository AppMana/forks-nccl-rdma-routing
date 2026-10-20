from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

import nccl_usb4

sys.path.append(str(Path(__file__).resolve().parent.parent))
from build_wheel import build_wheel  # noqa: E402


def test_activate_and_status():
    assert nccl_usb4.activate() is True
    st = nccl_usb4.status()
    assert st["active"] is True
    assert st["version"] == "2.30.7"
    assert st["usb4_rdma_provider_version"] == "0.2.73"
    assert os.path.isfile(st["nccl_so_path"])
    assert st["vllm_nccl_so_path"] == st["nccl_so_path"]
    assert st["nccl_net_plugin"] == "none"
    assert any("nccl_usb4/nccl/lib/libnccl.so.2" in p for p in st["loaded_nccl_maps"])


def test_torch_uses_only_nccl_usb4():
    import torch

    assert torch.cuda.nccl.version() == (2, 30, 7)
    maps = nccl_usb4.loaded_nccl_maps()
    assert len(maps) == 1, f"Expected only nccl_usb4 libnccl.so.2 in /proc/self/maps, got {maps}"
    assert "nccl_usb4/nccl/lib/libnccl.so.2" in maps[0]


def test_build_wheel_structure(tmp_path: Path):
    so_path = Path(nccl_usb4.nccl_lib_path())
    ibv_dir = Path(nccl_usb4.package_root()) / "ibverbs"
    whl = build_wheel(nccl_so=so_path, ibverbs_dir=ibv_dir, out_dir=tmp_path)
    assert whl.is_file()
    with zipfile.ZipFile(whl) as zf:
        names = set(zf.namelist())
        assert "nccl_usb4/__init__.py" in names
        assert "nccl_usb4/nccl/lib/libnccl.so.2" in names
        assert "nccl_usb4/ibverbs/libtbrxe-rdmav34.so" in names
        assert "nccl_usb4.pth" in names
