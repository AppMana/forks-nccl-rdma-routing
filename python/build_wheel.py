#!/usr/bin/env python3
"""Build the standalone ``nccl-usb4`` wheel.

Bundles:
  - ``nccl_usb4/__init__.py``
  - ``nccl_usb4/nccl/lib/libnccl.so.2`` (routed NCCL 2.32.3 from forks-nccl-rdma-routing)
  - ``nccl_usb4/ibverbs/libtbrxe-rdmav34.so`` (userspace libibverbs provider for thunderbolt_frame_rxe)
  - ``nccl_usb4/ibverbs/tbrxe.driver``
  - ``nccl_usb4.pth`` (auto-activates at Python startup before torch/vllm import)
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

PACKAGE_NAME = "nccl_usb4"
DIST_NAME = "nccl-usb4"
VERSION = "2.32.3"
WHEEL_TAG = "py3-none-manylinux_2_28_x86_64"
PTH_FILENAME = "nccl_usb4.pth"
PTH_CONTENT = "import nccl_usb4; nccl_usb4.activate()\n"


def _record_line(arcname: str, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
    return f"{arcname},sha256={digest},{len(data)}"


def _write_member(zf: zipfile.ZipFile, arcname: str, data: bytes, mode: int = 0o644) -> str:
    info = zipfile.ZipInfo(arcname)
    info.date_time = (2026, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (mode & 0xFFFF) << 16
    zf.writestr(info, data)
    return _record_line(arcname, data)


def _strip_shared_library(src_path: Path) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        shutil.copy2(src_path, tmp_path)
        subprocess.run(["strip", "--strip-unneeded", str(tmp_path)], check=True)
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


def build_wheel(
    *,
    nccl_so: Path,
    ibverbs_dir: Path,
    out_dir: Path,
    version: str = VERSION,
) -> Path:
    if not nccl_so.is_file():
        raise FileNotFoundError(f"NCCL shared library not found: {nccl_so}")

    provider_paths = sorted(ibverbs_dir.glob("libtbrxe-rdmav*.so"))
    if not provider_paths:
        raise FileNotFoundError(f"No libtbrxe-rdmav*.so providers found in {ibverbs_dir}")

    init_py = Path(__file__).resolve().parent / PACKAGE_NAME / "__init__.py"
    if not init_py.is_file():
        raise FileNotFoundError(f"Package __init__.py not found: {init_py}")

    out_dir.mkdir(parents=True, exist_ok=True)
    wheel_filename = f"{PACKAGE_NAME}-{version}-{WHEEL_TAG}.whl"
    wheel_path = out_dir / wheel_filename
    dist_info = f"{PACKAGE_NAME}-{version}.dist-info"

    nccl_bytes = _strip_shared_library(nccl_so)

    records: list[str] = []
    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. Python module
        records.append(
            _write_member(zf, f"{PACKAGE_NAME}/__init__.py", init_py.read_bytes(), mode=0o644)
        )

        # 2. Routed libnccl.so.2 inside <pkg>/nccl/lib/libnccl.so.2 so
        #    torch._get_cuda_dep_paths(sys.path[0], "nccl", "libnccl.so.*[0-9]")
        #    discovers it natively before site-packages/nvidia/nccl/lib/libnccl.so.2.
        records.append(
            _write_member(zf, f"{PACKAGE_NAME}/nccl/lib/libnccl.so.2", nccl_bytes, mode=0o755)
        )

        # 3. Userspace tbrxe libibverbs providers for thunderbolt_frame_rxe
        #    (rdmav59 for NGC rdma-core 63.0, rdmav34 for Ubuntu 24.04 rdma-core 50.0)
        for provider_path in provider_paths:
            records.append(
                _write_member(
                    zf,
                    f"{PACKAGE_NAME}/ibverbs/{provider_path.name}",
                    _strip_shared_library(provider_path),
                    mode=0o755,
                )
            )
        records.append(
            _write_member(
                zf,
                f"{PACKAGE_NAME}/ibverbs/tbrxe.driver",
                b"driver tbrxe\n",
                mode=0o644,
            )
        )

        # 4. Startup .pth hook in site-packages root
        records.append(
            _write_member(zf, PTH_FILENAME, PTH_CONTENT.encode("utf-8"), mode=0o644)
        )

        # 5. PEP 427 dist-info metadata
        metadata = (
            "Metadata-Version: 2.1\n"
            f"Name: {DIST_NAME}\n"
            f"Version: {version}\n"
            f"Summary: Single-package enabler for routed NCCL {version} and tbrxe libibverbs provider over USB4/Thunderbolt RDMA\n"
            "License: BSD-3-Clause\n"
            "Requires-Python: >=3.10\n"
        ).encode("utf-8")
        wheel_meta = (
            "Wheel-Version: 1.0\n"
            "Generator: nccl-usb4 build_wheel.py\n"
            "Root-Is-Purelib: false\n"
            f"Tag: {WHEEL_TAG}\n"
        ).encode("utf-8")
        top_level = f"{PACKAGE_NAME}\n".encode("utf-8")

        records.append(_write_member(zf, f"{dist_info}/METADATA", metadata, mode=0o644))
        records.append(_write_member(zf, f"{dist_info}/WHEEL", wheel_meta, mode=0o644))
        records.append(_write_member(zf, f"{dist_info}/top_level.txt", top_level, mode=0o644))

        record_arcname = f"{dist_info}/RECORD"
        records.append(f"{record_arcname},,")
        record_bytes = ("\n".join(records) + "\n").encode("utf-8")
        _write_member(zf, record_arcname, record_bytes, mode=0o644)

    return wheel_path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Build nccl-usb4 wheel")
    parser.add_argument(
        "--nccl-so",
        type=Path,
        default=repo_root / "build-2.32.3" / "lib" / f"libnccl.so.{VERSION}",
        help=f"Path to built libnccl.so.{VERSION}",
    )
    parser.add_argument(
        "--ibverbs-dir",
        type=Path,
        default=(
            repo_root / "build-ibverbs"
            if (repo_root / "build-ibverbs").is_dir()
            else Path("/usr/lib/x86_64-linux-gnu/libibverbs")
        ),
        help="Directory containing libtbrxe-rdmav*.so providers",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "dist",
        help="Output directory for built wheel",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=VERSION,
        help="Wheel version",
    )
    args = parser.parse_args()
    wheel_path = build_wheel(
        nccl_so=args.nccl_so,
        ibverbs_dir=args.ibverbs_dir,
        out_dir=args.out_dir,
        version=args.version,
    )
    print(f"Built wheel: {wheel_path} ({wheel_path.stat().st_size / (1024 * 1024):.2f} MiB)")


if __name__ == "__main__":
    main()
