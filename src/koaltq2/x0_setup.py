"""X0: environment check, bundle reassembly from <10MB parts, hash verification, data contract checks."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

CONTENT = Path(os.environ.get("KOALTQ_CONTENT", "/content"))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gpu_info() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"nvidia-smi unavailable: {e}"


def reassemble() -> Path:
    spec = json.loads((CONTENT / "code" / "parts.json").read_text())
    whole = CONTENT / "koaltq_bundle.zip"
    if whole.exists() and whole.stat().st_size == spec["zip_bytes"] and sha256_file(whole) == spec["zip_sha256"]:
        return whole  # uploaded as one file (native file dialog path)
    if whole.exists():
        print(f"koaltq_bundle.zip present but incomplete/mismatched ({whole.stat().st_size} bytes); trying parts")
    up = CONTENT / "upload"
    up.mkdir(exist_ok=True)
    for p in spec["parts"]:
        cands = [Path(c) for c in glob.glob(str(CONTENT / f"{p['name']}*")) + glob.glob(str(up / p["name"]))]
        src = next((c for c in cands if c.name.startswith(p["name"])), None)
        assert src is not None, f"missing part {p['name']}"
        if src.parent != up:
            shutil.move(str(src), up / p["name"])
        got = up / p["name"]
        assert got.stat().st_size == p["bytes"], f"{p['name']}: size {got.stat().st_size} != {p['bytes']} (upload incomplete?)"
        assert sha256_file(got) == p["sha256"], f"{p['name']}: sha256 mismatch"
    zpath = CONTENT / "koaltq_bundle.zip"
    with open(zpath, "wb") as out:
        for p in spec["parts"]:
            with open(up / p["name"], "rb") as f:
                shutil.copyfileobj(f, out)
    assert sha256_file(zpath) == spec["zip_sha256"], "full zip sha256 mismatch"
    return zpath


def main() -> None:
    print("python", sys.version.split()[0], platform.platform())
    print("gpu", gpu_info())
    try:
        import torch
        print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
    except Exception as e:  # noqa: BLE001
        print("torch import failed", e)
    bundle = CONTENT / "koaltq_bundle"
    if not (bundle / "MANIFEST.json").exists():
        z = reassemble()
        with zipfile.ZipFile(z) as zf:
            zf.extractall(CONTENT)
        print("bundle extracted:", z.stat().st_size, "bytes")
    manifest = json.loads((bundle / "MANIFEST.json").read_text())
    bad = [rel for rel, h in manifest["files"].items() if sha256_file(bundle / rel) != h]
    assert not bad, f"{len(bad)} files fail sha256, e.g. {bad[:3]}"
    print("manifest ok:", len(manifest["files"]), "files", manifest["counts"])
    sys.path.insert(0, str(CONTENT / "code"))
    from koaltq2.common import Data, test_shift_weights

    d = Data(bundle)
    w = test_shift_weights(d)
    print(f"data ok: train {d.n_train} test {len(d.test)} fold0 {(d.folds == 0).sum()} boundary_test {len(d.boundary_test)}"
          f" | test-shift weights mean {w.mean():.2f} max {w.max():.2f}")


if __name__ == "__main__":
    main()
