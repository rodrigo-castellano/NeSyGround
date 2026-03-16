"""Lightweight baseline utilities for grounder regression tests."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]


def utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _run_capture(cmd: list[str], cwd: Path | None = None) -> str:
    out = subprocess.run(
        cmd, cwd=str(cwd) if cwd else None,
        check=False, capture_output=True, text=True,
    )
    return out.stdout.strip()


def git_info(repo_root: Path = ROOT) -> dict[str, Any]:
    sha = _run_capture(["git", "rev-parse", "HEAD"], cwd=repo_root)
    status = _run_capture(["git", "status", "--porcelain"], cwd=repo_root)
    return {
        "repo_root": str(repo_root),
        "git_sha": sha if sha else None,
        "git_dirty": bool(status),
    }


def runtime_env() -> dict[str, Any]:
    gpu_name = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "unknown"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_gpu_0": gpu_name,
    }


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def fingerprint_paths(paths: list[Path]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in paths:
        k = str(p)
        if not p.exists():
            out[k] = {"exists": False}
            continue
        stat = p.stat()
        out[k] = {
            "exists": True,
            "size_bytes": int(stat.st_size),
            "mtime_epoch_s": int(stat.st_mtime),
            "sha256": _sha256_file(p),
        }
    return out


def canonicalize_dict(d: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(d, sort_keys=True, default=str))


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
