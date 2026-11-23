"""Facts about a training run, read from the environment and the files actually used.

Nothing here is hard-coded: versions, GPU and hashes are queried at the time of the run.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional

_LIBRARIES = ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "numpy", "pydantic", "tokenizers")


def library_versions() -> Dict[str, Optional[str]]:
    """Installed version of each library, or None when it is not installed."""
    versions: Dict[str, Optional[str]] = {}
    for name in _LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def hardware_info() -> Dict[str, Any]:
    """GPU name, CUDA version and device count from PyTorch; None entries when CUDA is absent."""
    info: Dict[str, Any] = {"cuda_available": False, "gpu_name": None, "cuda_version": None, "gpu_count": 0}
    try:
        import torch
    except ImportError:
        return info
    info["cuda_version"] = torch.version.cuda
    if torch.cuda.is_available():
        info["cuda_available"] = True
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        properties = torch.cuda.get_device_properties(0)
        info["gpu_memory_gb"] = round(properties.total_memory / 1024**3, 2)
    return info


def environment_info() -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libraries": library_versions(),
        "hardware": hardware_info(),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
