"""Run manifest for reproducibility (thesis Section 3.4.1, Appendix C).

Records the software and hardware environment, the seed, content hashes of the data files
(independent of Git) and the hyper-parameters that the training scripts stored next to the
model weights in ``models/<name>/run_metadata.json``.
"""

from __future__ import annotations

import json
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.data_pipeline.constants import SEED
from src.data_pipeline.io_utils import PROJECT_ROOT, sha256_file, utc_now_iso

LIBRARIES = (
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "torch",
    "transformers",
    "peft",
    "safetensors",
    "pydantic",
    "packaging",
    "requests",
    "jinja2",
    "pytest",
)


def library_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for name in LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def gpu_info() -> Dict[str, Any]:
    """GPU name and CUDA version when PyTorch with CUDA is present; otherwise an explicit negative."""
    try:
        import torch  # noqa: WPS433 (optional heavy dependency)
    except ImportError:
        return {"available": False, "torch": None, "cuda_version": None, "device_name": None}
    info: Dict[str, Any] = {
        "available": bool(torch.cuda.is_available()),
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": None,
    }
    if info["available"]:
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_count"] = torch.cuda.device_count()
    return info


def collect_hyperparameters(models_dir: Path) -> Dict[str, Any]:
    found: Dict[str, Any] = {}
    for path in sorted(Path(models_dir).glob("*/run_metadata.json")):
        try:
            found[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            found[path.parent.name] = {"error": f"unreadable run_metadata.json: {exc}"}
    return found


def file_hashes(files: Mapping[str, Path]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for name, path in files.items():
        path = Path(path)
        if path.is_file():
            try:
                shown = str(path.resolve().relative_to(PROJECT_ROOT))
            except ValueError:
                shown = str(path)
            result[name] = {"path": shown.replace("\\", "/"), "sha256": sha256_file(path)}
    return result


def build_manifest(
    *,
    data_files: Mapping[str, Path],
    models_dir: Optional[Path] = None,
    seed: int = SEED,
    run: Optional[Mapping[str, Any]] = None,
    probe_gpu: bool = True,
) -> Dict[str, Any]:
    """Assemble ``run_manifest.json``; ``run`` carries the evaluation options and outcomes."""
    return {
        "created_at_utc": utc_now_iso(),
        "seed": seed,
        "python": {"version": sys.version.split()[0], "implementation": platform.python_implementation()},
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "libraries": library_versions(),
        "gpu": gpu_info() if probe_gpu else {"available": None, "note": "not probed"},
        "data_files": file_hashes(data_files),
        "hyperparameters": collect_hyperparameters(models_dir or PROJECT_ROOT / "models"),
        "run": dict(run or {}),
    }
