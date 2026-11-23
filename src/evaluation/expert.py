"""Expert evaluation of explanation quality (thesis Section 3.3.3).

Raters score each explanation on Correctness, Completeness and Traceability (integers 1 to 5).

Step 1 (``prepare``): sample up to ``per_category`` explanations of the real system output per
predicted verdict category (fewer when a category has fewer; the shortfall is recorded) and
write one blind rating sheet per rater. A sheet shows the explanation, a policy snippet and
the cited evidence, never the ground-truth label.

Step 2 (``ingest``): read the completed sheets of at least two raters, refuse missing or
out-of-range ratings, and compute mean and standard deviation per dimension (sample standard
deviation, ddof = 1) and Krippendorff's alpha with the interval metric.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.data_pipeline.constants import SEED, VERDICTS
from src.data_pipeline.io_utils import DataError, PathLike

DIMENSIONS = ("correctness", "completeness", "traceability")
SCALE = (1, 5)
SHEET_COLUMNS = (
    "item_id",
    "rater_id",
    "explanation",
    "policy_snippet",
    "cited_evidence",
    "correctness",
    "completeness",
    "traceability",
    "comments",
)


# --------------------------------------------------------------------------- Krippendorff


def krippendorff_alpha(units: Sequence[Sequence[Optional[float]]], level: str = "interval") -> Optional[float]:
    """Krippendorff's alpha for a units x raters table; ``None`` or NaN marks a missing rating.

    alpha = 1 - D_o / D_e with the coincidence-based disagreements

        D_o = (1 / n) * sum over units of 1 / (m_u - 1) * sum over ordered value pairs in the unit of delta
        D_e = 1 / (n (n - 1)) * sum over ordered pairs of all values of delta

    where only units with at least two ratings are pairable and n counts their ratings. The
    interval metric uses delta = (a - b)^2, the nominal metric 0 for equal and 1 for different
    values. Returns ``None`` when alpha is undefined (no pairable data or no variation at all).
    """
    if level not in ("interval", "nominal"):
        raise ValueError("level must be 'interval' or 'nominal'")

    def delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a - b) ** 2 if level == "interval" else (a != b).astype(float)

    pairable: List[np.ndarray] = []
    for row in units:
        values = np.array([float(x) for x in row if x is not None and not math.isnan(float(x))])
        if len(values) >= 2:
            pairable.append(values)
    n = sum(len(v) for v in pairable)
    if n < 2:
        return None

    observed = 0.0
    for values in pairable:
        m = len(values)
        pair_sum = delta(values[:, None], values[None, :]).sum()  # diagonal terms are zero
        observed += pair_sum / (m - 1)
    d_o = observed / n

    every = np.concatenate(pairable)
    d_e = delta(every[:, None], every[None, :]).sum() / (n * (n - 1))
    if d_e == 0:
        return None
    return float(1.0 - d_o / d_e)


# --------------------------------------------------------------------------- sampling


def snippet_for(policy_text: str, commitment_text: Optional[str], window: int = 400, fallback: int = 800) -> str:
    """Policy excerpt around the extracted commitment, or the start of the policy when none was found."""
    text = policy_text or ""
    if commitment_text:
        position = text.find(commitment_text)
        if position >= 0:
            start = max(0, position - window)
            end = min(len(text), position + len(commitment_text) + window)
            return ("..." if start > 0 else "") + text[start:end] + ("..." if end < len(text) else "")
    return text[:fallback] + ("..." if len(text) > fallback else "")


def select_items(
    predictions: Sequence[Mapping[str, Any]],
    policies: Mapping[str, str],
    *,
    system: str,
    per_category: int = 6,
    seed: int = SEED,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Choose explanations per predicted verdict category and report how many were available."""
    rows = [r for r in predictions if r.get("system") == system]
    if not rows:
        raise DataError(f"predictions.jsonl contains no row for system {system!r}")
    rng = random.Random(seed)
    chosen: List[Dict[str, Any]] = []
    availability: Dict[str, Any] = {}
    for label in VERDICTS:
        pool = sorted(
            (r for r in rows if r["verdict"] == label and isinstance(r.get("explanation"), str) and r["explanation"].strip()),
            key=lambda r: r["package_name"],
        )
        take = min(per_category, len(pool))
        availability[label] = {
            "requested": per_category,
            "available": len(pool),
            "sampled": take,
            "shortfall": per_category - take,
        }
        for row in rng.sample(pool, take):
            if row["package_name"] not in policies:
                raise DataError(f"No policy text for {row['package_name']}; pass the test ground-truth file")
            evidence = row.get("evidence_trail") or []
            chosen.append(
                {
                    "package_name": row["package_name"],
                    "system": system,
                    "predicted_verdict": label,
                    "explanation": row["explanation"].strip(),
                    "policy_snippet": snippet_for(policies[row["package_name"]], (row.get("commitment") or {}).get("commitment_text")),
                    "cited_evidence": "\n".join(str(e) for e in evidence) if evidence else "(no evidence trail recorded)",
                }
            )
    if not chosen:
        raise DataError("No explanation could be sampled")
    rng.shuffle(chosen)
    for number, item in enumerate(chosen, start=1):
        item["item_id"] = f"E{number:03d}"
    return chosen, availability


def write_rating_sheets(
    items: Sequence[Mapping[str, Any]], out_dir: Path, n_raters: int, seed: int = SEED
) -> List[Path]:
    """One blind CSV per rater, each in its own item order (UTF-8 with BOM for spreadsheet programs)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(1, n_raters + 1):
        order = list(items)
        random.Random(seed + index).shuffle(order)
        path = out_dir / f"rating_sheet_rater{index}.csv"
        with open(path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
            writer.writeheader()
            for item in order:
                writer.writerow(
                    {
                        "item_id": item["item_id"],
                        "rater_id": f"rater{index}",
                        "explanation": item["explanation"],
                        "policy_snippet": item["policy_snippet"],
                        "cited_evidence": item["cited_evidence"],
                        "correctness": "",
                        "completeness": "",
                        "traceability": "",
                        "comments": "",
                    }
                )
        paths.append(path)
    return paths


# --------------------------------------------------------------------------- ingest


def _parse_rating(raw: str) -> Optional[int]:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        number = float(raw.replace(",", "."))
    except ValueError:
        return None
    if not number.is_integer():
        return None
    return int(number)


def read_sheet(path: PathLike, expected_items: Sequence[str]) -> Tuple[str, Dict[str, Dict[str, int]], List[str]]:
    """Parse one completed sheet; return ``(rater_id, ratings, problems)``."""
    path = Path(path)
    if not path.is_file():
        raise DataError(f"Rating sheet not found: {path}")
    problems: List[str] = []
    ratings: Dict[str, Dict[str, int]] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        absent = [c for c in ("item_id", "rater_id", *DIMENSIONS) if c not in (reader.fieldnames or [])]
        if absent:
            raise DataError(f"{path.name}: missing column(s) {absent}")
        rater_ids = set()
        for line, row in enumerate(reader, start=2):
            item = (row.get("item_id") or "").strip()
            rater_ids.add((row.get("rater_id") or "").strip())
            if item in ratings:
                problems.append(f"{path.name}:{line}: duplicate item {item}")
                continue
            scores: Dict[str, int] = {}
            for dim in DIMENSIONS:
                value = _parse_rating(row.get(dim, ""))
                if value is None:
                    problems.append(f"{path.name}:{line}: {item} has a missing or non-integer {dim} rating {row.get(dim)!r}")
                elif not SCALE[0] <= value <= SCALE[1]:
                    problems.append(f"{path.name}:{line}: {item} {dim} rating {value} is outside {SCALE[0]}-{SCALE[1]}")
                else:
                    scores[dim] = value
            ratings[item] = scores
    if len(rater_ids) != 1 or "" in rater_ids:
        raise DataError(f"{path.name}: the rater_id column must hold one non-empty identifier, found {sorted(rater_ids)}")
    missing = sorted(set(expected_items) - set(ratings))
    unknown = sorted(set(ratings) - set(expected_items))
    if missing:
        problems.append(f"{path.name}: {len(missing)} item(s) not rated, e.g. {missing[:3]}")
    if unknown:
        problems.append(f"{path.name}: unknown item(s) {unknown[:3]}")
    return rater_ids.pop(), ratings, problems


def summarise_ratings(
    ratings: Mapping[str, Mapping[str, Mapping[str, int]]], items: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    """Descriptive statistics and agreement from ``ratings[item][rater][dimension]``."""
    item_ids = sorted(ratings)
    rater_ids = sorted({r for per_item in ratings.values() for r in per_item})
    dimensions: Dict[str, Any] = {}
    pooled_units: List[List[Optional[float]]] = []
    for dim in DIMENSIONS:
        table = [[ratings[i].get(r, {}).get(dim) for r in rater_ids] for i in item_ids]
        flat = np.array([v for row in table for v in row if v is not None], dtype=float)
        dimensions[dim] = {
            "n_ratings": int(flat.size),
            "mean": float(flat.mean()),
            "std": float(flat.std(ddof=1)) if flat.size > 1 else None,
            "per_rater_mean": {
                r: float(np.mean([ratings[i][r][dim] for i in item_ids if r in ratings[i]])) for r in rater_ids
            },
            "krippendorff_alpha_interval": krippendorff_alpha(table, "interval"),
        }
        pooled_units.extend(table)
    every = np.array([v for row in pooled_units for v in row if v is not None], dtype=float)
    per_category: Dict[str, Any] = {}
    for label in VERDICTS:
        members = [i for i in item_ids if items[i]["predicted_verdict"] == label]
        if members:
            per_category[label] = {
                "n_items": len(members),
                **{
                    dim: float(np.mean([ratings[i][r][dim] for i in members for r in ratings[i]]))
                    for dim in DIMENSIONS
                },
            }
    return {
        "n_items": len(item_ids),
        "n_raters": len(rater_ids),
        "rater_ids": rater_ids,
        "scale": list(SCALE),
        "std_definition": "sample standard deviation (ddof=1) over all individual ratings",
        "dimensions": dimensions,
        "overall": {
            "mean": float(every.mean()),
            "std": float(every.std(ddof=1)),
            "krippendorff_alpha_interval": krippendorff_alpha(pooled_units, "interval"),
        },
        "per_predicted_verdict": per_category,
    }


def ingest_ratings(
    sheet_paths: Sequence[PathLike], sample_manifest: Mapping[str, Any], *, min_raters: int = 2
) -> Dict[str, Any]:
    """Validate and summarise the completed sheets; raise :class:`DataError` on any problem."""
    items = {entry["item_id"]: entry for entry in sample_manifest["items"]}
    expected = sorted(items)
    if len(sheet_paths) < min_raters:
        raise DataError(f"At least {min_raters} completed rating sheets are required, got {len(sheet_paths)}")
    all_problems: List[str] = []
    per_rater: Dict[str, Dict[str, Dict[str, int]]] = {}
    for path in sheet_paths:
        rater, ratings, problems = read_sheet(path, expected)
        all_problems.extend(problems)
        if rater in per_rater:
            raise DataError(f"rater_id {rater!r} occurs in more than one sheet")
        per_rater[rater] = ratings
    if all_problems:
        shown = all_problems[:20]
        raise DataError(
            f"Ratings refused: {len(all_problems)} problem(s):\n  " + "\n  ".join(shown)
            + ("\n  ..." if len(all_problems) > 20 else "")
        )
    ratings = {item: {rater: per_rater[rater][item] for rater in per_rater} for item in expected}
    summary = summarise_ratings(ratings, items)
    summary["sampling"] = sample_manifest.get("availability")
    summary["items"] = {
        i: {"package_name": e["package_name"], "predicted_verdict": e["predicted_verdict"], "system": e["system"]}
        for i, e in items.items()
    }
    summary["ratings"] = ratings
    return summary
