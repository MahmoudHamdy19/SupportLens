"""Stratified random sample of the PyPI dataset (thesis Section 3.3.1).

The thesis asks for "a stratified random sample of 300 policies" but does not name
the strata. Interpretation used here (documented in the sampling metadata):

    stratum = (stars tercile) x (package has at least one vulnerability record)

* The stars terciles are computed on the eligible population with the 1/3 and 2/3
  quantiles; a record belongs to tercile T1 (stars <= q1), T2 (q1 < stars <= q2)
  or T3 (stars > q2). Records without a star count form their own stratum.
* The sample size is allocated proportionally to stratum size (largest remainder).
* Within a stratum, records are drawn without replacement with ``random.Random(42)``
  after sorting the members by package name, so the result does not depend on the
  order of the input file.

Stars measure popularity and the vulnerability flag separates packages for which the
verdict can be Consistent, Partially Consistent or Potential Inconsistency from those
that are expected to end in Insufficient Evidence. Stratifying on both keeps each
group represented in the annotation effort.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .constants import SEED
from .io_utils import DataError

TERCILE_LABELS = ("T1_low", "T2_mid", "T3_high")


class InsufficientDataError(DataError):
    """Fewer eligible records than the requested sample size."""


def star_tercile_cutoffs(stars: Sequence[int]) -> Tuple[float, float]:
    """1/3 and 2/3 quantiles of the star counts (linear interpolation)."""
    if not stars:
        raise DataError("Cannot compute star terciles: no record has a star count.")
    q1, q2 = np.quantile(np.asarray(stars, dtype=float), [1 / 3, 2 / 3])
    return float(q1), float(q2)


def tercile_label(stars: Optional[int], cutoffs: Optional[Tuple[float, float]]) -> str:
    if stars is None or cutoffs is None:
        return "stars_unknown"
    q1, q2 = cutoffs
    if stars <= q1:
        return TERCILE_LABELS[0]
    if stars <= q2:
        return TERCILE_LABELS[1]
    return TERCILE_LABELS[2]


def has_vulnerability(record: Mapping[str, Any]) -> bool:
    return len(record.get("vulnerabilities") or []) > 0


def stratum_of(record: Mapping[str, Any], cutoffs: Optional[Tuple[float, float]]) -> str:
    stars = record.get("stars")
    stars = stars if isinstance(stars, int) and not isinstance(stars, bool) else None
    return f"{tercile_label(stars, cutoffs)}|{'vuln' if has_vulnerability(record) else 'no_vuln'}"


def allocate_proportionally(sizes: Mapping[str, int], total: int) -> Dict[str, int]:
    """Largest-remainder allocation of ``total`` draws over strata, capped by stratum size."""
    population = sum(sizes.values())
    if total > population:
        raise InsufficientDataError(f"Cannot allocate {total} draws from {population} records.")
    raw = {key: total * size / population for key, size in sizes.items()}
    allocation = {key: min(int(math.floor(value)), sizes[key]) for key, value in raw.items()}
    remaining = total - sum(allocation.values())
    order = sorted(sizes, key=lambda key: (-(raw[key] - math.floor(raw[key])), key))
    while remaining > 0:
        progressed = False
        for key in order:
            if remaining == 0:
                break
            if allocation[key] < sizes[key]:
                allocation[key] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise InsufficientDataError("Allocation failed: strata are exhausted.")
    return allocation


def stratified_sample(
    records: Sequence[Mapping[str, Any]],
    size: int = 300,
    *,
    seed: int = SEED,
    allow_smaller: bool = False,
) -> Tuple[List[dict], Dict[str, Any]]:
    """Return ``(sampled_records, metadata)``.

    Raises :class:`InsufficientDataError` when fewer than ``size`` records exist,
    unless ``allow_smaller`` is set, in which case every record is returned and the
    metadata states the real size.
    """
    names = [r.get("package_name") for r in records]
    if any(not isinstance(n, str) or not n for n in names):
        raise DataError("Every record needs a non-empty package_name.")
    duplicates = [n for n, c in Counter(names).items() if c > 1]
    if duplicates:
        raise DataError(f"Duplicate package names in the dataset: {sorted(duplicates)[:5]}")

    available = len(records)
    if available < size and not allow_smaller:
        raise InsufficientDataError(
            f"Only {available} eligible records are available; {size} are required "
            "(thesis Section 3.3.1). Collect more packages or pass --allow-smaller."
        )
    target = min(size, available)

    star_values = [r["stars"] for r in records if isinstance(r.get("stars"), int) and not isinstance(r.get("stars"), bool)]
    cutoffs = star_tercile_cutoffs(star_values) if star_values else None

    strata: Dict[str, List[dict]] = {}
    for record in sorted(records, key=lambda r: r["package_name"]):
        strata.setdefault(stratum_of(record, cutoffs), []).append(dict(record))

    allocation = allocate_proportionally({k: len(v) for k, v in strata.items()}, target)

    rng = random.Random(seed)
    sampled: List[dict] = []
    for key in sorted(strata):
        members = strata[key]
        chosen = rng.sample(members, allocation[key]) if allocation[key] < len(members) else list(members)
        for record in chosen:
            record["sampling_stratum"] = key
            sampled.append(record)
    sampled.sort(key=lambda r: r["package_name"])

    metadata: Dict[str, Any] = {
        "requested_size": size,
        "realised_size": len(sampled),
        "allow_smaller": allow_smaller,
        "shortfall": size - len(sampled),
        "seed": seed,
        "eligible_population": available,
        "design": "stars tercile x has-vulnerability, proportional allocation (largest remainder)",
        "interpretation_note": (
            "The thesis does not name the strata; this design is an interpretation of "
            "'stratified random sample' (Section 3.3.1)."
        ),
        "star_tercile_cutoffs": list(cutoffs) if cutoffs else None,
        "population_by_stratum": {k: len(v) for k, v in sorted(strata.items())},
        "sample_by_stratum": dict(sorted(Counter(r["sampling_stratum"] for r in sampled).items())),
    }
    return sampled, metadata
