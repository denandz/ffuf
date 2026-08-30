#!/usr/bin/env python3
"""
ffuf audit-log LOF outlier detector.

For each HTTP status code, responses are standard-scaled within the
status group (same feature set and same per-status scaling as
ffuf_analyse.py) and Local Outlier Factor is computed to flag individual
responses whose local density differs from their neighbours -- e.g. a
single 10,100-byte 200 response among 499 identical 10,000-byte ones,
which DBSCAN would absorb into the baseline cluster.

A response is flagged when its LOF score exceeds --threshold (higher
score = more unlike its neighbours; 1.0 means it is in a sparser region
than its neighbours).

Usage:
    python3 ffuf_lof.py <audit-log.json> [more-logs.json ...] [options]

Requires: numpy, scikit-learn
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

import ffuf_analyse as fa

# Same continuous features as ffuf_analyse, plus the overhead feature
# (content_length - payload_len), which carries the reflection signal.
LOF_FEATURES = list(fa.CORE_FEATURES) + ["overhead"]

FEATURE_FMT = {
    "content_length": (fa._fmt_bytes, "len"),
    "content_words": (lambda v: f"{round(v)}", "words"),
    "content_lines": (lambda v: f"{round(v)}", "lines"),
    "duration_us": (fa._fmt_us, "time"),
    "overhead": (fa._fmt_bytes, "o/h"),
}


def lof_group(
    recs: list[dict], k: int, threshold: float
) -> tuple[np.ndarray | None, int, int]:
    """Run LOF on one status group.

    Returns (scores for every record, k used, n distinct) or None when the
    group is too small / too degenerate for LOF.
    """
    X = np.array([[float(r[f]) for f in LOF_FEATURES] for r in recs], dtype=np.float64)
    # Collapse exact duplicates; map uniques -> records
    uniq, inverse = np.unique(X, axis=0, return_inverse=True)
    n_uniq = len(uniq)
    if n_uniq == 1:
        return np.zeros(len(recs)), k, 1
    if n_uniq < 4:
        return (None, 0, n_uniq)
    k = max(2, min(k, n_uniq - 1))

    Xu = StandardScaler().fit_transform(uniq)
    lof = LocalOutlierFactor(n_neighbors=k).fit(Xu)
    # negative_outlier_factor_ is the NEGATED LOF score (outliers are very
    # negative); flip it so higher = more unlike its neighbours.
    uniq_scores = -lof.negative_outlier_factor_
    return uniq_scores[inverse], k, n_uniq


def report_file(
    filepath: str,
    records: list[dict],
    k: int,
    threshold: float,
    min_group: int,
    top: int,
) -> int:
    """Print the LOF report for one log.  Returns total flagged count."""
    groups: dict[int, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        groups[fa._status_group(rec)].append(i)
    status_of = {
        slot: next(iter(r["status"] for r in (records[i] for i in idx)))
        for slot, idx in groups.items()
    }

    flagged_total = 0
    print(f"\n{'=' * 78}")
    print(
        f"  LOF Outliers  -  {Path(filepath).name}   " f"(k={k}, threshold={threshold})"
    )
    print(f"{'=' * 78}")

    for slot in sorted(groups, key=lambda s: -len(groups[s])):
        idx = groups[slot]
        status = status_of[slot]
        if len(idx) < min_group:
            print(
                f"\n  status {status:<6} n={len(idx):<5} "
                f"skipped (n < --min-group {min_group})"
            )
            continue
        recs = [records[i] for i in idx]
        scores, k_used, n_uniq = lof_group(recs, k, threshold)
        if scores is None:
            print(
                f"\n  status {status:<6} n={len(idx):<5} "
                f"skipped (too few distinct responses for LOF)"
            )
            continue

        flags = [
            (recs[j], float(scores[j]))
            for j in range(len(recs))
            if scores[j] > threshold
        ]
        flags.sort(key=lambda t: -t[1])
        flagged_total += len(flags)

        print(
            f"\n  status {status:<6} n={len(idx):<5} distinct={n_uniq:<5} "
            f"k={k_used:<3} flagged={len(flags)}"
        )
        if not flags:
            print(f"    no outliers above threshold {threshold}")
            continue

        # group medians for context
        meds = {f: median([r[f] for r in recs]) for f in LOF_FEATURES}
        shown = flags[:top]
        for rec, score in shown:
            feats = "  ".join(
                f"{label} {fmt(rec[f])}"
                f"{'' if rec[f] == meds[f] else f' (med {fmt(meds[f])})'}"
                for f, (fmt, label) in FEATURE_FMT.items()
            )
            print(f"    score {score:6.2f}  pos {rec['position']:>6}  " f"{feats}")
            print(f"    {' ' * 15}payload: {fa._truncate(rec['payload'], 60)}")
        if len(flags) > top:
            print(f"    … {len(flags) - top} more flagged (lower scores)")
    return flagged_total


def main():
    parser = argparse.ArgumentParser(
        description="LOF outlier detection for ffuf audit logs, "
        "per HTTP status code."
    )
    parser.add_argument("files", nargs="+", help="Audit log JSON file(s)")
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="LOF n_neighbors, clamped to distinct-1 " "(default: 10)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.5,
        help="Flag LOF scores above this value " "(default: 1.5)",
    )
    parser.add_argument(
        "--min-group",
        type=int,
        default=5,
        help="Skip status groups with fewer responses " "(default: 5)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Max flagged responses shown per status " "(default: 5)",
    )
    args = parser.parse_args()

    failed = 0
    for filepath in args.files:
        if not Path(filepath).exists():
            print(f"Error: file not found: {filepath}", file=sys.stderr)
            failed += 1
            continue
        records, skipped, unpaired = fa.parse_log(filepath)

        if skipped or unpaired:
            print(
                f"  note: {filepath}: skipped {skipped} malformed line(s), "
                f"{unpaired} response(s) without request info",
                file=sys.stderr,
            )

        if not records:
            print(f"No valid records in {filepath}", file=sys.stderr)
            failed += 1
            continue

        fa.enrich_records(records)
        n_flagged = report_file(
            filepath, records, args.k, args.threshold, args.min_group, args.top
        )
        print(f"\n  {Path(filepath).name}: {n_flagged} response(s) flagged")
        if n_flagged == 0:
            print("  (nothing to triage)")
        print()

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
