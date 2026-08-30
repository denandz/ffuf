#!/usr/bin/env python3
"""
ffuf audit-log response analyst - cluster-based response profiling.

Reads one or more ffuf JSON audit logs and clusters responses by their
measurable attributes.  An additional feature is added to each response:

  overhead: content_length minus decoded payload length
            (constant for reflective endpoints, variable for static ones)
            (helpful for detecting reflection and XSS)

This means a reflective endpoint collapses into one cluster regardless of
how much the content length varies, while distinct response types (500s,
slow testcases, etc.) still separate cleanly.

Clustering runs independently within each HTTP status code: status is a
hard partition (a 200 and a 404 with identical metrics are still
different response types), and the continuous features are standard-scaled
within each status group.  The distance unit is thus "one std of this
response family's own jitter", so a single response that deviates
moderately from a tight baseline (e.g. +100 B among 499 identical
responses) lands far outside eps and is reported in the noise bucket for
triage.  Statuses seen fewer than --min-cluster times are never clustered
and are reported as noise (outliers).

Usage:
    python3 ffuf_analyse.py <audit-log.json> [more-logs.json ...] [options]

Requires: numpy, scikit-learn
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

# HTTP status codes from the IANA registry:
#   https://www.iana.org/assignments/http-status-codes/http-status-codes.xhtml
# Each gets its own one-hot slot so 200 ≠ 201 ≠ 204, etc.
KNOWN_CODES = (
    100,
    101,
    102,
    103,
    104,
    200,
    201,
    202,
    203,
    204,
    205,
    206,
    207,
    208,
    226,
    300,
    301,
    302,
    303,
    304,
    305,
    306,
    307,
    308,
    400,
    401,
    402,
    403,
    404,
    405,
    406,
    407,
    408,
    409,
    410,
    411,
    412,
    413,
    414,
    415,
    416,
    417,
    418,
    421,
    422,
    423,
    424,
    425,
    426,
    428,
    429,
    431,
    451,
    500,
    501,
    502,
    503,
    504,
    505,
    506,
    507,
    508,
    510,
    511,
)
# O(1) status lookup; unknown codes share the final slot
STATUS_INDEX = {c: i for i, c in enumerate(KNOWN_CODES)}
UNKNOWN_STATUS_IDX = len(KNOWN_CODES)

# Core features extracted directly from each response record.
# Status is not a feature: it is the group key for per-status clustering.
CORE_FEATURES = ["content_length", "content_words", "content_lines", "duration_us"]

# ---------------------------------------------------------------------------
# Parsing & feature enrichment
# ---------------------------------------------------------------------------


def _extract_payload(input_map: dict) -> tuple[str, int]:
    """Return (payload_text, payload_byte_len) for a request Input map.

    Prefers the FUZZ key; falls back to the first other key so renamed
    fuzz words still work.  With several custom words only the first is
    used, so the payload length (and thus the overhead feature) is then an
    undercount.
    """
    keys = [k for k in input_map if k != "FFUFHASH"]
    if "FUZZ" in input_map:
        b64 = input_map["FUZZ"]
    elif keys:
        b64 = input_map[keys[0]]
    else:
        b64 = ""
    try:
        raw = base64.b64decode(b64)
    except Exception:
        raw = b64.encode("utf-8")  # keep something inspectable, don't crash
    return raw.decode("utf-8", errors="replace"), len(raw)


def parse_log(path: str) -> tuple[list[dict], int, int]:
    """Parse an ffuf audit log into a list of response records.

    Returns (records, skipped_lines, unpaired_responses).

    Recent ffuf embeds the originating request in each *ffuf.Response
    record (Data["Request"]), which makes records self-contained and
    order-independent.  For older logs without the embedded request we fall
    back to pairing with the most recently seen *ffuf.Request, which is
    only reliable when requests and responses strictly alternate.
    """
    records = []
    last_req = None
    skipped = 0
    unpaired = 0

    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(obj, dict):
                skipped += 1
                continue
            typ = obj.get("Type")
            if typ == "*ffuf.Request":
                last_req = obj.get("Data")
            elif typ == "*ffuf.Response":
                rd = obj.get("Data") or {}
                reqd = rd.get("Request") or last_req
                if not isinstance(reqd, dict):
                    unpaired += 1
                    continue
                payload, payload_len = _extract_payload(reqd.get("Input") or {})
                records.append(
                    {
                        "position": reqd.get("Position", 0),
                        "payload": payload,
                        "payload_len": payload_len,
                        "status": rd.get("StatusCode", 0),
                        "content_length": rd.get("ContentLength", 0),
                        "content_words": rd.get("ContentWords", 0),
                        "content_lines": rd.get("ContentLines", 0),
                        "content_type": rd.get("ContentType", ""),
                        # ffuf reports Duration in nanoseconds
                        "duration_us": int(rd.get("Duration", 0)) // 1000,
                    }
                )
    return records, skipped, unpaired


def enrich_records(records: list[dict]) -> list[dict]:
    """Add overhead feature to every record."""
    for rec in records:
        rec["overhead"] = rec["content_length"] - rec["payload_len"]
    return records


def _status_group(rec: dict) -> int:
    """Status group for a record (unknown codes share the last group)."""
    return STATUS_INDEX.get(rec["status"], UNKNOWN_STATUS_IDX)


def _group_matrix(records: list[dict], indices: list[int]) -> np.ndarray:
    """Standard-scaled continuous feature matrix for a status group.

    Scaling is fit on the group itself so the distance unit is one std of
    this response family's own variation. Features that are constant
    within the group centre to exactly 0 and contribute nothing to
    within-group distances.
    """
    X = np.array(
        [[float(records[i][f]) for f in CORE_FEATURES] for i in indices],
        dtype=np.float64,
    )
    return StandardScaler().fit_transform(X)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def cluster_records(
    records: list[dict], min_cluster_size: int = 3, eps: float = 0.5
) -> list[dict]:
    """
    Cluster responses using DBSCAN, independently within each http status group.

    Status is a hard partition: different status codes can never share a
    cluster regardless of eps. Continuous features are scaled within each
    group, so a response that deviates beyond eps of its own group's
    jitter becomes noise even if it is surrounded by a dense baseline.

    Status groups with fewer than min_cluster_size records are never
    clustered; all their records are reported as noise.

    Returns a list of cluster dicts, each with:
        label, records, stats, is_noise
    """
    n = len(records)
    if n == 0:
        return []

    groups: dict[int, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        groups[_status_group(rec)].append(i)

    labels = np.full(n, -1, dtype=int)
    next_label = 0
    for slot in sorted(groups):
        idx = groups[slot]
        if len(idx) < min_cluster_size:
            continue  # rare status -> stays in the noise bucket
        Xg = _group_matrix(records, idx)
        lab = DBSCAN(eps=eps, min_samples=min_cluster_size).fit_predict(Xg)
        for j, l in enumerate(lab):
            if l != -1:
                labels[idx[j]] = next_label + l
        if (lab != -1).any():
            next_label += int(lab.max()) + 1

    noise_count = int((labels == -1).sum())
    unique_labels = set(labels) - {-1}

    # Bail only if DBSCAN found no structure at all
    if not unique_labels:
        raise ValueError(
            f"DBSCAN found no clusters (all {n} points are noise) with "
            f"eps={eps}, min_cluster={min_cluster_size}. "
            f"Try a larger --eps (e.g. 1.0) or a smaller --min-cluster."
        )
    if noise_count > n * 0.3:
        print(
            f"  note: DBSCAN put {noise_count}/{n} "
            f"({noise_count / n * 100:.0f}%) points in the noise cluster",
            file=sys.stderr,
        )

    # Build cluster objects
    cluster_map: dict[int, list[int]] = defaultdict(list)
    for i, lb in enumerate(labels):
        cluster_map[lb].append(i)

    # Assign sequential labels 0..N, giving noise the highest number
    sorted_clusters = sorted(cluster_map.items(), key=lambda x: -len(x[1]))
    label_map = {}
    counter = 0
    for lb, _ in sorted_clusters:
        if lb != -1:
            label_map[lb] = counter
            counter += 1
    # Noise gets the final label
    if -1 in cluster_map:
        label_map[-1] = counter

    clusters = []
    for lb, indices in sorted_clusters:
        recs = [records[i] for i in indices]
        stats = _cluster_stats(recs)
        clusters.append(
            {
                "label": label_map[lb],
                "size": len(recs),
                "records": recs,
                "stats": stats,
                "is_noise": lb == -1,
            }
        )

    clusters.sort(key=lambda c: c["label"])
    return clusters


def _cluster_stats(recs: list[dict]) -> dict:
    """Compute per-cluster statistics for core features + overhead."""

    def _s(vals):
        n = len(vals)
        mn = mean(vals) if n else 0
        md = median(vals) if n else 0
        sd = stdev(vals) if n > 1 else 0
        return {"mean": mn, "median": md, "std": sd, "min": min(vals), "max": max(vals)}

    return {
        "content_length": _s([r["content_length"] for r in recs]),
        "content_words": _s([r["content_words"] for r in recs]),
        "content_lines": _s([r["content_lines"] for r in recs]),
        "duration_us": _s([r["duration_us"] for r in recs]),
        "overhead": _s([r["overhead"] for r in recs]),
        "payload_len": _s([r["payload_len"] for r in recs]),
        "content_types": Counter(r["content_type"] for r in recs),
        "top_status": (
            Counter(r["status"] for r in recs).most_common(1)[0][0] if recs else 0
        ),
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _global_stats(records: list[dict]) -> dict:
    """Compute global mean and stddev for each numeric feature."""

    def _s(vals):
        n = len(vals)
        mn = mean(vals) if n else 0
        sd = stdev(vals) if n > 1 else 0
        return {"mean": mn, "std": sd}

    return {
        "content_length": _s([r["content_length"] for r in records]),
        "content_words": _s([r["content_words"] for r in records]),
        "content_lines": _s([r["content_lines"] for r in records]),
        "duration_us": _s([r["duration_us"] for r in records]),
        "top_status": (
            Counter(r["status"] for r in records).most_common(1)[0][0] if records else 0
        ),
    }


def _distinctive_features(cl_stats: dict, gl_stats: dict) -> list[str]:
    """Compare cluster stats to global stats and return human-readable
    descriptions of features that deviate significantly.

    Uses z-scores on the mean: |z| > 1.5 is flagged.
    """
    distinctive = []

    # Check if the cluster's dominant status code differs from the global one
    if cl_stats.get("top_status") != gl_stats.get("top_status"):
        distinctive.append(
            f"status code {cl_stats['top_status']} (global {gl_stats['top_status']})"
        )

    labels = {
        "content_length": "response length",
        "content_words": "word count",
        "content_lines": "line count",
        "duration_us": "response time",
    }
    for key, label in labels.items():
        cl_mean = cl_stats[key]["mean"]
        gl_mean = gl_stats[key]["mean"]
        gl_std = gl_stats[key]["std"]
        if gl_std == 0:
            continue
        z = (cl_mean - gl_mean) / gl_std
        if abs(z) > 1.5:
            if key == "duration_us":
                distinctive.append(f"{label} {_fmt_us(cl_mean)} vs {_fmt_us(gl_mean)}")
            elif key == "content_length":
                distinctive.append(
                    f"{label} {_fmt_bytes(cl_mean)} vs {_fmt_bytes(gl_mean)}"
                )
            else:
                distinctive.append(
                    f"{label} {int(round(cl_mean))} (global {int(round(gl_mean))})"
                )
    return distinctive


def _status_label(records: list[dict]) -> str:
    """Return the most common status code, or 'mixed' if multiple."""
    codes = Counter(r["status"] for r in records)
    most_common = codes.most_common(2)
    if len(most_common) == 1:
        return str(most_common[0][0])
    # Show top codes if they're both significant (>5% of cluster)
    total = sum(codes.values())
    top = [str(c[0]) for c in most_common if c[1] / total > 0.05]
    return ",".join(top) if top else str(most_common[0][0])


def _fmt_bytes(n: float) -> str:
    n = round(n)
    if n >= 1_048_576:
        return f"{n / 1_048_576:.1f} MB"
    if n >= 1_024:
        return f"{n / 1_024:.1f} KB"
    return f"{n} B"


def _fmt_us(us: float) -> str:
    us = round(us)
    if us >= 1_000_000:
        return f"{us / 1_000_000:.2f}s"
    if us >= 1_000:
        return f"{us / 1_000:.1f} ms"
    return f"{us} µs"


def _truncate(s: str, maxlen: int = 100) -> str:
    return s[:maxlen] + ("…" if len(s) > maxlen else "")


def _overhead_slope(recs: list[dict]) -> float:
    """Regression slope of overhead against payload_len.

    ≈ 0  → payload reflected (overhead independent of payload length)
    ≈ −1 → static page (overhead = C − payload)
    ≈ +1 → overhead itself grows with the payload (e.g. double reflection)

    Returns None when undefined (n < 3 or constant payload length).
    """
    n = len(recs)
    if n < 3:
        return None
    xs = [r["payload_len"] for r in recs]
    ys = [r["overhead"] for r in recs]
    mx, my = mean(xs), mean(ys)
    varx = sum((x - mx) ** 2 for x in xs)
    if varx == 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / varx


def report(
    filepath: str, records: list[dict], clusters: list[dict], top_samples: int = 5
) -> None:
    print(f"\n{'=' * 78}")
    print(f"  Response Profile  -  {Path(filepath).name}")
    print(f"  {len(records)} request/response pairs  →  {len(clusters)} clusters")
    print(f"{'=' * 78}")

    # --- Overall summary table ---
    print(
        f"\n  {'Cluster':<4} {'Size':>6} {'%':>6}  {'Status':>10}  "
        f"{'Avg Length':>12}  {'Avg Words':>10}  {'Avg Time':>12}  "
        f"{'Content-Type'}"
    )
    print(
        f"  {'-' * 4} {'-' * 6} {'-' * 6}  {'-' * 10}  "
        f"{'-' * 12}  {'-' * 10}  {'-' * 12}  "
        f"{'-' * 25}"
    )

    for cl in clusters:
        s = cl["stats"]
        ct_top = s["content_types"].most_common(1)[0]
        ct_str = ct_top[0][:25] if ct_top else "(empty)"
        noise_tag = " (noise)" if cl["is_noise"] else ""
        status_str = _status_label(cl["records"])
        print(
            f"  #{cl['label']:<3} {cl['size']:>5}{noise_tag:>8} {cl['size']/len(records)*100:5.1f}%  "
            f"{status_str:>10}  "
            f"{_fmt_bytes(s['content_length']['mean']):>12}  "
            f"{round(s['content_words']['mean']):>10}  "
            f"{_fmt_us(s['duration_us']['mean']):>12}  "
            f"{ct_str}"
        )

    # --- Per-cluster overhead summary ---
    # overhead = content_length − payload_len
    # constant overhead + varying payload_len = reflective endpoint
    print(f"\n  {'Cluster':<8} {'Overhead':>10} {'Payload Range':>14}  Interpretation")
    print(f"  {'-' * 8} {'-' * 10} {'-' * 14}  {'-' * 30}")
    for cl in clusters:
        s = cl["stats"]
        oh = s["overhead"]
        pl_range = f"{_fmt_bytes(s['payload_len']['min'])} – {_fmt_bytes(s['payload_len']['max'])}"
        # Interpret the slope of overhead against payload_len:
        # ≈ 0 means the payload is reflected, ≈ −1 a static page
        # (overhead = C − payload), ≈ +1 that the overhead itself grows
        # with the payload (e.g. the payload appearing more than once).
        pl_std = s["payload_len"]["std"]
        slope = _overhead_slope(cl["records"])
        if pl_std <= 10:
            interp = "payload length ~constant - no reflection signal"
        elif slope is None:
            interp = "too few points to judge reflection"
        elif abs(slope) < 0.2:
            interp = "overhead independent of payload - payload reflected"
        elif slope < -0.5:
            interp = "static page (overhead shrinks as payload grows)"
        elif slope > 0.5:
            interp = "overhead grows with payload - check for double reflection"
        else:
            interp = "overhead varies with payload - no clear reflection"
        print(
            f"  #{cl['label']:<7} {_fmt_bytes(oh['mean']):>10} {pl_range:>14}  {interp}"
        )
    print("  note: the reflection heuristic assumes uncompressed bodies; 204/304,")
    print("  gzip responses and multiple fuzz words weaken the signal.")

    # --- Distinctive features per cluster ---
    gl = _global_stats(records)
    print(f"\n  {'Cluster':<8}  What makes it distinct")
    print(f"  {'-' * 8}  {'-' * 60}")
    for cl in clusters:
        feats = _distinctive_features(cl["stats"], gl)
        desc = "; ".join(feats) if feats else "baseline - no distinctive features"
        print(f"  #{cl['label']:<7}  {desc}")

    # --- Detail per cluster ---
    print()
    for cl in clusters:
        s = cl["stats"]
        tag = " [noise]" if cl["is_noise"] else ""
        feats = _distinctive_features(s, gl)
        print(f"{'-' * 78}")
        print(f"  Cluster #{cl['label']}  (n={cl['size']}){tag}")
        print(f"  Status codes: {_status_label(cl['records'])}")
        if feats:
            print(f"  Distinctive: {'; '.join(feats)}")
        print(f"{'-' * 78}")

        metrics = [
            ("Content-Length", "content_length", _fmt_bytes),
            ("Words", "content_words", int),
            ("Lines", "content_lines", int),
            ("Response Time", "duration_us", _fmt_us),
            ("Overhead", "overhead", _fmt_bytes),
        ]
        print(
            f"  {'Metric':<17} {'Mean':>12} {'Median':>12} {'StdDev':>12} {'Min':>12} {'Max':>12}"
        )
        print(f"  {'-' * 17} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12}")
        for label, key, fmt in metrics:
            d = s[key]
            print(
                f"  {label:<17} {fmt(d['mean']):>12} {fmt(d['median']):>12} "
                f"{fmt(d['std']):>12} {fmt(d['min']):>12} {fmt(d['max']):>12}"
            )

        # Content types
        if len(s["content_types"]) > 1:
            print(f"\n  Content types: {dict(s['content_types'])}")

        # Sample payloads
        samples = cl["records"][:top_samples]
        print(
            f"\n  Sample payloads (showing {min(top_samples, cl['size'])} of {cl['size']}):"
        )
        for rec in samples:
            print(f"    pos {rec['position']:>5}  |  {_truncate(rec['payload'], 70)}")

        print()

    print(f"{'=' * 78}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Cluster-based response analysis for ffuf audit logs.",
    )
    parser.add_argument("files", nargs="+", help="Audit log JSON file(s)")
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Sample payloads shown per cluster (default: 5)",
    )
    parser.add_argument(
        "--min-cluster",
        type=int,
        default=3,
        help="DBSCAN min_samples: min neighbours for a core " "point (default: 3)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.5,
        help="DBSCAN eps in the per-status scaled feature " "space (default: 0.5)",
    )
    args = parser.parse_args()

    failed = 0
    for filepath in args.files:
        if not Path(filepath).exists():
            print(f"Error: file not found: {filepath}", file=sys.stderr)
            failed += 1
            continue
        records, skipped, unpaired = parse_log(filepath)
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
        records = enrich_records(records)

        try:
            clusters = cluster_records(
                records, min_cluster_size=args.min_cluster, eps=args.eps
            )
        except ValueError as exc:
            print(f"\n  ✗ {filepath}: {exc}\n", file=sys.stderr)
            failed += 1
            continue
        report(filepath, records, clusters, top_samples=args.samples)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
