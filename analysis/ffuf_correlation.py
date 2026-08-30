#!/usr/bin/env python3
"""
Calculate Pearson correlation between ffuf input payload length
and response length from ffuf audit JSON logs.

Usage:
    python ffuf_correlation.py <audit-log.json>
"""

import json
import sys
import base64
from pathlib import Path


def pearson_correlation(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient between two lists."""
    n = len(x)
    if n == 0:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
    den_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5

    if den_x == 0 or den_y == 0:
        return 0.0

    return num / (den_x * den_y)


def parse_audit_log(filepath: str) -> list[tuple[int, int]]:
    """
    Parse an ffuf audit log and return pairs of (payload_length, response_length).
    Each pair corresponds to one fuzz iteration.
    """
    pairs = []
    current_request = None

    with open(filepath, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"Warning: skipping malformed JSON on line {line_num}",
                    file=sys.stderr,
                )
                continue

            rec_type = record.get("Type", "")

            if rec_type == "*ffuf.Request":
                current_request = record
            elif rec_type == "*ffuf.Response":
                if current_request is None:
                    print(
                        f"Warning: response without request on line {line_num}",
                        file=sys.stderr,
                    )
                    continue

                # Extract fuzz payload length (decode from base64)
                fuzz_payload = b""
                input_map = current_request.get("Data", {}).get("Input", {})
                fuzz_val = input_map.get("FUZZ", "")
                if fuzz_val:
                    try:
                        fuzz_payload = base64.b64decode(fuzz_val)
                    except Exception:
                        fuzz_payload = fuzz_val.encode()

                payload_length = len(fuzz_payload)

                # Extract response content length
                resp_data = record.get("Data", {})
                response_length = resp_data.get("ContentLength", 0)

                pairs.append((payload_length, response_length))
                current_request = None

    return pairs


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <audit-log.json>", file=sys.stderr)
        sys.exit(1)

    filepath = sys.argv[1]
    if not Path(filepath).exists():
        print(f"Error: file not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing {filepath}...")
    pairs = parse_audit_log(filepath)

    if not pairs:
        print("No valid request/response pairs found.")
        sys.exit(1)

    payload_lengths = [p for p, _ in pairs]
    response_lengths = [r for _, r in pairs]

    r = pearson_correlation(payload_lengths, response_lengths)

    print(f"\n{'='*50}")
    print(f"  ffuf Payload Length vs Response Length")
    print(f"{'='*50}")
    print(f"  Total pairs analysed : {len(pairs)}")
    print(f"  Payload length range : {min(payload_lengths)} - {max(payload_lengths)}")
    print(f"  Response length range: {min(response_lengths)} - {max(response_lengths)}")
    print(f"  Pearson r            : {r:.6f}")
    print(f"{'='*50}")
    print()
    print(f"Interpretation:")
    if r > 0.7:
        print(f"  Strong positive correlation - response length grows with payload")
    elif r > 0.3:
        print(f"  Weak positive correlation")
    elif r > -0.3:
        print(f"  No meaningful correlation")
    elif r > -0.7:
        print(f"  Weak negative correlation")
    else:
        print(f"  Strong negative correlation - response length shrinks with payload")


if __name__ == "__main__":
    main()
