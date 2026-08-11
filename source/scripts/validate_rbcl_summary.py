"""Return success only for a complete RBCL summary with finite metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def finite_tree(value) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--expected-experiences", type=int, default=10)
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    valid = (
        finite_tree(payload)
        and len(payload.get("experience_results", [])) == args.expected_experiences
        and bool(payload.get("last_metrics"))
    )
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
