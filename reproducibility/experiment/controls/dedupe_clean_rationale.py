#!/usr/bin/env python3
"""Retain the earliest row for each model-item-condition JSONL key."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    if args.out.exists():
        parser.error(f"output exists: {args.out}")

    seen = set()
    kept = 0
    duplicates = 0
    with args.out.open("x") as output:
        for line_number, line in enumerate(args.source.open(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON on line {line_number}") from exc
            key = (row.get("model_id"), row.get("question_id"), row.get("condition"))
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
    print(f"kept={kept} duplicates_removed={duplicates} wrote={args.out}")


if __name__ == "__main__":
    main()
