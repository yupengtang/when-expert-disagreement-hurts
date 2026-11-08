#!/usr/bin/env python3
"""Analyze the branch-paired fallibility and track-record conditions."""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "data"
OUT = ROOT / "results" / "analysis" / "credibility_gradient.csv"


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.open():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return pd.DataFrame(rows)


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def paired_contrast(wide: pd.DataFrame, hi: str, lo: str):
    pair = wide[[hi, lo]].dropna().astype(bool)
    hi_only = int((pair[hi] & ~pair[lo]).sum())
    lo_only = int((~pair[hi] & pair[lo]).sum())
    return (100 * (pair[hi].mean() - pair[lo].mean()),
            mcnemar(hi_only, lo_only))


def main() -> None:
    specs = [
        ("paper", DATA / "controls.jsonl", DATA / "gradient_main.jsonl"),
        ("current", DATA / "controls_sota.jsonl", DATA / "gradient_sota.jsonl"),
    ]
    records = []
    for label, base_path, gradient_path in specs:
        base, gradient = read_jsonl(base_path), read_jsonl(gradient_path)
        df = pd.concat([base[base.condition.isin(["A1", "A4", "A3"])], gradient])
        df = df[df.status == "ok"]
        for model, group in df.groupby("model_id"):
            wide = group.pivot_table(index="question_id", columns="condition",
                                     values="reversal", aggfunc="first")
            needed = ["A1", "A4", "A3_unc", "A3", "A3_track"]
            if not set(needed).issubset(wide.columns):
                continue
            rates = {c: 100 * wide[c].dropna().astype(bool).mean() for c in needed}
            record_diff, record_p = paired_contrast(wide, "A3_track", "A3")
            hedge_diff, hedge_p = paired_contrast(wide, "A3_unc", "A3")
            records.append({
                "set": label,
                "model": model,
                "n": len(wide[needed].dropna()),
                "anon": round(rates["A1"], 1),
                "neutral": round(rates["A4"], 1),
                "expert+hedge": round(rates["A3_unc"], 1),
                "expert": round(rates["A3"], 1),
                "expert+record": round(rates["A3_track"], 1),
                "record_minus_expert": round(record_diff, 1),
                "p_record": record_p,
                "hedge_minus_expert": round(hedge_diff, 1),
                "p_hedge": hedge_p,
                "expert_levels_ordered": (
                    rates["A3_unc"] < rates["A3"] < rates["A3_track"]),
            })
    table = pd.DataFrame(records)
    table.to_csv(OUT, index=False)
    print(table.to_string(index=False))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
