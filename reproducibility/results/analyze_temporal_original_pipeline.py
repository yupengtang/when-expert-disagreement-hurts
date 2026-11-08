#!/usr/bin/env python3
"""Compare the January audit with the July re-run of the original pipeline."""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results" / "data" / "main_audit" / "neurips_dedup_trials_FULL.csv"
NEW = ROOT / "results" / "data" / "temporal_orig_pipeline.jsonl"
OUT = ROOT / "results" / "analysis" / "temporal_original_pipeline_replication.csv"


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def main() -> None:
    old = pd.read_csv(OLD)
    old = old[(old.model == "GPT-4o-mini") & (old.domain == "Medicine")]
    rows = []
    for line in NEW.open():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    new = pd.DataFrame(rows)
    new = new[(new.status == "ok") & (new.domain == "Medicine")]

    result = []
    for condition in ("A0", "A1", "A3"):
        jan = (old[old.condition == condition]
               .drop_duplicates("question_id").set_index("question_id"))
        jul = (new[new.condition == condition]
               .drop_duplicates("question_id").set_index("question_id"))
        ids = jan.index.intersection(jul.index)
        jan, jul = jan.loc[ids], jul.loc[ids]
        a, b = jan.reversal.astype(bool), jul.reversal.astype(bool)
        jul_only = int((~a & b).sum())
        jan_only = int((a & ~b).sum())
        result.append({
            "condition": condition,
            "n_paired": len(ids),
            "jan2026_reversal_pct": round(100 * a.mean(), 2),
            "jul2026_reversal_pct": round(100 * b.mean(), 2),
            "change_pp": round(100 * (b.mean() - a.mean()), 2),
            "jul_only_reversal": jul_only,
            "jan_only_reversal": jan_only,
            "mcnemar_p": mcnemar(jul_only, jan_only),
            "same_pass1_label_pct": round(
                100 * (jan.pass1.to_numpy() == jul.pass1.to_numpy()).mean(), 2),
        })
    table = pd.DataFrame(result)
    table.to_csv(OUT, index=False)
    print(table.to_string(index=False))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
