#!/usr/bin/env python3
"""Analyze the no-padding rationale controls with paired contrasts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "results"))

from analyze_controls import mcnemar, wilson  # noqa: E402


ORDER = ["A1_rat_clean", "A3_rat_clean", "A3_rat_gated_clean"]


def load(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        for line in path.open():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    frame = pd.DataFrame(rows)
    frame = frame[frame.status == "ok"].copy()
    key = ["model_id", "question_id", "condition"]
    duplicate_rows = int(frame.duplicated(key, keep="first").sum())
    if duplicate_rows:
        print(
            f"dropping {duplicate_rows} duplicate rows; retaining the earliest "
            "record for each model-item-condition key"
        )
        frame = frame.drop_duplicates(key, keep="first")
    for column in ("reversal", "pass1_correct", "pass2_correct"):
        frame[column] = frame[column].astype(bool)
    return frame


def condition_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, condition), group in frame.groupby(["model_id", "condition"]):
        reversals = int(group.reversal.sum())
        lo, hi = wilson(reversals, len(group))
        harmful = int((group.pass1_correct & ~group.pass2_correct).sum())
        helpful = int((~group.pass1_correct & group.pass2_correct).sum())
        rows.append({
            "model": model,
            "condition": condition,
            "n": len(group),
            "reversal": group.reversal.mean(),
            "reversal_ci_lo": lo,
            "reversal_ci_hi": hi,
            "final_accuracy": group.pass2_correct.mean(),
            "accuracy_change_pp": 100 * (
                group.pass2_correct.mean() - group.pass1_correct.mean()
            ),
            "correct_to_wrong": harmful,
            "wrong_to_correct": helpful,
        })
    result = pd.DataFrame(rows)
    result["order"] = result.condition.map({c: i for i, c in enumerate(ORDER)})
    return result.sort_values(["model", "order"]).drop(columns="order")


def paired_contrast(group: pd.DataFrame, high: str, low: str,
                    stratum: str = "all") -> dict | None:
    if stratum == "benchmark_matching":
        group = group[group.rationale_target_relation == "benchmark_matching"]
    elif stratum == "benchmark_conflicting":
        group = group[group.rationale_target_relation == "benchmark_conflicting"]

    wide = group.pivot_table(
        index=["model_id", "question_id"], columns="condition",
        values="reversal", aggfunc="first"
    )
    if high not in wide or low not in wide:
        return None
    paired = wide[[high, low]].dropna().astype(bool)
    if paired.empty:
        return None
    high_only = int((paired[high] & ~paired[low]).sum())
    low_only = int((~paired[high] & paired[low]).sum())
    return {
        "stratum": stratum,
        "contrast": f"{high} - {low}",
        "n_paired": len(paired),
        "rate_high": paired[high].mean(),
        "rate_low": paired[low].mean(),
        "difference_pp": 100 * (paired[high].mean() - paired[low].mean()),
        "high_only": high_only,
        "low_only": low_only,
        "mcnemar_p": mcnemar(high_only, low_only),
    }


def paired_summaries(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    contrasts = [
        ("A3_rat_clean", "A1_rat_clean"),
        ("A3_rat_gated_clean", "A3_rat_clean"),
    ]
    strata = ["all", "benchmark_matching", "benchmark_conflicting"]
    for model, group in frame.groupby("model_id"):
        for high, low in contrasts:
            for stratum in strata:
                row = paired_contrast(group, high, low, stratum)
                if row:
                    row["model"] = model
                    rows.append(row)
    for high, low in contrasts:
        for stratum in strata:
            row = paired_contrast(frame, high, low, stratum)
            if row:
                row["model"] = "POOLED"
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--tag", default="clean_rationale")
    args = parser.parse_args()

    frame = load(args.inputs)
    output_dir = ROOT / "results" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = condition_summary(frame)
    contrasts = paired_summaries(frame)
    conditions.to_csv(output_dir / f"{args.tag}_conditions.csv", index=False)
    contrasts.to_csv(output_dir / f"{args.tag}_paired_contrasts.csv", index=False)
    print(conditions.to_string(index=False))
    print("\nPAIRED CONTRASTS")
    print(contrasts.to_string(index=False))


if __name__ == "__main__":
    main()
