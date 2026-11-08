#!/usr/bin/env python3
"""Paired accuracy changes and the Gemini-2.0-Flash heterogeneity analysis.

Because Pass 1 is shared across conditions, accuracy change and the A1->A3
selectivity shift are both exactly paired, so McNemar applies directly.
"""

from __future__ import annotations

import io
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "data" / "main_audit" / "neurips_dedup_trials_FULL.csv"
OUT = ROOT / "results" / "analysis"


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def load() -> pd.DataFrame:
    df = pd.read_csv(SRC)
    for c in ("reversal", "pass1_correct", "pass2_correct"):
        df[c] = df[c].astype(str).str.lower().map({"true": True, "false": False})
    return df[df.model.isin(["GPT-4o-mini", "Claude-Sonnet-4", "Gemini-2.0-Flash"])]


def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, cond), g in df.groupby(["model", "condition"]):
        b = int((g.pass1_correct & ~g.pass2_correct).sum())   # correct -> wrong
        c = int((~g.pass1_correct & g.pass2_correct).sum())   # wrong -> correct
        rows.append({
            "model": model, "condition": cond, "n": len(g),
            "pass1_acc": round(g.pass1_correct.mean(), 4),
            "pass2_acc": round(g.pass2_correct.mean(), 4),
            "delta_acc_pp": round(100 * (g.pass2_correct.mean() - g.pass1_correct.mean()), 2),
            "C_to_W": b, "W_to_C": c, "mcnemar_p": mcnemar(b, c),
        })
    return pd.DataFrame(rows).sort_values(["model", "condition"])


def selectivity_shift(df: pd.DataFrame) -> pd.DataFrame:
    """Does the expert label change WHERE the model revises (A1 -> A3)?

    Computed on the shared Pass-1, so each item contributes to the same
    correctness stratum in both conditions.
    """
    rows = []
    for model, g in df.groupby("model"):
        for stratum, sub in (("Pass-1 correct", g[g.pass1_correct]),
                             ("Pass-1 wrong", g[~g.pass1_correct])):
            w = sub.pivot_table(index="question_id", columns="condition",
                                values="reversal", aggfunc="first")
            if "A1" not in w or "A3" not in w:
                continue
            p = w[["A1", "A3"]].dropna()
            a1, a3 = p["A1"].astype(bool), p["A3"].astype(bool)
            b = int((~a1 & a3).sum())
            c = int((a1 & ~a3).sum())
            rows.append({
                "model": model, "stratum": stratum, "n_paired": len(p),
                "A1_reversal": round(a1.mean(), 4), "A3_reversal": round(a3.mean(), 4),
                "shift_pp": round(100 * (a3.mean() - a1.mean()), 2),
                "mcnemar_p": mcnemar(b, c),
            })
    out = pd.DataFrame(rows)
    # selectivity ratio per model/condition
    extra = []
    for model, g in df.groupby("model"):
        for cond in ("A0", "A1", "A3"):
            s = g[g.condition == cond]
            rc = s[s.pass1_correct].reversal.mean()
            rw = s[~s.pass1_correct].reversal.mean()
            extra.append({"model": model, "condition": cond,
                          "R_given_correct": round(rc, 4), "R_given_wrong": round(rw, 4),
                          "selectivity_ratio": round(rw / rc, 3) if rc else np.nan})
    return out, pd.DataFrame(extra)


def main() -> None:
    df = load()
    acc = accuracy_table(df)
    shift, ratios = selectivity_shift(df)
    for name, t in {"accuracy_paired.csv": acc,
                    "selectivity_shift_A1_to_A3.csv": shift,
                    "selectivity_ratios.csv": ratios}.items():
        t.to_csv(OUT / name, index=False)
        print("=" * 80)
        print(name)
        print(t.to_string(index=False))


if __name__ == "__main__":
    main()
