#!/usr/bin/env python3
"""Paired analyses of the five-model main audit.

Data source: results/data/main_audit/neurips_dedup_trials_FULL.csv, the
de-duplicated trial-level table (29,778 rows). Its per-model A0/A1/A3 counts
reproduce the main-audit accounting for all five models.

Outputs (results/analysis/):
  pairing_check.csv               shared-Pass-1 verification
  conditional_reversal.csv        P(R | initial correct / wrong)
  relative_effects.csv            odds ratios and risk ratios
  direction_and_label.csv         Yes->No vs No->Yes, by true label
  class_balance.csv               true-label balance by domain
  base_rate_decomposition.csv     harmful fraction vs Pass-1 accuracy
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "data" / "main_audit" / "neurips_dedup_trials_FULL.csv"
OUT = ROOT / "results" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

API_MODELS = ["GPT-4o-mini", "Claude-Sonnet-4", "Gemini-2.0-Flash"]
OPEN_MODELS = ["Llama-3.1-8B", "Mistral-7B"]


def load() -> pd.DataFrame:
    df = pd.read_csv(SRC)
    for c in ("reversal", "pass1_correct", "pass2_correct"):
        df[c] = df[c].astype(str).str.lower().map({"true": True, "false": False})
    return df


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def diff_ci(k1, n1, k2, n2, z=1.96):
    """Newcombe hybrid-score CI for a difference of independent proportions."""
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = k1 / n1 - k2 / n2
    lo = d - np.sqrt((k1 / n1 - l1) ** 2 + (u2 - k2 / n2) ** 2)
    hi = d + np.sqrt((u1 - k1 / n1) ** 2 + (k2 / n2 - l2) ** 2)
    return d, lo, hi


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar p-value on discordant pairs."""
    from math import comb

    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, statistic, seed: int,
                     n_boot: int = 50_000) -> tuple[float, float]:
    """Percentile CI from resampling paired items, not independent margins."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    values = []
    chunk = 1_000
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        idx = rng.integers(0, len(a), size=(size, len(a)))
        values.extend(statistic(a[idx].mean(axis=1), b[idx].mean(axis=1)))
    return tuple(np.quantile(np.asarray(values), [0.025, 0.975]))


# --------------------------------------------------------------------------
def pairing_check(df: pd.DataFrame) -> pd.DataFrame:
    """Is the full audit branch-paired on the same Pass-1 answer?"""
    rows = []
    for model, g in df.groupby("model"):
        wide = g.pivot_table(index="question_id", columns="condition",
                             values="pass1", aggfunc="first")
        for a, b in (("A1", "A3"), ("A0", "A1"), ("A0", "A3")):
            if a not in wide or b not in wide:
                continue
            sub = wide[[a, b]].dropna()
            same = int((sub[a] == sub[b]).sum())
            rows.append({
                "model": model, "contrast": f"{a} vs {b}",
                "items_with_both": len(sub),
                "same_pass1": same,
                "same_pass1_pct": round(100 * same / max(1, len(sub)), 3),
            })
    return pd.DataFrame(rows)


def conditional_reversal(df: pd.DataFrame) -> pd.DataFrame:
    """P(R | Pass-1 correct) and P(R | Pass-1 wrong), per model and condition."""
    rows = []
    for (model, cond), g in df.groupby(["model", "condition"]):
        gc, gw = g[g.pass1_correct], g[~g.pass1_correct]
        kc, nc = int(gc.reversal.sum()), len(gc)
        kw, nw = int(gw.reversal.sum()), len(gw)
        lc = wilson(kc, nc)
        lw = wilson(kw, nw)
        rows.append({
            "model": model, "condition": cond,
            "n_pass1_correct": nc, "R_given_correct": round(kc / max(1, nc), 4),
            "R_given_correct_lo": round(lc[0], 4), "R_given_correct_hi": round(lc[1], 4),
            "n_pass1_wrong": nw, "R_given_wrong": round(kw / max(1, nw), 4),
            "R_given_wrong_lo": round(lw[0], 4), "R_given_wrong_hi": round(lw[1], 4),
            # selectivity: how much more often the model revises when it was wrong
            "selectivity_diff": round(kw / max(1, nw) - kc / max(1, nc), 4),
            "selectivity_ratio": round((kw / max(1, nw)) / max(1e-9, kc / max(1, nc)), 3),
            "pass1_acc": round(nc / max(1, nc + nw), 4),
        })
    return pd.DataFrame(rows).sort_values(["model", "condition"])


def relative_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Complement the additive P with the odds ratio and the risk ratio.

    A1 and A3 share the same Pass-1 per item, so the contrast is also computed
    as a within-item paired (McNemar) test.
    """
    rows = []
    for model_idx, (model, g) in enumerate(df.groupby("model")):
        w = g.pivot_table(index="question_id", columns="condition",
                          values="reversal", aggfunc="first")
        if "A1" not in w or "A3" not in w:
            continue
        sub = w[["A1", "A3"]].dropna()
        a1, a3 = sub["A1"].astype(bool), sub["A3"].astype(bool)
        n = len(sub)
        k1, k3 = int(a1.sum()), int(a3.sum())
        p1, p3 = k1 / n, k3 / n
        b = int((~a1 & a3).sum())   # reversed only under expert label
        c = int((a1 & ~a3).sum())   # reversed only under anonymous
        odds = (p3 / (1 - p3)) / (p1 / (1 - p1)) if 0 < p1 < 1 and 0 < p3 < 1 else np.nan
        rr = p3 / p1 if p1 > 0 else np.nan
        seed = 9907 + model_idx
        or_ci = paired_bootstrap(
            a1.to_numpy(), a3.to_numpy(),
            lambda x, y: (y / (1 - y)) / (x / (1 - x)), seed)
        rr_ci = paired_bootstrap(
            a1.to_numpy(), a3.to_numpy(), lambda x, y: y / x, seed + 100)
        diff_ci_paired = paired_bootstrap(
            a1.to_numpy(), a3.to_numpy(), lambda x, y: 100 * (y - x), seed + 200)
        rows.append({
            "model": model, "n_paired_items": n,
            "A1_reversal": round(p1, 4), "A3_reversal": round(p3, 4),
            "P_additive_pp": round(100 * (p3 - p1), 2),
            "P_paired_boot_ci_lo_pp": round(diff_ci_paired[0], 2),
            "P_paired_boot_ci_hi_pp": round(diff_ci_paired[1], 2),
            "odds_ratio": round(odds, 3),
            "OR_paired_boot_ci_lo": round(or_ci[0], 3),
            "OR_paired_boot_ci_hi": round(or_ci[1], 3),
            "risk_ratio": round(rr, 3),
            "RR_paired_boot_ci_lo": round(rr_ci[0], 3),
            "RR_paired_boot_ci_hi": round(rr_ci[1], 3),
            "discordant_A3only": b, "discordant_A1only": c,
            "mcnemar_p": mcnemar_exact(b, c),
        })
    return pd.DataFrame(rows)


def direction_and_label(df: pd.DataFrame) -> pd.DataFrame:
    """Reversal by initial answer direction and by true label."""
    rows = []
    for (model, cond), g in df.groupby(["model", "condition"]):
        for key, sub in (("pass1=Yes (Yes->No)", g[g.pass1 == "Yes"]),
                         ("pass1=No (No->Yes)", g[g.pass1 == "No"])):
            k, n = int(sub.reversal.sum()), len(sub)
            lo, hi = wilson(k, n)
            rows.append({"model": model, "condition": cond, "stratum_type": "initial answer",
                         "stratum": key, "n": n, "reversal": round(k / max(1, n), 4),
                         "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)})
        for key, sub in (("true=Yes", g[g.ground_truth == "Yes"]),
                         ("true=No", g[g.ground_truth == "No"])):
            k, n = int(sub.reversal.sum()), len(sub)
            lo, hi = wilson(k, n)
            rows.append({"model": model, "condition": cond, "stratum_type": "true label",
                         "stratum": key, "n": n, "reversal": round(k / max(1, n), 4),
                         "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)})
    return pd.DataFrame(rows)


def prestige_by_stratum(df: pd.DataFrame) -> pd.DataFrame:
    """Is P = A3 - A1 present within BOTH answer directions and BOTH labels?"""
    rows = []
    for model, g in df.groupby("model"):
        for stype, col, vals in (("initial answer", "pass1", ("Yes", "No")),
                                 ("true label", "ground_truth", ("Yes", "No"))):
            for v in vals:
                s = g[g[col] == v]
                w = s.pivot_table(index="question_id", columns="condition",
                                  values="reversal", aggfunc="first")
                if "A1" not in w or "A3" not in w:
                    continue
                sub = w[["A1", "A3"]].dropna()
                if len(sub) == 0:
                    continue
                a1, a3 = sub["A1"].astype(bool), sub["A3"].astype(bool)
                n = len(sub)
                b = int((~a1 & a3).sum())
                c = int((a1 & ~a3).sum())
                d, lo, hi = diff_ci(int(a3.sum()), n, int(a1.sum()), n)
                rows.append({
                    "model": model, "stratum_type": stype, "stratum": f"{col}={v}",
                    "n_paired": n,
                    "A1_reversal": round(a1.mean(), 4), "A3_reversal": round(a3.mean(), 4),
                    "P_pp": round(100 * d, 2),
                    "ci_lo_pp": round(100 * lo, 2), "ci_hi_pp": round(100 * hi, 2),
                    "mcnemar_p": mcnemar_exact(b, c),
                })
    return pd.DataFrame(rows)


def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    items = df.drop_duplicates("question_id")[["question_id", "domain", "ground_truth"]]
    tab = (items.groupby(["domain", "ground_truth"]).size()
           .unstack(fill_value=0).rename(columns=lambda c: f"true_{c}"))
    tab["total"] = tab.sum(axis=1)
    tab["yes_rate"] = (tab["true_Yes"] / tab["total"]).round(4)
    tab.loc["All"] = tab.sum()
    tab.loc["All", "yes_rate"] = round(tab.loc["All", "true_Yes"] / tab.loc["All", "total"], 4)
    return tab.reset_index()


def base_rate_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """The harmful fraction depends partly on Pass-1 accuracy.

    If revision were completely blind to initial correctness the expected share
    of Correct->Wrong among reversals would just be Pass-1 accuracy. Comparing
    the observed harmful fraction with that null makes the excess explicit and
    explains the gap between closed- and open-weight models.
    """
    rows = []
    for (model, cond), g in df.groupby(["model", "condition"]):
        rev = g[g.reversal]
        cw = int((rev.pass1_correct & ~rev.pass2_correct).sum())
        wc = int((~rev.pass1_correct & rev.pass2_correct).sum())
        n_rev = len(rev)
        acc1 = g.pass1_correct.mean()
        gc, gw = g[g.pass1_correct], g[~g.pass1_correct]
        r_c = gc.reversal.mean() if len(gc) else np.nan
        r_w = gw.reversal.mean() if len(gw) else np.nan
        # harmful fraction predicted by base rate alone (equal reversal hazard)
        null_harmful = acc1
        # harmful fraction predicted by the observed conditional hazards
        pred = (acc1 * r_c) / (acc1 * r_c + (1 - acc1) * r_w) if n_rev else np.nan
        lo, hi = wilson(cw, n_rev)
        rows.append({
            "model": model, "condition": cond, "pass1_acc": round(acc1, 4),
            "reversals": n_rev, "C_to_W": cw, "W_to_C": wc,
            "harmful_frac": round(cw / max(1, n_rev), 4),
            "harmful_ci_lo": round(lo, 4), "harmful_ci_hi": round(hi, 4),
            "harmful_if_revision_blind": round(null_harmful, 4),
            "harmful_excess_over_blind": round(cw / max(1, n_rev) - null_harmful, 4),
            "R_given_correct": round(r_c, 4), "R_given_wrong": round(r_w, 4),
            "harmful_pred_from_hazards": round(pred, 4) if pred == pred else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["condition", "model"])


def main() -> None:
    df = load()
    cov = df.groupby(["model", "condition"]).size().unstack(fill_value=0)
    print("coverage by model and condition:\n", cov, "\n")
    cov.to_csv(OUT / "data_coverage.csv")

    api = df.copy()   # the trial table covers all five models
    print(f"rows used: {len(api)} across {api.model.nunique()} models\n")

    tables = {
        "pairing_check.csv": pairing_check(df),
        "conditional_reversal.csv": conditional_reversal(api),
        "relative_effects.csv": relative_effects(api),
        "direction_and_label.csv": direction_and_label(api),
        "prestige_by_stratum.csv": prestige_by_stratum(api),
        "class_balance.csv": class_balance(api),
        "base_rate_decomposition.csv": base_rate_decomposition(api),
    }
    for name, t in tables.items():
        t.to_csv(OUT / name, index=False)
        print("=" * 78)
        print(name)
        print(t.to_string(index=False))
    print(f"\nWrote {len(tables)} tables to {OUT}")


if __name__ == "__main__":
    main()
