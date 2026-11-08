#!/usr/bin/env python3
"""Conditional reversal rates from aggregates, and the preference-report mismatch.

(1) Conditional reversal rates for all five main models, derived exactly from
    the main-audit aggregates. In a binary task a reversal from a correct
    Pass-1 answer necessarily lands on the wrong answer, so

        harmful fraction h = P(C1 = 1 | R = 1)

    and Bayes' rule gives the two conditional reversal rates in closed form:

        R | correct = h * R / acc1
        R | wrong   = (1 - h) * R / (1 - acc1)

    with R the condition reversal rate and acc1 the Pass-1 accuracy. The
    derivation is checked against the trial-level table. The gap in harmful
    fraction between closed and open models is a difference in selectivity,
    not in how damaging each revision is.

(2) Why output reversals need not move the teacher-forced preference sign.
    Uses the Llama-3.1-8B log-odds retained in the de-duplicated trial table.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "data" / "main_audit" / "neurips_dedup_trials_FULL.csv"
OUT = ROOT / "results" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

# Main-audit aggregates: Pass-1 accuracy and reversal rates (%) under A0, A1 and A3.
PUB = {
    #                acc1     A0     A1     A3
    "GPT-4o-mini":      (0.870,  3.4,  11.5, 18.4),
    "Claude-Sonnet-4":  (0.853,  5.6,  21.1, 23.7),
    "Gemini-2.0-Flash": (0.897,  5.5,  13.4, 11.5),
    "Llama-3.1-8B":     (0.830, 13.1,  18.2, 23.4),
    "Mistral-7B":       (0.877,  4.4,  13.1, 18.1),
}
# Main-audit harmful fraction under A3.
HARMFUL_A3 = {
    "GPT-4o-mini": 0.800, "Claude-Sonnet-4": 0.839, "Gemini-2.0-Flash": 0.798,
    "Llama-3.1-8B": 0.548, "Mistral-7B": 0.553,
}
OPEN = {"Llama-3.1-8B", "Mistral-7B"}


def load_trials() -> pd.DataFrame:
    df = pd.read_csv(SRC)
    for c in ("reversal", "pass1_correct", "pass2_correct"):
        df[c] = df[c].astype(str).str.lower().map({"true": True, "false": False})
    return df


def derived_conditional() -> pd.DataFrame:
    rows = []
    for model, (acc1, a0, a1, a3) in PUB.items():
        h = HARMFUL_A3[model]
        R = a3 / 100.0
        r_c = h * R / acc1
        r_w = (1 - h) * R / (1 - acc1)
        rows.append({
            "model": model,
            "family": "open-weight" if model in OPEN else "closed API",
            "condition": "A3 (expert)",
            "pass1_acc": acc1,
            "reversal_rate": round(R, 4),
            "harmful_frac_published": h,
            "R_given_correct": round(r_c, 4),
            "R_given_wrong": round(r_w, 4),
            "selectivity_ratio": round(r_w / r_c, 3),
            "harmful_if_revision_blind": acc1,
            "harmful_excess_over_blind_pp": round(100 * (h - acc1), 1),
        })
    return pd.DataFrame(rows)


def validate(df_trials: pd.DataFrame, derived: pd.DataFrame) -> pd.DataFrame:
    """Check the closed-form derivation against the trial-level table."""
    rows = []
    a3 = df_trials[df_trials.condition == "A3"]
    for model, g in a3.groupby("model"):
        rev = g[g.reversal]
        if len(rev) == 0:
            continue
        obs_c = g[g.pass1_correct].reversal.mean()
        obs_w = g[~g.pass1_correct].reversal.mean()
        d = derived[derived.model == model]
        if d.empty:
            continue
        rows.append({
            "model": model, "n_trials": len(g),
            "observed_R_given_correct": round(obs_c, 4),
            "derived_R_given_correct": float(d.R_given_correct.iloc[0]),
            "observed_R_given_wrong": round(obs_w, 4),
            "derived_R_given_wrong": float(d.R_given_wrong.iloc[0]),
            "note": "complete" if len(g) > 1500 else "partial mirror",
        })
    return pd.DataFrame(rows)


def logit_mechanism(df_trials: pd.DataFrame) -> tuple:
    """Why do output reversals not always flip the preference sign?"""
    lla = df_trials[df_trials.logodds2.notna()].copy()
    lla["signflip"] = np.sign(lla.logodds1) != np.sign(lla.logodds2)
    lla["abs_l2"] = lla.logodds2.abs()
    lla["abs_l1"] = lla.logodds1.abs()
    lla["delta"] = lla.logodds2 - lla.logodds1
    # The opponent always states the opposite of Pass 1. Positive aligned_delta
    # therefore denotes movement toward the opponent for either answer direction.
    lla["aligned_delta"] = lla.delta * np.where(lla.pass1 == "Yes", -1.0, 1.0)

    rows = []
    for (mdl, cond), g in lla.groupby(["model", "condition"]):
        rev = g[g.reversal]
        norev = g[~g.reversal]
        mism = rev[~rev.signflip]
        rows.append({
            "model": mdl, "condition": cond, "n": len(g),
            "reversals": len(rev),
            "reversals_without_signflip": len(mism),
            "mismatch_rate": round(len(mism) / max(1, len(rev)), 4),
            "median_|l2|_reversed": round(rev.abs_l2.median(), 3) if len(rev) else np.nan,
            "median_|l2|_not_reversed": round(norev.abs_l2.median(), 3) if len(norev) else np.nan,
            "median_|l2|_mismatched_reversals": round(mism.abs_l2.median(), 3) if len(mism) else np.nan,
            "mean_delta_toward_opponent": round(g.aligned_delta.mean(), 3),
            "mean_delta_l_reversed": round(rev.delta.mean(), 3) if len(rev) else np.nan,
            "mean_delta_l_not_reversed": round(norev.delta.mean(), 3) if len(norev) else np.nan,
        })
    per_cond = pd.DataFrame(rows)

    # How concentrated near the decision boundary are mismatched reversals?
    bins = [0, 0.5, 1, 2, 4, 8, np.inf]
    labels = ["|l2|<0.5", "0.5-1", "1-2", "2-4", "4-8", ">8"]
    d2 = lla[lla.condition.isin(["A1", "A3"])].copy()
    d2["bucket"] = pd.cut(d2.abs_l2, bins=bins, labels=labels)
    grp = d2.groupby("bucket", observed=False).apply(
        lambda g: pd.Series({
            "n": len(g),
            "reversal_rate": round(g.reversal.mean(), 4) if len(g) else np.nan,
            "share_of_reversals": round(g.reversal.sum() / max(1, d2.reversal.sum()), 4),
        }), include_groups=False)
    return per_cond, grp.reset_index()


def main() -> None:
    trials = load_trials()
    derived = derived_conditional()
    val = validate(trials, derived)
    per_cond, buckets = logit_mechanism(trials)

    for name, t in {
        "derived_conditional_all_models.csv": derived,
        "derivation_validation.csv": val,
        "logit_mechanism_by_condition.csv": per_cond,
        "logit_boundary_buckets.csv": buckets,
    }.items():
        t.to_csv(OUT / name, index=False)
        print("=" * 80)
        print(name)
        print(t.to_string(index=False))

    print("\n" + "=" * 80)
    print("closed vs open, A3 (means):")
    print(derived.groupby("family")[
        ["pass1_acc", "reversal_rate", "harmful_frac_published",
         "R_given_correct", "R_given_wrong", "selectivity_ratio"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
