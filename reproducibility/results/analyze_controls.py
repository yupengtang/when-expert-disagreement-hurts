#!/usr/bin/env python3
"""Analysis of the branch-paired extended control runs.

Every condition shares one Pass-1 generation per (item, model), so all
contrasts below are within-item, within-initial-answer, and paired: McNemar's
exact test on discordant items is the appropriate test throughout.

Usage:
  python results/analyze_controls.py results/data/controls.jsonl --tag main
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

ORDER = ["A0", "A1", "A2", "A4", "A3", "A3g", "A1_agree", "A3_agree", "A3_rat", "A3_rat_gated"]
LABEL = {
    "A0": "drift (no opponent)",
    "A1": "anonymous disagree",
    "A2": "low-prestige disagree",
    "A4": "neutral-metadata disagree",
    "A3": "expert disagree",
    "A3g": "expert disagree + gate",
    "A1_agree": "anonymous AGREE",
    "A3_agree": "expert AGREE",
    "A3_rat": "expert disagree + rationale",
    "A3_rat_gated": "expert disagree + rationale + gate",
}


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def _read(path: Path) -> list:
    rows = []
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def load(path: Path, fixed: Path | None = None) -> pd.DataFrame:
    """Load a control run, taking the rationale conditions from the compliant-rationale run.

    The two rationale conditions use arguments that state the assigned answer
    (experiment/controls/fix_opponent_rationales.py) and were run from the same
    stored Pass 1 (experiment/controls/rerun_rationale.py). Where such a row
    exists in `fixed`, it is used in place of the row in `path`.
    """
    rows = _read(path)
    if fixed and fixed.exists():
        repl = _read(fixed)
        keys = {(r.get("model_id"), r.get("question_id"), r.get("condition"))
                for r in repl}
        before = len(rows)
        rows = [r for r in rows
                if (r.get("model_id"), r.get("question_id"), r.get("condition")) not in keys]
        rows += repl
        print(f"  replaced {before - (len(rows) - len(repl))} rationale rows "
              f"with {len(repl)} rows from {fixed.name}")
    df = pd.DataFrame(rows)
    bad = df[df.status != "ok"]
    print(f"{path.name}: {len(df)} records, ok={int((df.status == 'ok').sum())}, "
          f"non-ok={len(bad)} ({dict(bad.status.value_counts()) if len(bad) else {}})")
    df = df[df.status == "ok"].copy()
    # rows dropped above leave the boolean columns as object dtype
    for c in ("reversal", "pass1_correct", "pass2_correct"):
        df[c] = df[c].fillna(False).astype(bool)
    # drop the handful of rationale trials whose stimulus never verified
    if "rationale_compliant" in df.columns:
        drop = df.condition.isin(["A3_rat", "A3_rat_gated"]) & (df.rationale_compliant == False)  # noqa: E712
        if int(drop.sum()):
            print(f"  dropping {int(drop.sum())} rationale trials with unverified stimulus")
            df = df[~drop]
    return df


def per_condition(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, cond), g in df.groupby(["model_id", "condition"]):
        k, n = int(g.reversal.sum()), len(g)
        lo, hi = wilson(k, n)
        rev = g[g.reversal]
        cw = int((rev.pass1_correct & ~rev.pass2_correct).sum())
        wc = int((~rev.pass1_correct & rev.pass2_correct).sum())
        b = int((g.pass1_correct & ~g.pass2_correct).sum())
        c = int((~g.pass1_correct & g.pass2_correct).sum())
        gc, gw = g[g.pass1_correct], g[~g.pass1_correct]
        rows.append({
            "model": model, "condition": cond, "label": LABEL.get(cond, cond),
            "n": n,
            "pass1_acc": round(g.pass1_correct.mean(), 4),
            "final_acc": round(g.pass2_correct.mean(), 4),
            "delta_acc_pp": round(100 * (g.pass2_correct.mean() - g.pass1_correct.mean()), 2),
            "acc_mcnemar_p": mcnemar(b, c),
            "reversal": round(k / n, 4), "rev_lo": round(lo, 4), "rev_hi": round(hi, 4),
            "C_to_W": cw, "W_to_C": wc,
            "harmful_frac": round(cw / max(1, k), 4),
            "R_given_correct": round(gc.reversal.mean(), 4) if len(gc) else np.nan,
            "R_given_wrong": round(gw.reversal.mean(), 4) if len(gw) else np.nan,
        })
    out = pd.DataFrame(rows)
    out["ord"] = out.condition.map({c: i for i, c in enumerate(ORDER)})
    return out.sort_values(["model", "ord"]).drop(columns="ord")


def paired_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    """Paired within-item contrasts for the control suite."""
    contrasts = [
        ("A3", "A1", "prestige increment P (expert vs anonymous)"),
        ("A3", "A2", "expert vs low-prestige"),
        ("A3", "A4", "expert vs neutral metadata"),
        ("A4", "A1", "neutral metadata vs bare anonymous"),
        ("A2", "A4", "low-prestige vs neutral metadata"),
        ("A1", "A0", "disagreement sensitivity G"),
        ("A3_agree", "A3", "expert agreement vs expert disagreement"),
        ("A3_agree", "A0", "expert agreement vs neutral reflection"),
        ("A3_agree", "A1_agree", "prestige effect within agreement"),
        ("A3g", "A3", "evidence gate on bare expert disagreement"),
        ("A3_rat", "A3", "padded expert+rationale prompt package vs answer-only"),
        ("A3_rat_gated", "A3_rat", "padded gated vs ungated rationale prompt package"),
    ]
    rows = []
    for model, g in df.groupby("model_id"):
        w = g.pivot_table(index="question_id", columns="condition",
                          values="reversal", aggfunc="first")
        for hi, lo, name in contrasts:
            if hi not in w or lo not in w:
                continue
            p = w[[hi, lo]].dropna()
            if len(p) == 0:
                continue
            a, b_ = p[hi].astype(bool), p[lo].astype(bool)
            bb = int((~b_ & a).sum())
            cc = int((b_ & ~a).sum())
            rows.append({
                "model": model, "contrast": f"{hi} - {lo}", "meaning": name,
                "n_paired": len(p),
                f"rate_hi": round(a.mean(), 4), f"rate_lo": round(b_.mean(), 4),
                "diff_pp": round(100 * (a.mean() - b_.mean()), 2),
                "discordant_hi_only": bb, "discordant_lo_only": cc,
                "mcnemar_p": mcnemar(bb, cc),
            })
    return pd.DataFrame(rows)


def agreement_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """Difference in the prestige contrast under disagreement vs agreement."""
    rows = []
    cols = ["A1", "A3", "A1_agree", "A3_agree"]
    for model, g in df.groupby("model_id"):
        w = g.pivot_table(index="question_id", columns="condition",
                          values="reversal", aggfunc="first")
        if not set(cols).issubset(w.columns):
            continue
        p = w[cols].dropna().astype(float)
        if p.empty:
            continue
        item_interaction = ((p.A3 - p.A1) -
                            (p.A3_agree - p.A1_agree)).to_numpy()
        rng = np.random.default_rng(9907 + sum(model.encode("utf-8")))
        values = []
        for start in range(0, 50_000, 1_000):
            idx = rng.integers(0, len(item_interaction),
                               size=(1_000, len(item_interaction)))
            values.extend(100 * item_interaction[idx].mean(axis=1))
        lo, hi = np.quantile(values, [0.025, 0.975])
        rows.append({
            "model": model,
            "n_paired": len(p),
            "prestige_effect_disagreement_pp": round(100 * (p.A3 - p.A1).mean(), 2),
            "prestige_effect_agreement_pp": round(100 * (p.A3_agree - p.A1_agree).mean(), 2),
            "interaction_pp": round(100 * item_interaction.mean(), 2),
            "interaction_paired_boot_ci_lo_pp": round(lo, 2),
            "interaction_paired_boot_ci_hi_pp": round(hi, 2),
        })
    return pd.DataFrame(rows)


def evidence_gate_discrimination(df: pd.DataFrame) -> pd.DataFrame:
    """Does the evidence gate retain beneficial revision?

    In the rationale conditions the opponent always contradicts Pass 1. Its
    assigned answer therefore matches the benchmark label exactly when Pass 1
    was wrong. This is a directional test, not independent factual validation of
    every generated rationale. A selective gate should keep reversal high in the
    first stratum and cut it in the second.
    """
    rows = []
    conds = ["A3", "A3g", "A3_rat", "A3_rat_gated"]
    for model, g in df.groupby("model_id"):
        for stratum, sub in (("Pass-1 wrong: target matches benchmark label",
                              g[~g.pass1_correct]),
                             ("Pass-1 correct: target conflicts with benchmark label",
                              g[g.pass1_correct])):
            w = sub.pivot_table(index="question_id", columns="condition",
                                values="reversal", aggfunc="first")
            have = [c for c in conds if c in w]
            p = w[have].dropna()
            if len(p) == 0:
                continue
            rec = {"model": model, "stratum": stratum, "n_paired": len(p)}
            for c in have:
                rec[f"rev_{c}"] = round(p[c].astype(bool).mean(), 4)
            if "A3_rat" in have and "A3_rat_gated" in have:
                a = p["A3_rat_gated"].astype(bool)
                b_ = p["A3_rat"].astype(bool)
                rec["gate_effect_pp"] = round(100 * (a.mean() - b_.mean()), 2)
                rec["gate_mcnemar_p"] = mcnemar(int((~b_ & a).sum()), int((b_ & ~a).sum()))
            rows.append(rec)
    return pd.DataFrame(rows)


def accuracy_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Which revision policy leaves the model most accurate? Pooled over models."""
    rows = []
    for cond, g in df.groupby("condition"):
        b = int((g.pass1_correct & ~g.pass2_correct).sum())
        c = int((~g.pass1_correct & g.pass2_correct).sum())
        rows.append({
            "condition": cond, "label": LABEL.get(cond, cond), "n": len(g),
            "pass1_acc": round(g.pass1_correct.mean(), 4),
            "final_acc": round(g.pass2_correct.mean(), 4),
            "delta_acc_pp": round(100 * (g.pass2_correct.mean() - g.pass1_correct.mean()), 2),
            "reversal": round(g.reversal.mean(), 4),
            "C_to_W": b, "W_to_C": c, "mcnemar_p": mcnemar(b, c),
        })
    out = pd.DataFrame(rows)
    out["ord"] = out.condition.map({c: i for i, c in enumerate(ORDER)})
    return out.sort_values("ord").drop(columns="ord")


def by_domain(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dom, cond), g in df.groupby(["domain", "condition"]):
        rows.append({"domain": dom, "condition": cond, "n": len(g),
                     "reversal": round(g.reversal.mean(), 4),
                     "delta_acc_pp": round(100 * (g.pass2_correct.mean() - g.pass1_correct.mean()), 2)})
    out = pd.DataFrame(rows)
    out["ord"] = out.condition.map({c: i for i, c in enumerate(ORDER)})
    return out.sort_values(["domain", "ord"]).drop(columns="ord")


def direction(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, cond), g in df.groupby(["model_id", "condition"]):
        for key, sub in (("pass1=Yes", g[g.pass1 == "Yes"]), ("pass1=No", g[g.pass1 == "No"])):
            rows.append({"model": model, "condition": cond, "stratum": key,
                         "n": len(sub), "reversal": round(sub.reversal.mean(), 4) if len(sub) else np.nan})
    out = pd.DataFrame(rows)
    out["ord"] = out.condition.map({c: i for i, c in enumerate(ORDER)})
    return out.sort_values(["model", "ord", "stratum"]).drop(columns="ord")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--fixed", default=None,
                    help="jsonl of rationale-condition rows from rerun_rationale.py")
    args = ap.parse_args()

    df = load(Path(args.path), Path(args.fixed) if args.fixed else None)
    if df.empty:
        print("no ok records yet")
        return
    print(f"models: {sorted(df.model_id.unique())}")
    print(f"items per model: {df.groupby('model_id').question_id.nunique().to_dict()}\n")

    tables = {
        f"controls_{args.tag}_per_condition.csv": per_condition(df),
        f"controls_{args.tag}_contrasts.csv": paired_contrasts(df),
        f"controls_{args.tag}_agreement_interaction.csv": agreement_interaction(df),
        f"controls_{args.tag}_gate_discrimination.csv": evidence_gate_discrimination(df),
        f"controls_{args.tag}_pooled_accuracy.csv": accuracy_ranking(df),
        f"controls_{args.tag}_by_domain.csv": by_domain(df),
        f"controls_{args.tag}_direction.csv": direction(df),
    }
    for name, t in tables.items():
        t.to_csv(OUT / name, index=False)
        print("=" * 100)
        print(name)
        print(t.to_string(index=False))


if __name__ == "__main__":
    main()
