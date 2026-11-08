#!/usr/bin/env python3
"""Main-audit tables and figures from the de-duplicated trial table.

load_trials() builds that table from raw run_experiment.py output; see
experiment/build_trial_table.py.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
FIGS = OUT / "figs"

QUESTIONS = OUT / "data-repair" / "outputs" / "original_benchmark_recovered.jsonl"
RESULTS = [OUT / "runs" / "main_audit_trials.jsonl"]

MODEL_NAMES = {
    "openai/gpt-4o-mini": "GPT-4o-mini",
    "anthropic/claude-sonnet-4": "Claude-Sonnet-4",
    "google/gemini-2.0-flash-001": "Gemini-2.0-Flash",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B",
}
INCLUDE_MODEL_IDS = set(MODEL_NAMES)
MODEL_ORDER = [
    "GPT-4o-mini",
    "Claude-Sonnet-4",
    "Gemini-2.0-Flash",
    "Llama-3.1-8B",
    "Mistral-7B",
]
OPEN_ORDER = ["Llama-3.1-8B", "Mistral-7B"]
REPS = ["A0", "A1", "A3"]


def load_questions():
    questions = {}
    with QUESTIONS.open() as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            questions[q["question_id"]] = {
                "ground_truth": q.get("ground_truth"),
                "domain": q.get("domain"),
                "confidence_tier": q.get("confidence_tier"),
            }
    return questions


def load_trials():
    questions = load_questions()
    rows = []
    seen = set()
    duplicates = {}
    for path in RESULTS:
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                rep = r.get("reputation_factor")
                if r.get("evidence_factor") != "B0" or rep not in REPS:
                    continue
                if r.get("model_id") not in INCLUDE_MODEL_IDS:
                    continue
                qid = r["question_id"]
                q = questions.get(qid)
                if not q or not q.get("ground_truth"):
                    continue
                p1 = r.get("pass1", {})
                p2 = r.get("pass2", {})
                if not p1.get("parsed_label") or not p2.get("parsed_label"):
                    continue
                if p1.get("truncated") or p2.get("truncated"):
                    continue
                if p1.get("format_violation") or p2.get("format_violation"):
                    continue

                key = (r["model_id"], rep, qid)
                if key in seen:
                    duplicates[(r["model_id"], rep)] = duplicates.get((r["model_id"], rep), 0) + 1
                    continue
                seen.add(key)

                gt = q["ground_truth"]
                l1 = p1.get("logodds_yes_over_no")
                l2 = p2.get("logodds_yes_over_no")
                rows.append(
                    {
                        "model": MODEL_NAMES.get(r["model_id"], r["model_id"]),
                        "model_id": r["model_id"],
                        "condition": rep,
                        "question_id": qid,
                        "domain": q.get("domain"),
                        "confidence_tier": q.get("confidence_tier"),
                        "pass1": p1["parsed_label"],
                        "pass2": p2["parsed_label"],
                        "ground_truth": gt,
                        "reversal": p1["parsed_label"] != p2["parsed_label"],
                        "pass1_correct": p1["parsed_label"] == gt,
                        "pass2_correct": p2["parsed_label"] == gt,
                        "logodds1": l1,
                        "logodds2": l2,
                        "duplicate_removed": duplicates.get((r["model_id"], rep), 0),
                    }
                )
    return pd.DataFrame(rows), duplicates


def write_tables(df, duplicates):
    df.to_csv(OUT / "neurips_dedup_trials.csv", index=False)

    rows = []
    for model in MODEL_ORDER:
        for rep in REPS:
            g = df[(df.model == model) & (df.condition == rep)]
            if g.empty:
                continue
            rows.append(
                {
                    "Model": model,
                    "Condition": rep,
                    "N": len(g),
                    "Pass-1 Acc": g.pass1_correct.mean(),
                    "Pass-2 Acc": g.pass2_correct.mean(),
                    "Delta Acc": g.pass2_correct.mean() - g.pass1_correct.mean(),
                    "Reversal Rate": g.reversal.mean(),
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "neurips_accuracy_table_dedup.csv", index=False)

    rows = []
    for model in MODEL_ORDER:
        g = df[(df.model == model) & (df.condition == "A3") & df.reversal]
        cw = (g.pass1_correct & ~g.pass2_correct).sum()
        wc = (~g.pass1_correct & g.pass2_correct).sum()
        rows.append(
            {
                "Model": model,
                "Reversals": len(g),
                "Correct_to_Wrong": int(cw),
                "Wrong_to_Correct": int(wc),
                "Harmful Percent": 100 * cw / len(g),
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "neurips_reversal_direction_dedup.csv", index=False)

    rows = []
    for model in MODEL_ORDER:
        rows.append(
            {
                "Model": model,
                "A0 valid": len(df[(df.model == model) & (df.condition == "A0")]),
                "A1 valid": len(df[(df.model == model) & (df.condition == "A1")]),
                "A3 valid": len(df[(df.model == model) & (df.condition == "A3")]),
                "Duplicates removed": sum(
                    duplicates.get((mid, rep), 0)
                    for mid, name in MODEL_NAMES.items()
                    if name == model
                    for rep in REPS
                ),
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "neurips_trial_accounting_dedup.csv", index=False)

    rows = []
    for model in MODEL_ORDER:
        mdf = df[df.model == model]
        valid_sets = [
            set(mdf[mdf.condition == rep].question_id)
            for rep in REPS
        ]
        if any(not s for s in valid_sets):
            continue
        matched = set.intersection(*valid_sets)
        row = {"model": model, "N_items": len(matched)}
        for rep in REPS:
            g = mdf[(mdf.condition == rep) & (mdf.question_id.isin(matched))]
            row[f"{rep}_rev"] = 100 * g.reversal.mean()
            row[f"{rep}_final"] = g.pass2_correct.mean()
            row[f"{rep}_pass1"] = g.pass1_correct.mean()
        row["P"] = row["A3_rev"] - row["A1_rev"]
        row["G"] = row["A1_rev"] - row["A0_rev"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "matched_item_robustness.csv", index=False)


def plot_accuracy(df):
    summary = df.groupby(["model", "condition"]).agg(delta=("pass2_correct", "mean")).reset_index()
    p1 = df.groupby(["model", "condition"]).agg(p1=("pass1_correct", "mean")).reset_index()
    summary = summary.merge(p1)
    summary["delta"] = summary["delta"] - summary["p1"]

    colors = {"A0": "#9ecae1", "A1": "#6baed6", "A3": "#2171b5"}
    labels = {"A0": "Drift", "A1": "Anon", "A3": "Expert"}
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    width = 0.24
    x = range(len(MODEL_ORDER))
    for i, rep in enumerate(REPS):
        vals = [
            100 * summary[(summary.model == model) & (summary.condition == rep)].delta.iloc[0]
            for model in MODEL_ORDER
        ]
        xpos = [j + (i - 1) * width for j in x]
        ax.bar(xpos, vals, width=width, label=labels[rep], color=colors[rep], edgecolor="black", linewidth=0.3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Pass-2 minus Pass-1 accuracy (pp)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(MODEL_ORDER, rotation=25, ha="right")
    ax.set_ylim(-18, 3)
    ax.legend(frameon=False, ncol=3, loc="lower left")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(FIGS / "accuracy_delta_barplot.pdf")


def plot_transition_stacked(df):
    a3 = df[df.condition == "A3"].copy()
    categories = [
        ("Stable correct", lambda g: g.pass1_correct & g.pass2_correct, "#8bcf8f"),
        ("Wrong$\\to$Correct", lambda g: ~g.pass1_correct & g.pass2_correct, "#3b6fb6"),
        ("Correct$\\to$Wrong", lambda g: g.pass1_correct & ~g.pass2_correct, "#c73e3a"),
        ("Stable wrong", lambda g: ~g.pass1_correct & ~g.pass2_correct, "#c9c9c9"),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(MODEL_ORDER))
    left = np.zeros(len(MODEL_ORDER))

    for label, mask_fn, color in categories:
        vals = []
        for model in MODEL_ORDER:
            g = a3[a3.model == model]
            vals.append(100 * mask_fn(g).mean())
        vals = np.array(vals)
        ax.barh(y, vals, left=left, height=0.62, label=label, color=color, edgecolor="white", linewidth=0.7)
        if label in {"Wrong$\\to$Correct", "Correct$\\to$Wrong"}:
            for i, val in enumerate(vals):
                if val >= 2.8:
                    ax.text(left[i] + val / 2, i, f"{val:.1f}", ha="center", va="center", fontsize=8, color="white")
        left += vals

    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of expert-labeled trials (%)")
    ax.set_yticks(y)
    ax.set_yticklabels(MODEL_ORDER)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.22, linewidth=0.5)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "transition_stacked_a3.pdf")


def bootstrap_p_interval(df, model, n_boot=4000, seed=7):
    rng = np.random.default_rng(seed)
    mdf = df[(df.model == model) & (df.condition.isin(["A1", "A3"]))]
    piv = mdf.pivot_table(index="question_id", columns="condition", values="reversal", aggfunc="first")
    piv = piv.dropna(subset=["A1", "A3"])
    vals = piv[["A1", "A3"]].astype(float).to_numpy()
    n = len(vals)
    if n == 0:
        return np.nan, np.nan, np.nan
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sample = vals[rng.integers(0, n, n)]
        diffs[i] = 100 * (sample[:, 1].mean() - sample[:, 0].mean())
    point = 100 * (vals[:, 1].mean() - vals[:, 0].mean())
    low, high = np.percentile(diffs, [2.5, 97.5])
    return point, low, high


def plot_reversal(df):
    colors = {"A0": "#9ecae1", "A1": "#6baed6", "A3": "#2171b5"}
    labels = {"A0": "Drift", "A1": "Anon", "A3": "Expert"}
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    width = 0.24
    x = range(len(MODEL_ORDER))
    for i, rep in enumerate(REPS):
        vals = [
            100 * df[(df.model == model) & (df.condition == rep)].reversal.mean()
            for model in MODEL_ORDER
        ]
        xpos = [j + (i - 1) * width for j in x]
        bars = ax.bar(xpos, vals, width=width, label=labels[rep], color=colors[rep], edgecolor="black", linewidth=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Reversal rate (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(MODEL_ORDER, rotation=25, ha="right")
    ax.set_ylim(0, 27)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(FIGS / "expertise_effect.pdf")

    fig, (ax_bar, ax_p) = plt.subplots(
        1,
        2,
        figsize=(7.4, 3.8),
        gridspec_kw={"width_ratios": [2.2, 1.0], "wspace": 0.08},
        sharey=True,
    )
    colors = {"A0": "#c6dbef", "A1": "#6baed6", "A3": "#08519c"}
    labels = {"A0": "Drift", "A1": "Anon", "A3": "Expert"}
    y = np.arange(len(MODEL_ORDER))
    height = 0.22
    offsets = {"A0": -height, "A1": 0.0, "A3": height}
    for rep in REPS:
        vals = [
            100 * df[(df.model == model) & (df.condition == rep)].reversal.mean()
            for model in MODEL_ORDER
        ]
        ax_bar.barh(
            y + offsets[rep],
            vals,
            height=height,
            label=labels[rep],
            color=colors[rep],
            edgecolor="black",
            linewidth=0.25,
        )
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(MODEL_ORDER)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Reversal rate (%)")
    ax_bar.set_xlim(0, 27)
    ax_bar.grid(axis="x", alpha=0.22, linewidth=0.5)
    ax_bar.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.52, 1.01), fontsize=8)

    points, lows, highs = [], [], []
    for model in MODEL_ORDER:
        point, low, high = bootstrap_p_interval(df, model)
        points.append(point)
        lows.append(low)
        highs.append(high)
    points = np.array(points)
    lows = np.array(lows)
    highs = np.array(highs)
    xerr = np.vstack([points - lows, highs - points])
    p_colors = ["#08519c" if p >= 0 else "#b35806" for p in points]
    ax_p.axvline(0, color="black", linewidth=0.8)
    ax_p.errorbar(points, y, xerr=xerr, fmt="none", ecolor="#555555", elinewidth=1.0, capsize=2.5, zorder=1)
    ax_p.scatter(points, y, s=34, color=p_colors, edgecolor="black", linewidth=0.35, zorder=2)
    ax_p.set_xlabel("$P=A_3-A_1$ (pp)")
    ax_p.set_xlim(-5, 10)
    ax_p.grid(axis="x", alpha=0.22, linewidth=0.5)
    ax_p.tick_params(axis="y", left=False, labelleft=False)
    for spine in ["left"]:
        ax_p.spines[spine].set_visible(False)

    fig.subplots_adjust(left=0.18, right=0.98, top=0.82, bottom=0.18, wspace=0.08)
    fig.savefig(FIGS / "reversal_decomposition.pdf")


def plot_open_weight(df):
    odf = df[df.model.isin(OPEN_ORDER) & df.logodds1.notna() & df.logodds2.notna()].copy()
    odf["delta_logodds"] = odf.logodds2 - odf.logodds1
    odf["sign_flip"] = (odf.logodds1 >= 0) != (odf.logodds2 >= 0)

    labels, mismatch, signchange = [], [], []
    for model in OPEN_ORDER:
        for rep in REPS:
            rev = odf[(odf.model == model) & (odf.condition == rep) & odf.reversal]
            labels.append(f"{model}\n{rep}")
            mismatch.append((~rev.sign_flip).sum())
            signchange.append(rev.sign_flip.sum())
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    x = range(len(labels))
    ax.bar(x, signchange, label="Preference sign changed", color="#74c476", edgecolor="black", linewidth=0.3)
    ax.bar(x, mismatch, bottom=signchange, label="Mismatch", color="#fb6a4a", edgecolor="black", linewidth=0.3)
    ax.set_ylabel("Reversed trials")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(FIGS / "taxonomy_breakdown.pdf")

    labels, vals = [], []
    for model in OPEN_ORDER:
        for rep in REPS:
            g = odf[(odf.model == model) & (odf.condition == rep)]
            labels.append(f"{model}\n{rep}")
            vals.append(g.delta_logodds.mean())
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.bar(range(len(vals)), vals, color=["#9ecae1", "#6baed6", "#2171b5"] * 2, edgecolor="black", linewidth=0.3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean $\\Delta$ log-odds")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(FIGS / "delta_logodds.pdf")


def main():
    df, duplicates = load_trials()
    write_tables(df, duplicates)
    plot_accuracy(df)
    plot_transition_stacked(df)
    plot_reversal(df)
    plot_open_weight(df)


if __name__ == "__main__":
    main()
