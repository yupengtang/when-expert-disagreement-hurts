"""Saved-output estimates and paired item-bootstrap intervals for the newer APIs."""
import numpy as np
import pandas as pd

MODELS = [
    ("openai/gpt-5.6-terra", "GPT-5.6-terra"),
    ("x-ai/grok-4.5", "Grok-4.5"),
    ("google/gemini-3.6-flash", "Gemini-3.6-Flash"),
    ("anthropic/claude-opus-4.8", "Claude-Opus-4.8"),
]


def interval(difference, seed):
    """Resample paired question differences, not independent condition margins."""
    values = np.asarray(difference, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.concatenate([
        values[rng.integers(0, len(values), size=(1000, len(values)))].mean(axis=1)
        for _ in range(50)
    ])
    return 100 * np.quantile(draws, [.025, .975])


def summarize(frame):
    rows = []
    for i, (identifier, name) in enumerate(MODELS):
        g = frame[frame.model_id == identifier]
        assert not g.duplicated(["question_id", "condition"]).any()
        w = g.pivot(index="question_id", columns="condition", values="reversal")
        pair = w[["A1", "A3"]].dropna().astype(float)
        initial = g.pivot(index="question_id", columns="condition", values="pass1")
        assert (initial.loc[pair.index, "A1"] == initial.loc[pair.index, "A3"]).all()
        difference = pair.A3 - pair.A1
        g3 = g[g.condition == "A3"].sort_values("question_id")
        accuracy_difference = g3.pass2_correct.astype(int) - g3.pass1_correct.astype(int)
        assert (g3.reversal == (g3.pass1 != g3.pass2)).all()
        seed = 20260926 + 2 * i
        lo, hi = interval(difference, seed)
        alo, ahi = interval(accuracy_difference, seed + 1)
        rows.append(dict(model_id=identifier, model=name, n_paired=len(pair), n_a3=len(g3),
                         prestige_pp=100*difference.mean(), prestige_ci_lo=lo, prestige_ci_hi=hi,
                         accuracy_pp=100*accuracy_difference.mean(), accuracy_ci_lo=alo, accuracy_ci_hi=ahi,
                         bootstrap_samples=50000, prestige_seed=seed, accuracy_seed=seed+1))
    return pd.DataFrame(rows)


def pooled_clean(frame, seed=20261104):
    """Pool the four September no-padding contrasts, resampling questions with their model block."""
    wide = frame.pivot(index=["question_id", "model_id"], columns="condition", values="reversal")
    pair = wide[["A1_rat_clean", "A3_rat_clean"]].dropna().astype(float)
    grouped = (pair.A3_rat_clean - pair.A1_rat_clean).groupby("question_id").agg(["sum", "size"])
    total, size = grouped["sum"].to_numpy(), grouped["size"].to_numpy()
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(50):
        idx = rng.integers(0, len(total), size=(1000, len(total)))
        draws.append(total[idx].sum(axis=1) / size[idx].sum(axis=1))
    lo, hi = 100 * np.quantile(np.concatenate(draws), [.025, .975])
    return pd.DataFrame([dict(n_questions=len(total), n_model_item_pairs=int(size.sum()),
                              expert_only=int((pair.A3_rat_clean > pair.A1_rat_clean).sum()),
                              anonymous_only=int((pair.A1_rat_clean > pair.A3_rat_clean).sum()),
                              P_pp=100 * total.sum() / size.sum(), cluster_ci_lo=lo, cluster_ci_hi=hi,
                              bootstrap_samples=50000, seed=seed)])


def plot(summary, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1, 1]})
    fig.subplots_adjust(left=.24, right=.98, bottom=.23, top=.82, wspace=.32)
    for ax, metric, color, title, limits, ticks, count in [
        (axes[0], "prestige", "#1764a0", "(a) Expert-label effect", (-4, 16), [-4, 0, 4, 8, 12], "n_paired"),
        (axes[1], "accuracy", "#b54b34", "(b) Accuracy change under expert", (-15, 8), [-12, -8, -4, 0, 4], "n_a3"),
    ]:
        ax.axvline(0, color="0.4", linewidth=.8, linestyle="--", zorder=1)
        for y, row in enumerate(summary.to_dict("records")):
            value, lo, hi = row[f"{metric}_pp"], row[f"{metric}_ci_lo"], row[f"{metric}_ci_hi"]
            ax.errorbar(value, y, xerr=[[value-lo], [hi-value]], fmt="o", markersize=4,
                        color=color, capsize=3, linewidth=1.2, zorder=3)
            ax.annotate(f"{value:+.2f}", (value, y), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8, color=color)
            ax.text(.98, y, f"n={row[count]}", transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=7, color="0.3")
        ax.set_ylim(3.5, -.65)
        ax.set_xlim(*limits)
        ax.set_xticks(ticks)
        ax.set_yticks(range(4))
        ax.set_yticklabels(summary.model if metric == "prestige" else [""]*4, fontsize=9)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", alpha=.16)
        ax.set_title(title, fontsize=9, pad=15)
        ax.set_xlabel("Reversal: expert − anonymous (pp)" if metric == "prestige"
                      else "Accuracy: Pass 2 − Pass 1 (pp)", fontsize=8)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
