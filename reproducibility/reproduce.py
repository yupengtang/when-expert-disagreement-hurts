"""Regenerate the camera-ready analyses and appendix tables without any API calls."""
from pathlib import Path
import contextlib
import io
import sys

import numpy as np
import pandas as pd
import newer_models
import core_display
import clean_display
import new_clean_audit

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "rebuttal"))
import analyze_main_audit as audit
import analyze_controls as controls
import analyze_clean_rationale_controls as clean
import analyze_derived_and_logits as logits
import analyze_gemini_and_accuracy as accuracy
import analyze_credibility_gradient as reliability
import analyze_temporal_original_pipeline as temporal
import coverage_tables

OUT = ROOT / "rebuttal/analysis"
TABLES = ROOT / "tables"
ORDER = ["GPT-4o-mini", "Claude-Sonnet-4", "Gemini-2.0-Flash", "Llama-3.1-8B", "Mistral-7B"]
NAMES = {"openai/gpt-4o-mini": "GPT-4o-mini", "anthropic/claude-sonnet-4": "Claude-Sonnet-4",
         "google/gemini-2.5-flash": "Gemini-2.5-Flash", "meta-llama/llama-3.1-8b-instruct": "Llama-3.1-8B",
         "meta-llama/llama-3.3-70b-instruct": "Llama-3.3-70B", "openai/gpt-5.6-terra": "GPT-5.6-terra",
         "x-ai/grok-4.5": "Grok-4.5", "google/gemini-3.6-flash": "Gemini-3.6-Flash",
         "anthropic/claude-opus-4.8": "Claude-Opus-4.8"}


def ptex(p):
    if p < .001:
        a, b = f"{p:.1e}".split("e")
        return rf"${float(a):g}\times10^{{{int(b)}}}$"
    return f"{p:.3f}" if p < .1 else f"{p:.2f}"


def table(name, caption, label, cols, headers, rows):
    text = [r"\begin{table}[htbp]", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
            r"\centering", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
            rf"\begin{{tabular}}{{{cols}}}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    text += [" & ".join(map(str, row)) + r" \\" for row in rows]
    text += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / name).write_text("\n".join(text) + "\n")


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    TABLES.mkdir(exist_ok=True)
    new_clean_audit.build(ROOT, OUT, TABLES, NAMES, table)
    # The original analysis functions preserve the response's estimands and seeds.
    with contextlib.redirect_stdout(io.StringIO()):
        audit.main()
        accuracy.main()
        reliability.main()
        temporal.main()
    df = audit.load()
    assert len(df) == 29778 and not df.duplicated(["model", "condition", "question_id"]).any()
    assert (df.reversal == (df.pass1 != df.pass2)).all()
    assert (df.pass1_correct == (df.pass1 == df.ground_truth)).all()
    assert (df.pass2_correct == (df.pass2 == df.ground_truth)).all()
    a3 = df[df.condition == "A3"]
    domain = a3.groupby("domain").agg(N=("reversal", "size"), pass1=("pass1_correct", "mean"),
                                     pass2=("pass2_correct", "mean"), reversal=("reversal", "mean"))
    domain.to_csv(OUT / "camera_domain_results.csv")
    high = a3[a3.confidence_tier == "High"]
    pd.DataFrame([{"n": len(high), "pass1_accuracy": high.pass1_correct.mean(),
                   "pass2_accuracy": high.pass2_correct.mean(),
                   "harmful_fraction": high[high.reversal].pass1_correct.mean()}]).to_csv(OUT / "camera_high_confidence.csv", index=False)
    table("domain.tex", r"Domain-stratified results under $A_3$, pooled over valid de-duplicated trials from the five full-benchmark models.",
          "tab:domain", "lrrrr", ["Domain", "$N$", "Pass-1 acc.", "Pass-2 acc.", "Reversal (\\%)"],
          [[d, f"{domain.loc[d,'N']:,}", f"{domain.loc[d,'pass1']:.3f}", f"{domain.loc[d,'pass2']:.3f}",
            f"{100*domain.loc[d,'reversal']:.1f}"] for d in ["Medicine", "Science", "Law"]])
    transition = []
    for m in ORDER:
        g = a3[a3.model == m]
        cw = int((g.pass1_correct & ~g.pass2_correct).sum()); wc = int((~g.pass1_correct & g.pass2_correct).sum())
        transition.append([m, len(g), cw, wc, f"{100*cw/(cw+wc):.1f}"])
    table("transitions.tex", r"Expert-disagreement transition counts on all valid $A_3$ trials.",
          "tab:transition_counts", "lrrrr", ["Model", "$N$", r"Correct$\to$Wrong", r"Wrong$\to$Correct", "Harmful (\\%)"], transition)
    strata = pd.read_csv(OUT / "prestige_by_stratum.csv")
    keys = ["pass1=Yes", "pass1=No", "ground_truth=Yes", "ground_truth=No"]
    table("strata.tex", r"Paired prestige effects by initial answer and true label. Cells give $P$ in percentage points and exact McNemar $p$; tests are exploratory and unadjusted.",
          "tab:strata", "lcccc", ["Model", "Initial Yes", "Initial No", "True Yes", "True No"],
          [[m] + [f"{(r:=strata[(strata.model==m)&(strata.stratum==k)].iloc[0]).P_pp:+.2f} ({ptex(r.mcnemar_p)})" for k in keys] for m in ORDER])
    direction_rows = []
    for m in ORDER:
        g = df[df.model == m]
        for field, value, label in [("pass1", "Yes", "Initial Yes"), ("pass1", "No", "Initial No"),
                                    ("ground_truth", "Yes", "True Yes"), ("ground_truth", "No", "True No")]:
            row = [m, label]
            for condition in ["A0", "A1", "A3"]:
                x = g[(g[field] == value) & (g.condition == condition)]
                row += [f"{100*x.reversal.mean():.1f} ({len(x)})"]
            direction_rows.append(row)
    table("direction_rates.tex", r"Reversal percentages by initial answer and true label, with valid $N$ in parentheses. Initial-Yes reversal is Yes$\to$No; initial-No reversal is No$\to$Yes.",
          "tab:direction_rates", "llccc", ["Model", "Stratum", "$A_0$", "$A_1$", "$A_3$"], direction_rows)
    interaction_rows = []
    suites = {}
    for tag, stem in [("main", "controls"), ("open", "controls_open"), ("sota", "controls_sota"), ("t0", "controls_t0")]:
        with contextlib.redirect_stdout(io.StringIO()):
            g = controls.load(ROOT / f"rebuttal/data/{stem}.jsonl", ROOT / f"rebuttal/data/{stem}_rat_fixed.jsonl")
        g = g[g.model_id.isin(NAMES)].copy()  # Exclude failed/unreported Qwen endpoint.
        assert not g.duplicated(["model_id", "question_id", "condition"]).any()
        suites[tag] = g
        for key, fn in [("per_condition", controls.per_condition), ("contrasts", controls.paired_contrasts),
                        ("agreement_interaction", controls.agreement_interaction),
                        ("gate_discrimination", controls.evidence_gate_discrimination),
                        ("direction", controls.direction)]:
            result = fn(g); result.to_csv(OUT / f"controls_{tag}_{key}.csv", index=False)
            if key == "agreement_interaction" and tag != "t0":
                for r in result.itertuples():
                    ci = f"{r.interaction_pp:+.2f} [{r.interaction_paired_boot_ci_lo_pp:+.2f}, {r.interaction_paired_boot_ci_hi_pp:+.2f}]"
                    interaction_rows.append([NAMES[r.model], r.n_paired, f"{r.prestige_effect_disagreement_pp:+.2f}",
                                             f"{r.prestige_effect_agreement_pp:+.2f}", ci])
    table("agreement.tex", r"Expert-label effects under disagreement and agreement and their interaction (pp). All columns in a row use the same four-condition intersection; CIs use 50,000 paired item resamples. Legacy padded suite.",
          "tab:agreement_interaction", "lrccc", ["Model", "$N$", "Disagreement", "Agreement", "Interaction [95\\% CI]"], interaction_rows)
    t0 = pd.read_csv(OUT / "controls_t0_contrasts.csv")
    t0 = t0[t0.contrast == "A3 - A1"]
    table("api_t0.tex", r"July API temperature-zero check: $A_1/A_3$ use the same paired items. This tests decoding within the padded follow-up suite, not temporal equivalence with January.",
          "tab:api_t0", "lrcccc", ["Model", "$N$", "$A_1$ (\\%)", "$A_3$ (\\%)", "$P$ (pp)", "$p$"],
          [[NAMES[r.model], r.n_paired, f"{100*r.rate_lo:.1f}", f"{100*r.rate_hi:.1f}", f"{r.diff_pp:+.1f}", ptex(r.mcnemar_p)] for r in t0.itertuples()])
    metadata_rows = []
    for tag in ["main", "sota"]:
        contrast = pd.read_csv(OUT / f"controls_{tag}_contrasts.csv")
        for r in contrast[contrast.contrast == "A3 - A4"].itertuples():
            metadata_rows.append([NAMES[r.model], r.n_paired, f"{100*r.rate_lo:.1f}", f"{100*r.rate_hi:.1f}", f"{r.diff_pp:+.1f}", ptex(r.mcnemar_p)])
    table("neutral.tex", r"Expert versus neutral source metadata on each pairwise intersection. Rates are percentages; effects are percentage points. Legacy padded suite.",
          "tab:neutral", "lrcccc", ["Model", "$N$", "$A_4$", "$A_3$", "$A_3-A_4$", "$p$"], metadata_rows)
    original_clean = clean.load([ROOT / "rebuttal/data/clean_rationale_250_dedup.jsonl"])
    additional_clean = clean.load([ROOT / "rebuttal/data/clean_rationale_sota_20260926.jsonl"])
    frame = pd.concat([original_clean, additional_clean], ignore_index=True)
    clean.condition_summary(frame).to_csv(OUT / "clean_conditions.csv", index=False)
    contrasts = pd.concat([clean.paired_summaries(original_clean),
                           clean.paired_summaries(additional_clean).query('model != "POOLED"')], ignore_index=True)
    contrasts.to_csv(OUT / "clean_paired_contrasts.csv", index=False)
    clean_display.build(frame, contrasts, NAMES, OUT, TABLES, ptex, table)
    gates = contrasts[(contrasts.contrast == "A3_rat_gated_clean - A3_rat_clean") &
                      (contrasts.model != "POOLED") & (contrasts.stratum != "all")]
    table("clean_gate.tex", r"No-padding gate control. Target matching the benchmark means Pass 1 was wrong; a conflicting target means it was correct. Rates are reversal percentages; $p$ is paired exact McNemar.",
          "tab:clean_gate", "llrcccc", ["Model", "Target", "$N$", "No gate", "Gate", "$\Delta$ (pp)", "$p$"],
          [[NAMES[r.model], "Matching" if r.stratum == "benchmark_matching" else "Conflicting", r.n_paired,
            f"{100*r.rate_low:.1f}", f"{100*r.rate_high:.1f}", f"{r.difference_pp:+.1f}", ptex(r.mcnemar_p)] for r in gates.itertuples()])
    # Preserve cross-model dependence by resampling questions, keeping their model block together.
    wide = original_clean.pivot(index=["question_id", "model_id"], columns="condition", values="reversal")
    differences = (wide.A3_rat_clean.astype(float) - wide.A1_rat_clean.astype(float)).groupby("question_id").mean().to_numpy()
    rng = np.random.default_rng(9907); boot = []
    for _ in range(50):
        idx = rng.integers(0, len(differences), size=(1000, len(differences)))
        boot.extend(100*differences[idx].mean(axis=1))
    lo, hi = np.quantile(boot, [.025, .975])
    pd.DataFrame([{"n_questions": len(differences), "n_model_item_pairs": len(wide), "P_pp": 100*differences.mean(),
                   "cluster_ci_lo": lo, "cluster_ci_hi": hi, "bootstrap_samples": 50000}]).to_csv(OUT / "pooled_cluster.csv", index=False)
    (TABLES / "pooled_ci.tex").write_text(f"{100*differences.mean():.2f} points (95\\% CI [{lo:.2f}, {hi:.2f}])")
    # The September models share one protocol, so they are pooled with each other and not with July.
    newer_models.pooled_clean(additional_clean).to_csv(OUT / "pooled_cluster_september.csv", index=False)
    # Report the exact complete-case sizes for current-model comparisons.
    newer_summary = newer_models.summarize(suites["sota"])
    newer_summary.to_csv(OUT / "newer_models_figure.csv", index=False)
    core = core_display.build(df, suites["sota"], newer_summary, pd.read_csv(OUT / "relative_effects.csv"), OUT, TABLES)
    coverage_tables.build(suites["sota"], core, OUT, TABLES, table, ptex)
    current = []
    for model, g in suites["sota"].groupby("model_id"):
        w = g.pivot(index="question_id", columns="condition", values="reversal")
        pair = w[["A1", "A3"]].dropna().astype(float)
        g3 = g[g.condition == "A3"]
        current.append([NAMES[model], f"{len(w.A0.dropna())}/{len(w.A1.dropna())}/{len(w.A3.dropna())}", len(pair),
                        f"{100*pair.A1.mean():.1f}", f"{100*pair.A3.mean():.1f}",
                        f"{100*(pair.A3-pair.A1).mean():+.2f}", f"{100*(g3.pass2_correct.mean()-g3.pass1_correct.mean()):+.1f}"])
    table("current_counts.tex", r"Retention for the four additional models out of 250 attempted items each. $P$ and reversal rates use the A1/A3 intersection; $\Delta$accuracy uses all valid $A_3$ outputs.",
          "tab:current_counts", "llrcccc", ["Model", "$N_0/N_1/N_3$", "$N_{13}$", "$A_1$", "$A_3$", "$P$", "$\Delta$acc."], current)
    for name, data in zip(["logit_mechanism_by_condition.csv", "logit_boundary_buckets.csv"], logits.logit_mechanism(df)):
        data.to_csv(OUT / name, index=False)
    print("Verified main rows, regenerated paired analyses and appendix tables without network/API calls.")
    print(f"Clean pooled effect with item clustering: {100*differences.mean():.2f} [{lo:.2f}, {hi:.2f}] pp")


if __name__ == "__main__":
    main()
