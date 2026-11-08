"""Appendix coverage tables for the valid records of the additional models."""
import numpy as np
import pandas as pd
from newer_models import MODELS
from analyze_controls import mcnemar
from tex_signs import math_signs


def build(frame, core, out, tables, table, ptex):
    def write(name, rows):
        (tables / name).write_text(math_signs("\n".join(" & ".join(map(str, row)) + r" \\" for row in rows) + "\n"))

    def append(name, rows):
        path = tables / name
        text = path.read_text()
        assert text.count(r"\bottomrule") == 1
        text = text.replace(r"\bottomrule", r"\midrule" + "\n" +
                            "\n".join(" & ".join(map(str, row)) + r" \\" for row in rows) + "\n" + r"\bottomrule")
        path.write_text(math_signs(text))

    transitions, strata, directions, conditional, relative, matched, rationale = [], [], [], [], [], [], []
    contrast = pd.read_csv(out / "controls_sota_contrasts.csv")
    condition = pd.read_csv(out / "controls_sota_per_condition.csv")
    for identifier, name in MODELS:
        g = frame[frame.model_id == identifier]
        g3 = g[g.condition == "A3"]
        cw = int((g3.pass1_correct & ~g3.pass2_correct).sum())
        wc = int((~g3.pass1_correct & g3.pass2_correct).sum())
        transitions.append([name, len(g3), cw, wc, f"{100*cw/(cw+wc):.1f}"])
        r = core[core.model == name].iloc[0]
        conditional.append([name, "/".join(f"{r[f'R_{c}_1']:.1f}" for c in ["A0", "A1", "A3"]),
                            "/".join(f"{r[f'R_{c}_0']:.1f}" for c in ["A0", "A1", "A3"]),
                            f"{100*r.pass1_acc:.1f}", f"{100*cw/(cw+wc):.1f}"])
        c = contrast[(contrast.model == identifier)&(contrast.contrast == "A3 - A1")].iloc[0]
        relative.append([name, f"{r.OR:.2f} [{r.OR_lo:.2f}, {r.OR_hi:.2f}]",
                         f"{r.RR:.2f} [{r.RR_lo:.2f}, {r.RR_hi:.2f}]",
                         f"{c.discordant_hi_only} / {c.discordant_lo_only}"])
        w = g.pivot(index="question_id", columns="condition", values="reversal")
        triple = w[["A0", "A1", "A3"]].dropna().astype(float)
        matched.append([name, len(triple)] + [f"{100*triple[c].mean():.1f}" for c in ["A0", "A1", "A3"]] +
                       [f"{100*(triple.A3-triple.A1).mean():+.1f}"])
        pair = w[["A1", "A3"]].dropna().astype(bool)
        meta = g[g.condition == "A1"].set_index("question_id").loc[pair.index]
        srow = [name]
        for field, value, label in [("pass1", "Yes", "Initial Yes"), ("pass1", "No", "Initial No"),
                                    ("ground_truth", "Yes", "True Yes"), ("ground_truth", "No", "True No")]:
            sub = pair[meta[field] == value]
            b, c = int((sub.A3 & ~sub.A1).sum()), int((sub.A1 & ~sub.A3).sum())
            srow.append(f"{100*(sub.A3.mean()-sub.A1.mean()):+.2f} ({ptex(mcnemar(b,c))})")
            dr = [name, label]
            for cond in ["A0", "A1", "A3"]:
                x = g[(g.condition == cond) & (g[field] == value)]
                dr.append(f"{100*x.reversal.mean():.1f} ({len(x)})")
            directions.append(dr)
        strata.append(srow)
        c = contrast[(contrast.model == identifier)&(contrast.contrast == "A3_rat - A3")].iloc[0]
        a3 = g[g.condition == "A3"]
        rat = g[g.condition == "A3_rat"]
        rationale.append([name, f"{100*a3.reversal.mean():.1f}", f"{100*rat.reversal.mean():.1f}",
                          f"{c.diff_pp:+.1f} ({ptex(c.mcnemar_p)})",
                          f"${100*(rat.pass2_correct.astype(int)-rat.pass1_correct.astype(int)).mean():+.1f}$"])
    append("transitions.tex", transitions)
    append("strata.tex", strata)
    append("direction_rates.tex", directions)
    write("newer_relative_rows.tex", relative)
    write("newer_conditional_rows.tex", conditional)
    write("newer_matched_rows.tex", matched)
    write("newer_rationale_rows.tex", rationale)
    # Separate domain results: the additional APIs did not use the original full benchmark.
    a3 = frame[frame.model_id.isin(dict(MODELS)) & (frame.condition == "A3")]
    domain = a3.groupby("domain").agg(N=("reversal", "size"), pass1=("pass1_correct", "mean"),
                                     pass2=("pass2_correct", "mean"), reversal=("reversal", "mean"))
    domain.to_csv(out / "additional_model_domains.csv")
    table("additional_domains.tex", r"Domain-stratified $A_3$ results for GPT-5.6-terra, Grok-4.5, Gemini-3.6-Flash, and Claude-Opus-4.8. These records use the 250-item pool, not the original full benchmark.",
          "tab:additional_domains", "lrrrr", ["Domain", "$N$", "Pass-1 acc.", "Pass-2 acc.", "Reversal (\\%)"],
          [[d, int(domain.loc[d,"N"]), f"{domain.loc[d,'pass1']:.3f}", f"{domain.loc[d,'pass2']:.3f}",
            f"{100*domain.loc[d,'reversal']:.1f}"] for d in ["Medicine", "Science", "Law"]])
    # Report saved gate effects, including non-improvements, under their actual padded protocol.
    rows = []
    for identifier, name in MODELS:
        for key, label in [("A3g - A3", "Answer only"), ("A3_rat_gated - A3_rat", "Rationale")]:
            r = contrast[(contrast.model == identifier)&(contrast.contrast == key)].iloc[0]
            rows.append([name, label, int(r.n_paired), f"{100*r.rate_lo:.1f}", f"{100*r.rate_hi:.1f}",
                         f"{r.diff_pp:+.2f}", ptex(r.mcnemar_p)])
    table("additional_gates.tex", r"Evidence-gate comparisons for the four additional models. Rates are paired reversal percentages; differences are percentage points and $p$ is exact McNemar. Padding changes with the gate, so these are prompt-package comparisons, not isolated gate effects.",
          "tab:additional_gates", "llrcccc", ["Model", "Context", "$N$", "No gate", "Gate", "$\Delta$ (pp)", "$p$"], rows)
