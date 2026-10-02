"""Integrate recorded January and July cohorts without pooling their estimands."""
from pathlib import Path
import numpy as np
import pandas as pd
from newer_models import MODELS

ORIGINAL = ["GPT-4o-mini", "Claude-Sonnet-4", "Gemini-2.0-Flash", "Llama-3.1-8B", "Mistral-7B"]
ORDER = ORIGINAL + [name for _, name in MODELS]


def build(original, newer, new_summary, relative, output, tables):
    new = newer[newer.model_id.isin(dict(MODELS)) & newer.condition.isin(["A0", "A1", "A3"])].copy()
    new["model"] = new.model_id.map(dict(MODELS))
    columns = ["model", "question_id", "condition", "pass1_correct", "pass2_correct", "reversal"]
    df = pd.concat([original[columns], new[columns]], ignore_index=True)
    rows = []
    for name in ORDER:
        g = df[df.model == name]
        a3 = g[g.condition == "A3"]
        pair = g.pivot(index="question_id", columns="condition", values="reversal")[["A1", "A3"]].dropna().astype(float)
        row = dict(model=name, cohort="full_benchmark" if name in ORIGINAL else "250_item_controls",
                   n_paired=len(pair), pass1_acc=a3.pass1_correct.mean(),
                   delta_acc=100*(a3.pass2_correct.astype(int)-a3.pass1_correct.astype(int)).mean(),
                   A1=100*pair.A1.mean(), A3=100*pair.A3.mean())
        for c in ["A0", "A1", "A3"]:
            gc = g[g.condition == c]
            row[f"n_{c}"] = len(gc)
            for correct in [True, False]:
                subset = gc[gc.pass1_correct == correct]
                row[f"R_{c}_{int(correct)}"] = 100*subset.reversal.mean()
        row["A0"] = 100*g[g.condition == "A0"].reversal.mean()
        row["selectivity"] = row["R_A3_0"] / row["R_A3_1"]
        for key, mask in [("stable_correct", a3.pass1_correct & a3.pass2_correct),
                          ("wrong_correct", ~a3.pass1_correct & a3.pass2_correct),
                          ("correct_wrong", a3.pass1_correct & ~a3.pass2_correct),
                          ("stable_wrong", ~a3.pass1_correct & ~a3.pass2_correct)]:
            row[key] = 100*mask.mean()
        assert np.isclose(sum(row[k] for k in ["stable_correct", "wrong_correct", "correct_wrong", "stable_wrong"]), 100)
        if name in ORIGINAL:
            r = relative.set_index("model").loc[name]
            row.update(P=r.P_additive_pp, P_lo=r.P_paired_boot_ci_lo_pp, P_hi=r.P_paired_boot_ci_hi_pp,
                       RR=r.risk_ratio, RR_lo=r.RR_paired_boot_ci_lo, RR_hi=r.RR_paired_boot_ci_hi, p=r.mcnemar_p)
        else:
            from analyze_controls import mcnemar
            r = new_summary.set_index("model").loc[name]
            x, y = pair.A1.to_numpy(), pair.A3.to_numpy()
            rng = np.random.default_rng(int(r.prestige_seed)+1000)
            draws, odds_draws = [], []
            with np.errstate(divide="ignore", invalid="ignore"):
                for _ in range(50):
                    idx = rng.integers(0, len(pair), size=(1000, len(pair)))
                    bx, by = x[idx].mean(axis=1), y[idx].mean(axis=1)
                    draws.extend(by/bx)
                    odds_draws.extend((by/(1-by))/(bx/(1-bx)))
            draws = np.asarray(draws)
            # 0/0 is undefined; retain +infinity when only the denominator is zero.
            lo, hi = np.quantile(draws[~np.isnan(draws)], [.025, .975])
            odds_draws = np.asarray(odds_draws)
            olo, ohi = np.quantile(odds_draws[~np.isnan(odds_draws)], [.025, .975])
            row.update(P=r.prestige_pp, P_lo=r.prestige_ci_lo, P_hi=r.prestige_ci_hi,
                       RR=y.mean()/x.mean(), RR_lo=lo, RR_hi=hi,
                       RR_undefined_draws=int(np.isnan(draws).sum()), RR_seed=int(r.prestige_seed)+1000,
                       OR=(y.mean()/(1-y.mean()))/(x.mean()/(1-x.mean())), OR_lo=olo, OR_hi=ohi,
                       OR_undefined_draws=int(np.isnan(odds_draws).sum()),
                       p=mcnemar(int(((x==0)&(y==1)).sum()), int(((x==1)&(y==0)).sum())))
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(output / "core_nine_models.csv", index=False)
    def write(name, content):
        (tables / name).write_text("\n".join(content) + "\n")
    subset = result[~result.model.isin(ORIGINAL)]
    write("newer_accounting_rows.tex", [f"{r.model} & {r.n_A0} & {r.n_A1} & {r.n_A3} & 0 \\\\" for r in subset.itertuples()])
    write("newer_accuracy_rows.tex", [
        f"{r.model} & {r.pass1_acc:.3f} & ${r.delta_acc:+.1f}$ & ${r.R_A1_1:.1f}\\to{r.R_A3_1:.1f}$ & ${r.R_A1_0:.1f}\\to{r.R_A3_0:.1f}$ & {r.selectivity:.2f} \\\\"
        for r in subset.itertuples()])
    def ptex(p):
        if p < .001:
            a, b = f"{p:.1e}".split("e")
            return f"${float(a):g}\\times10^{{{int(b)}}}$"
        return f"{p:.3f}"
    write("newer_reversal_rows.tex", [
        f"{r.model} & {r.A0:.1f} & {r.A1:.1f} & {r.A3:.1f} & ${r.P:+.1f}$ [{r.P_lo:.1f}, {r.P_hi:.1f}] & {r.RR:.2f} [{r.RR_lo:.2f}, {r.RR_hi:.2f}] & {ptex(r.p)} \\\\"
        for r in subset.itertuples()])
    return result


def plot(summary, directory):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    summary = summary.set_index('model').loc[ORDER].reset_index()
    y = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9])
    ink, blue, orange, green = '#263442', '#245A81', '#D55E00', '#009E73'
    plt.rcParams.update({'svg.fonttype': 'none', 'svg.hashsalt': 'expert-audit',
                         'text.color': ink, 'axes.labelcolor': ink,
                         'xtick.color': ink, 'ytick.color': ink})

    def decorate(ax):
        ax.set_ylim(9.7, -1.4)
        ax.axhspan(5.45, 9.7, color='#F3F6F8', zorder=0)
        ax.grid(axis='x', color='#E4E9ED', linewidth=.55)
        ax.set_axisbelow(True)
        for side in ['top', 'right', 'left']:
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color('#B6C1CB')
        ax.tick_params(axis='y', length=0)
        ax.tick_params(axis='x', labelsize=8, length=3)

    def groups(ax):
        for yy, label in [(-.95, '1,989-item benchmark'), (5.12, '250-item pool · different protocol')]:
            ax.text(0, yy, label, transform=ax.get_yaxis_transform(),
                    fontsize=7.5, color='#596976', va='center')

    def save(fig, name):
        for extension in ['pdf', 'svg']:
            path = directory / f'{name}.{extension}'
            metadata = {'Creator': 'Reproducible paired audit'}
            if extension == 'svg':
                metadata['Date'] = None
            fig.savefig(path, metadata=metadata)
            if extension == 'svg':
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
        plt.close(fig)

    # Both directions use all valid A3 trials, not separate correctness strata.
    fig, (ax, net) = plt.subplots(1, 2, figsize=(6.8, 3.4), sharey=True,
                                 gridspec_kw={'width_ratios': [3.6, 1.15]})
    fig.subplots_adjust(left=.235, right=.98, top=.85, bottom=.16, wspace=.1)
    decorate(ax)
    groups(ax)
    ax.axvline(0, color='#7C8A96', linewidth=.8)
    ax.barh(y, -summary.correct_wrong, height=.58, color=orange, label='Correct → Wrong')
    ax.barh(y, summary.wrong_correct, height=.58, color=green, label='Wrong → Correct')
    for yy, row in zip(y, summary.itertuples()):
        ax.text(-row.correct_wrong-.5, yy, f'{row.correct_wrong:.1f}', va='center', ha='right', fontsize=7.6)
        ax.text(row.wrong_correct+.5, yy, f'{row.wrong_correct:.1f}', va='center', fontsize=7.6)
    ax.set_xlim(-25, 25)
    ax.set_xticks([-20, -10, 0, 10, 20])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{abs(x):g}'))
    ax.set_yticks(y, summary.model, fontsize=8.7)
    ax.set_xlabel('Share of valid expert-labeled trials (%)', fontsize=8.5)
    ax.legend(frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(.5, 1.015), fontsize=8,
              handlelength=1.1, columnspacing=1.5)
    net.set_xlim(0, 1)
    net.axis('off')
    net.text(.19, 1.03, 'Δ accuracy', transform=net.transAxes, ha='center', fontsize=8.3, weight='medium')
    net.text(.19, .975, '(pp)', transform=net.transAxes, ha='center', fontsize=7.5)
    net.text(.84, 1.03, 'Valid n', transform=net.transAxes, ha='center', fontsize=8.3)
    for yy, row in zip(y, summary.itertuples()):
        assert np.isclose(row.wrong_correct-row.correct_wrong, row.delta_acc)
        net.text(.19, yy, f'{row.delta_acc:+.1f}', ha='center', va='center', fontsize=8.5)
        net.text(.84, yy, f'{row.n_A3:,}', ha='center', va='center', fontsize=8)
    # Keep the established filename so existing source exports remain compatible.
    save(fig, 'transition_stacked_a3')

    fig, (effects, rates) = plt.subplots(1, 2, figsize=(6.8, 3.4), sharey=True,
                                       gridspec_kw={'width_ratios': [1.3, 1]})
    fig.subplots_adjust(left=.235, right=.98, top=.85, bottom=.16, wspace=.2)
    for panel in [effects, rates]:
        decorate(panel)
    groups(effects)
    effects.axvline(0, color='#7C8A96', linewidth=.8)
    effects.errorbar(summary.P, y, xerr=np.vstack([summary.P-summary.P_lo, summary.P_hi-summary.P]),
                     fmt='o', color=blue, markersize=4, capsize=2.5, linewidth=1.1)
    effects.set_xlim(-5, 11)
    effects.set_xticks([-5, 0, 5, 10])
    effects.set_yticks(y, summary.model, fontsize=8.7)
    effects.set_title('Expert-label effect', fontsize=9.5, pad=11, weight='medium')
    effects.set_xlabel('Expert − anonymous reversal (pp)', fontsize=8.5)
    for offset, key, label, color, marker in [(-.22, 'A0', 'Drift', '#87929D', 's'),
                                             (0, 'A1', 'Anon.', blue, 'o'),
                                             (.22, 'A3', 'Expert', orange, '^')]:
        rates.plot(summary[key], y+offset, linestyle='none', marker=marker, markersize=3.7,
                   color=color, label=label)
    rates.set_xlim(0, 27)
    rates.set_xticks([0, 10, 20])
    rates.set_xlabel('Reversal rate (%)', fontsize=8.5)
    rates.tick_params(axis='y', labelleft=False)
    rates.legend(frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(.5, 1.015),
                 fontsize=7.7, handlelength=.65, columnspacing=.7)
    save(fig, 'reversal_decomposition')
