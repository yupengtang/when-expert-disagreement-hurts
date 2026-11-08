"""Seven-model no-padding analyses; never pool across collection periods."""
import json
import numpy as np
import pandas as pd
from tex_signs import math_signs

ORDER = ['openai/gpt-4o-mini', 'anthropic/claude-sonnet-4', 'google/gemini-2.5-flash',
         'openai/gpt-5.6-terra', 'x-ai/grok-4.5', 'google/gemini-3.6-flash', 'anthropic/claude-opus-4.8']


def build(frame, contrasts, names, out, tables, ptex, table):
    rows, results = [], []
    for i, model in enumerate(ORDER):
        g = frame[frame.model_id == model]
        assert not g.duplicated(['question_id', 'condition']).any()
        assert g.groupby('question_id').pass1.nunique().max() == 1
        for j, (high, low) in enumerate([
                ('A3_rat_clean', 'A1_rat_clean'), ('A3_rat_gated_clean', 'A3_rat_clean')]):
            r = contrasts[(contrasts.model == model) & (contrasts.stratum == 'all') &
                          (contrasts.contrast == f'{high} - {low}')].iloc[0]
            w = g.pivot(index='question_id', columns='condition', values='reversal')
            pair = w[[high, low]].dropna().astype(float)
            diff = (pair[high] - pair[low]).to_numpy()
            rng = np.random.default_rng(20260926 + 2*i + j)
            boot = np.concatenate([diff[rng.integers(len(diff), size=(1000, len(diff)))].mean(axis=1)
                                   for _ in range(50)]) * 100
            lo, hi = np.quantile(boot, [.025, .975])
            results.append({'model': names[model], 'contrast': 'label' if j == 0 else 'gate',
                            'n': len(pair), 'difference_pp': r.difference_pp,
                            'ci_lo': lo, 'ci_hi': hi, 'p': r.mcnemar_p,
                            'seed': 20260926+2*i+j, 'bootstrap_samples': 50000})
            if j == 0:
                a3 = g[g.condition == high]
                accuracy = 100*(a3.pass2_correct.mean()-a3.pass1_correct.mean())
                rows.append([names[model], len(pair), f'{100*r.rate_low:.1f}', f'{100*r.rate_high:.1f}',
                             f'{r.difference_pp:+.1f}', ptex(r.mcnemar_p), f'${accuracy:+.1f}$'])
    text = []
    for i, row in enumerate(rows):
        if i == 3:
            text.append(r'\midrule')
        text.append(' & '.join(map(str, row)) + r' \\')
    (tables / 'clean_model_rows.tex').write_text(math_signs('\n'.join(text) + '\n'))
    pd.DataFrame(results).to_csv(out / 'clean_seven_models.csv', index=False)
    table('clean_intervals.tex', r'No-padding paired contrasts. Source-label effects compare expert versus anonymous sources with identical rationales; gate effects compare gated versus ungated expert sources. CIs use 50,000 paired item resamples. Negative gate effects mean fewer reversals, not necessarily greater accuracy.',
          'tab:clean_intervals', 'llrcc', ['Model', 'Contrast', '$N$', 'Effect [95\\% CI] (pp)', '$p$'],
          [[r['model'], 'Source label' if r['contrast']=='label' else 'Gate', r['n'],
            f"{r['difference_pp']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]", ptex(r['p'])] for r in results])


def plot(frame, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, key, title in zip(axes, ['label', 'gate'], ['Expert − anonymous', 'Gate − no gate']):
        g = frame[frame.contrast == key]
        y = np.arange(len(g))
        ax.axvline(0, color='.5', linewidth=.8)
        ax.errorbar(g.difference_pp, y, xerr=[g.difference_pp-g.ci_lo, g.ci_hi-g.difference_pp],
                    fmt='o', capsize=3, color='#29608c')
        ax.set_title(title)
        ax.set_xlabel('Reversal difference (percentage points)')
        ax.set_yticks(y, g.model)
        ax.axhline(2.5, color='.8', linewidth=.8)
        ax.spines[['top','right']].set_visible(False)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
