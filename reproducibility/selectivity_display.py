"""Selectivity of revision relative to the break-even ratio implied by initial accuracy.

For a binary decision, revision under condition A changes accuracy by
(1 - a) * b_A - a * h_A, where a is Pass-1 accuracy, h_A the reversal rate of
initially correct answers and b_A that of initially wrong answers. Revision
raises accuracy only when b_A / h_A exceeds a / (1 - a), so the plotted
quantity (b_A / h_A) / (a / (1 - a)) is above 1 exactly when it does.
"""
import numpy as np

from core_display import ORDER, ORIGINAL


def ratios(summary):
    rows = summary.set_index('model').loc[ORDER].reset_index()
    breakeven = rows.pass1_acc / (1 - rows.pass1_acc)
    with np.errstate(divide='ignore', invalid='ignore'):
        rows['rel_A1'] = (rows.R_A1_0 / rows.R_A1_1) / breakeven
        rows['rel_A3'] = (rows.R_A3_0 / rows.R_A3_1) / breakeven
    rows['breakeven'] = breakeven
    # The identity ties the side of 1 to the sign of the accuracy change.
    for r in rows.itertuples():
        assert (r.rel_A3 > 1) == (r.delta_acc > 0), r.model
    return rows


def plot(summary, path):
    import matplotlib.pyplot as plt
    rows = ratios(summary)
    y = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9])
    ink, blue, orange, green = '#263442', '#245A81', '#D55E00', '#009E73'
    plt.rcParams.update({'svg.fonttype': 'none', 'svg.hashsalt': 'expert-audit',
                         'text.color': ink, 'axes.labelcolor': ink,
                         'xtick.color': ink, 'ytick.color': ink})
    fig, ax = plt.subplots(figsize=(6.8, 3.1))
    fig.subplots_adjust(left=.235, right=.885, top=.84, bottom=.17)
    lo, hi = .09, 2.2
    ax.set_xscale('log')
    ax.set_xlim(lo, hi)
    ax.set_ylim(9.7, -1.4)
    ax.axvspan(lo, 1, color='#FBEDE3', zorder=0)
    ax.axvspan(1, hi, color='#E4F3EC', zorder=0)
    ax.axhspan(5.45, 9.7, color='#EEF1F4', alpha=.55, zorder=0)
    ax.axvline(1, color=ink, linewidth=.9, zorder=1)
    ax.text(.94, -1.05, 'revision lowers accuracy', ha='right', va='center', fontsize=7.6, color=orange)
    ax.text(1.06, -1.05, 'raises', ha='left', va='center', fontsize=7.6, color=green)
    ax.text(1.03, -1.05, 'Expert', transform=ax.get_yaxis_transform(), ha='left', va='center',
            fontsize=7.6, color='#596976', clip_on=False)
    for yy, r in zip(y, rows.itertuples()):
        if np.isfinite(r.rel_A1) and r.rel_A1 > 0:
            ax.annotate('', xy=(r.rel_A3, yy), xytext=(r.rel_A1, yy),
                        arrowprops=dict(arrowstyle='->', color='#8A97A3', lw=.9,
                                        shrinkA=3.2, shrinkB=3.6), zorder=2)
            ax.plot(r.rel_A1, yy, 'o', markersize=4.6, markerfacecolor='white',
                    markeredgecolor=blue, markeredgewidth=1, zorder=3)
        ax.plot(r.rel_A3, yy, 'o', markersize=4.8, color=orange, zorder=4)
        ax.text(1.03, yy, f'{r.rel_A3:.2f}', transform=ax.get_yaxis_transform(),
                ha='left', va='center', fontsize=7.6, clip_on=False)
    ax.plot([], [], 'o', markersize=4.6, markerfacecolor='white', markeredgecolor=blue,
            linestyle='none', label='Anonymous ($A_1$)')
    ax.plot([], [], 'o', markersize=4.8, color=orange, linestyle='none', label='Expert ($A_3$)')
    ax.legend(frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(.5, 1.02),
              fontsize=8, handlelength=.9, columnspacing=1.4)
    for yy, label in [(-.95, '1,989-item benchmark'), (5.12, '250-item pool · different protocol')]:
        ax.text(0, yy, label, transform=ax.get_yaxis_transform(), fontsize=7.5,
                color='#596976', va='center')
    ax.set_xticks([.1, .25, .5, 1, 2], ['0.1', '0.25', '0.5', '1', '2'])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yticks(y, rows.model, fontsize=8.7)
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', labelsize=8, length=3)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#B6C1CB')
    ax.set_xlabel('Selectivity relative to break-even:  $(b/h)\\,/\\,(a/(1-a))$', fontsize=8.5)
    metadata = {'Creator': 'Reproducible paired audit'}
    fig.savefig(path.with_suffix('.pdf'), metadata=metadata)
    fig.savefig(path.with_suffix('.svg'), metadata={**metadata, 'Date': None})
    svg = path.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)
    return rows
