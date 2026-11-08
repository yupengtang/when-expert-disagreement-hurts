"""Export the project-page figures and paired-effect data from reproducibility/.

Run from the repository root after reproduce.py, make_figures.py, reanalysis/audit.py
and reanalysis/render.py:
    python .github/scripts/export_page_assets.py
The script copies three paper figures into docs/assets/ and writes docs/assets/results.json.
"""
import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / 'reproducibility'
CORE = ARTIFACT / 'results/analysis/core_nine_models.csv'
PAIRED = ARTIFACT / 'reanalysis/outputs/accuracy_decomposition.csv'
FIGURES = [(ARTIFACT / 'figs/transition_stacked_a3.svg', 'transitions.svg'),
           (ARTIFACT / 'figs/selectivity_breakeven.svg', 'selectivity.svg'),
           (ARTIFACT / 'reanalysis/outputs/source_accuracy_effects.svg', 'reversal.svg')]
MODELS = {
    'GPT-4o-mini': 'openai/gpt-4o-mini',
    'Claude-Sonnet-4': 'anthropic/claude-sonnet-4',
    'Gemini-2.0-Flash': 'google/gemini-2.0-flash-001',
    'Llama-3.1-8B': 'meta-llama/Llama-3.1-8B-Instruct',
    'Mistral-7B': 'mistralai/Mistral-7B-Instruct-v0.3',
    'GPT-5.6-terra': 'openai/gpt-5.6-terra',
    'Grok-4.5': 'x-ai/grok-4.5',
    'Gemini-3.6-Flash': 'google/gemini-3.6-flash',
    'Claude-Opus-4.8': 'anthropic/claude-opus-4.8',
}
COLUMNS = ['model', 'cohort', 'n_paired', 'n_A3', 'A0', 'A1', 'A3',
           'P', 'P_lo', 'P_hi', 'p', 'delta_acc', 'correct_wrong', 'wrong_correct']


def rows():
    """Per-model estimates: reversal and accuracy effects on the shared A1/A3 intersection."""
    with CORE.open() as stream:
        core = list(csv.DictReader(stream))
    with PAIRED.open() as stream:
        pairs = {r['model_id']: r for r in csv.DictReader(stream)
                 if r['cohort'] in ('main', 'controls_four') and r['intersection'] == 'A1/A3'}
    result = []
    for row in core:
        item = {k: row[k] if k in ('model', 'cohort') else float(row[k]) for k in COLUMNS}
        pair = pairs[MODELS[item['model']]]
        item['n_paired'], item['n_A3'] = int(pair['n']), int(item['n_A3'])
        for field, source in [('P', 'P_R_pp'), ('P_lo', 'P_R_pp_lo'), ('P_hi', 'P_R_pp_hi'),
                              ('p', 'P_R_exact_p'), ('P_Acc', 'P_Acc_pp'),
                              ('P_Acc_lo', 'P_Acc_pp_lo'), ('P_Acc_hi', 'P_Acc_pp_hi')]:
            item[field] = float(pair[source])
        for condition in ('A1', 'A3'):
            item[condition] = float(pair[f'harmful_{condition}_pp']) + float(pair[f'beneficial_{condition}_pp'])
        result.append(item)
    return result


def export(destination):
    destination.mkdir(parents=True, exist_ok=True)
    for source, name in FIGURES:
        # Matplotlib records the creation time in the SVG metadata; drop it so the asset is stable.
        svg = re.sub(r'\n\s*<dc:date>[^<]*</dc:date>', '', source.read_text(encoding='utf-8'))
        (destination / name).write_text(svg, encoding='utf-8')
    payload = {'source': 'reproducibility/results/analysis/core_nine_models.csv',
               'paired_source': 'reproducibility/reanalysis/outputs/accuracy_decomposition.csv',
               'interval': '95% paired item-bootstrap interval; 20,000 draws',
               'models': rows()}
    (destination / 'results.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/assets')
    export(parser.parse_args().output)
