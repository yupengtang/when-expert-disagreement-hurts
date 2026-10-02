"""Export project-page estimates from the revised compatible-pair analysis."""
import argparse
import csv
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
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


def export(destination):
    with (HERE.parent / 'rebuttal/analysis/core_nine_models.csv').open() as stream:
        old = list(csv.DictReader(stream))
    with (HERE / 'outputs/accuracy_decomposition.csv').open() as stream:
        pairs = {r['model_id']: r for r in csv.DictReader(stream)
                 if r['cohort'] in ('main', 'controls_four') and r['intersection'] == 'A1/A3'}
    columns = ['model', 'cohort', 'n_paired', 'n_A3', 'A0', 'A1', 'A3',
               'P', 'P_lo', 'P_hi', 'p', 'delta_acc', 'correct_wrong', 'wrong_correct']
    rows = []
    for row in old:
        item = {k: row[k] if k in ('model', 'cohort') else float(row[k]) for k in columns}
        pair = pairs[MODELS[item['model']]]
        item['n_paired'], item['n_A3'] = int(pair['n']), int(item['n_A3'])
        for field, source in [('P', 'P_R_pp'), ('P_lo', 'P_R_pp_lo'), ('P_hi', 'P_R_pp_hi'),
                              ('p', 'P_R_exact_p'), ('P_Acc', 'P_Acc_pp'),
                              ('P_Acc_lo', 'P_Acc_pp_lo'), ('P_Acc_hi', 'P_Acc_pp_hi')]:
            item[field] = float(pair[source])
        for condition in ('A1', 'A3'):
            item[condition] = float(pair[f'harmful_{condition}_pp']) + float(pair[f'beneficial_{condition}_pp'])
        rows.append(item)
    destination.mkdir(parents=True, exist_ok=True)
    with (HERE / 'outputs/web_models.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {'source': 'reproducibility/reanalysis/outputs/web_models.csv',
               'paired_source': 'reproducibility/reanalysis/outputs/accuracy_decomposition.csv',
               'interval': '95% paired item-bootstrap interval; 20,000 draws',
               'models': rows}
    (destination / 'results.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    shutil.copy2(HERE / 'outputs/source_accuracy_effects.svg', destination / 'reversal.svg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--web-output', type=Path, required=True)
    export(parser.parse_args().web_output)
