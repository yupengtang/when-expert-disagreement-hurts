"""Validate the completed September run and report cost and output retention."""
import collections
import json
from pathlib import Path


def build(root, out, tables, names, table):
    path = root / 'results/data/clean_rationale_sota_20260926.jsonl'
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    expected = {'openai/gpt-5.6-terra': 249, 'x-ai/grok-4.5': 249,
                'google/gemini-3.6-flash': 233, 'anthropic/claude-opus-4.8': 248}
    conditions = ['A1_rat_clean', 'A3_rat_clean', 'A3_rat_gated_clean']
    assert len(rows) == 2937
    assert len({(r['model_id'], r['question_id'], r['condition']) for r in rows}) == len(rows)
    assert all(r['status'] in ['ok', 'pass2_invalid'] for r in rows)
    assert all(r['charged_cost_usd'] is not None for r in rows)
    cost = sum(r['charged_cost_usd'] for r in rows)
    assert cost <= 50
    output, models = [], {}
    for model, n in expected.items():
        g = [r for r in rows if r['model_id'] == model]
        assert len(g) == 3*n
        assert all(r['response']['model'] == model for r in g)
        assert all(r['request']['max_tokens'] == 600 and r['padding_policy'] == 'none' for r in g)
        for qid in {r['question_id'] for r in g}:
            branches = {r['condition']: r for r in g if r['question_id'] == qid}
            assert set(branches) == set(conditions)
            assert len({r['pass1'] for r in branches.values()}) == 1
            prompts = {c: r['request']['messages'][1]['content'] for c, r in branches.items()}
            expertise = f" This assistant is specialized in {branches[conditions[0]]['domain'].lower()}."
            assert prompts[conditions[1]].replace(expertise, '', 1) == prompts[conditions[0]]
            gate = ('Before revising, check whether the provided evidence excerpt itself supports '
                    'changing the answer. Revise only if the excerpt supports the change. If the '
                    'disagreement provides no new evidence, rely on the excerpt rather than the '
                    'source label.')
            assert prompts[conditions[2]].replace(gate+'\n\n', '', 1) == prompts[conditions[1]]
        counts = collections.Counter(r['condition'] for r in g if r['status']=='ok')
        for condition in conditions:
            assert sum(r['condition']==condition for r in g) == n
        for r in g:
            if r['status'] == 'ok':
                assert r['pass2'] in ['Yes', 'No']
                assert r['reversal'] == (r['pass1'] != r['pass2'])
                assert r['pass2_correct'] == (r['pass2'] == r['ground_truth'])
        models[model] = {'calls':len(g), 'eligible_items':n, 'valid_by_condition':dict(counts),
                         'invalid':sum(r['status']!='ok' for r in g),
                         'finish_reason_counts':dict(collections.Counter(r['finish_reason'] for r in g)),
                         'cost_usd':sum(r['charged_cost_usd'] for r in g),
                         'prompt_tokens':sum(r['usage']['prompt_tokens'] for r in g),
                         'completion_tokens':sum(r['usage']['completion_tokens'] for r in g),
                         'omitted_unsupported_parameters':g[0]['omitted_unsupported_parameters']}
        output.append([names[model], n] + [counts[c] for c in conditions] +
                      [models[model]['invalid'], models[model]['finish_reason_counts'].get('length',0)])
    report = {'calls':len(rows), 'valid_outputs':sum(r['status']=='ok' for r in rows),
              'cost_usd':cost, 'started_utc':min(r['started_utc'] for r in rows),
              'finished_utc':max(r['finished_utc'] for r in rows), 'models':models}
    (out / 'new_clean_run_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    table('clean_retention.tex', r'No-padding run on September 26, 2026. $N$ is the eligible stored-Pass-1 item count; the next three columns are parseable outputs. Invalid and length-stop counts cover all three conditions. A length stop does not imply an invalid leading answer; parseable leading Yes/No answers are retained. No automatic retries were made.',
          'tab:clean_retention', 'lrrrrrr', ['Model', '$N$', '$A_1^{\\rm rat}$', '$A_3^{\\rm rat}$', 'Gated', 'Invalid', 'Length'], output)
    (tables / 'new_clean_compute.tex').write_text(
        f"The September no-padding run adds {len(rows):,} API calls, retaining {report['valid_outputs']:,} parseable condition outputs. Recorded API usage charges total \\${cost:.2f}; the artifact preserves per-call usage, requested parameters, provider responses, and timestamps.")
