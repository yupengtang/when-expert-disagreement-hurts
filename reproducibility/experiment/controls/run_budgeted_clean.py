#!/usr/bin/env python3
"""Run the no-padding rationale protocol on the four newer models under a USD budget.

No implicit retries, replacement models, or outcome-dependent exclusions. The
stored July Pass-1 answers (results/data/controls_sota.jsonl) are shared across
the three contemporaneous branches. Unsupported decoding parameters are omitted
and recorded. Requests go directly to the OpenRouter HTTP API so that the
charged cost of every call can be logged.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import random
import threading
from datetime import datetime, timezone

import requests
import run_clean_rationale_controls as clean

MODELS = ['openai/gpt-5.6-terra', 'x-ai/grok-4.5',
          'google/gemini-3.6-flash', 'anthropic/claude-opus-4.8']


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--budget', type=float, default=50)
    p.add_argument('--pilot-items', type=int, default=0)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    out = Path(args.out)
    source = clean.ITEMS.parent / 'controls_sota.jsonl'
    manifest_path = out.with_suffix('.manifest.json')
    if out.exists() and not args.resume:
        p.error('Existing output requires --resume')
    clean.load_env(clean.ROOT / '.env')
    key = os.environ.get('OPENROUTER_API_KEY')
    if not key:
        p.error('OPENROUTER_API_KEY is not configured')
    response = requests.get('https://openrouter.ai/api/v1/models', timeout=40)
    response.raise_for_status()
    catalog = {r['id']: r for r in response.json()['data'] if r['id'] in MODELS}
    assert set(catalog) == set(MODELS), 'A requested endpoint is unavailable'
    fingerprint = {str(x.relative_to(clean.ROOT.parent)): digest(x)
                   for x in [source, clean.ITEMS, clean.RATIONALES]}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest['input_sha256'] == fingerprint, 'Inputs changed'
        assert manifest['budget_usd'] == args.budget, 'Budget changed'
    else:
        manifest = {'created_utc': stamp(), 'budget_usd': args.budget,
                    'input_sha256': fingerprint, 'catalog': catalog,
                    'conditions': list(clean.CONDITIONS), 'requested_temperature': .7,
                    'requested_top_p': 1., 'max_tokens': 600, 'automatic_retries': 0,
                    'selection': 'First 250 items; existing valid Pass-1; direction-compliant rationale.',
                    'historical_pass1': True, 'padding_policy': 'none'}
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    records = [json.loads(s) for s in out.read_text().splitlines()] if out.exists() else []
    if any(r.get('charged_cost_usd') is None for r in records):
        p.error('Unresolved cost in existing records; reconcile before resuming')
    spent = sum(r['charged_cost_usd'] for r in records)
    reserved = 0.
    stop = threading.Event()
    lock = threading.Lock()
    done = {(r['model_id'], r['question_id'], r['condition']) for r in records}
    items = [json.loads(s) for s in clean.ITEMS.read_text().splitlines() if s.strip()][:250]
    rationales = json.loads(clean.RATIONALES.read_text())
    pass1 = clean.load_shared_pass1([source], set(MODELS))
    jobs = []
    for model in MODELS:
        eligible = 0
        for item in items:
            qid = item['question_id']
            answer = pass1.get((model, qid))
            if answer not in ('Yes', 'No'):
                continue
            opponent = 'No' if answer == 'Yes' else 'Yes'
            record = rationales.get(qid, {})
            if not record.get(opponent) or (record.get('compliant') or {}).get(opponent) is False:
                continue
            eligible += 1
            if args.pilot_items and eligible > args.pilot_items:
                break
            for condition in clean.CONDITIONS:
                if (model, qid, condition) not in done:
                    jobs.append((model, item, answer, record, condition))
    random.Random(20260926).shuffle(jobs)
    print(json.dumps({'pending_calls': len(jobs), 'previous_cost_usd': spent,
                      'budget_usd': args.budget}), flush=True)

    def run(job):
        nonlocal spent, reserved
        model, item, pass1, rationale, condition = job
        opponent = 'No' if pass1 == 'Yes' else 'Yes'
        prompt = clean.build_prompt(item, pass1, rationale[opponent], condition)
        supported = catalog[model]['supported_parameters']
        params = {'temperature': .7, 'top_p': 1., 'frequency_penalty': 0., 'presence_penalty': 0.}
        payload = {'model': model, 'messages': [
            {'role': 'system', 'content': clean.PASS1_SYSTEM},
            {'role': 'user', 'content': prompt}], 'max_tokens': 600,
            **{k: v for k, v in params.items() if k in supported}}
        pricing = catalog[model]['pricing']
        # Reserve byte-count input plus 4096 output tokens, despite requested 600.
        # This leaves a margin for reasoning/provider accounting and in-flight work.
        reserve = ((len(prompt.encode()) + len(clean.PASS1_SYSTEM.encode()) + 256)
                   * float(pricing['prompt']) + 4096 * float(pricing['completion']))
        with lock:
            if stop.is_set() or spent + reserved + reserve > args.budget:
                stop.set()
                return
            reserved += reserve
        row = {'model_id': model, 'question_id': item['question_id'],
               'domain': item['domain'], 'source': item.get('source'), 'condition': condition,
               'pass1': pass1, 'ground_truth': item['ground_truth'],
               'pass1_correct': pass1 == item['ground_truth'],
               'opponent_answer': opponent, 'opponent_agrees': False,
               'gated': clean.CONDITIONS[condition]['gate'], 'has_rationale': True,
               'rationale_target_relation': 'benchmark_matching' if opponent == item['ground_truth'] else 'benchmark_conflicting',
               'rationale_compliant': (rationale.get('compliant') or {}).get(opponent),
               'rationale_factually_validated': False, 'padding_policy': 'none',
               'historical_pass1': True, 'started_utc': stamp(),
               'temperature': payload.get('temperature'), 'requested_temperature': .7,
               'omitted_unsupported_parameters': sorted(set(params) - set(supported)),
               'request': payload, 'reservation_usd': reserve,
               'status': 'request_error', 'charged_cost_usd': None}
        try:
            r = requests.post('https://openrouter.ai/api/v1/chat/completions',
                              headers={'Authorization': f'Bearer {key}'}, json=payload, timeout=180)
            row['http_status'] = r.status_code
            data = r.json()
            row['response'] = data
            usage = data.get('usage') or {}
            row['usage'] = usage
            cost = usage.get('cost')
            if isinstance(cost, (int, float)) and cost >= 0:
                row['charged_cost_usd'] = cost
            if r.ok and data.get('choices'):
                choice = data['choices'][0]
                raw = choice['message'].get('content') or ''
                answer = clean.parse_answer(raw)
                row.update(raw_pass2=raw, pass2=answer,
                           status='ok' if answer else 'pass2_invalid',
                           pass2_correct=answer == item['ground_truth'] if answer else None,
                           reversal=answer != pass1 if answer else False,
                           finish_reason=choice.get('finish_reason'))
        except Exception as exc:
            row['error_type'] = type(exc).__name__  # Never log credentials/headers.
        row['finished_utc'] = stamp()
        with lock:
            with out.open('a') as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
            reserved -= reserve
            if row['charged_cost_usd'] is None:
                spent += reserve
                stop.set()
            else:
                spent += row['charged_cost_usd']
                if row['charged_cost_usd'] > reserve or spent >= args.budget:
                    stop.set()
            print(json.dumps({'model': model, 'condition': condition, 'status': row['status'],
                              'cost_usd': row['charged_cost_usd'], 'total_usd': round(spent, 6)}), flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(run, jobs))
    print(json.dumps({'finished_utc': stamp(), 'cost_usd': spent,
                      'stopped_for_budget_or_accounting': stop.is_set()}), flush=True)


if __name__ == '__main__':
    main()
