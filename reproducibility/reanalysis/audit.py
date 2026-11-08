"""Paired accuracy analysis and data checks from saved outputs.

No model calls; the inputs are read-only and verified against frozen hashes.
Run from the artifact root: python reanalysis/audit.py
The main analyses are reproduced separately by reproduce.py.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import re
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / 'analysis_plan.json').read_text())
DATA = ROOT / 'results/data'
REPAIR = ROOT / 'data-repair/outputs'
CONDITIONS = ['A0', 'A1', 'A3', 'A1_agree', 'A3_agree',
              'A1_rat_clean', 'A3_rat_clean', 'A3_rat_gated_clean']
COHORTS = [
    ('main', 'main_audit/neurips_dedup_trials_FULL.csv', 'original', '2026-01 (reported)', 'active length-matching filler', 'same-run (reported)'),
    ('controls_api', 'controls.jsonl', 'controls', '2026-07 (reported)', 'active length-matching filler', 'same-run (reported)'),
    ('controls_open', 'controls_open.jsonl', 'controls', '2026-07 (reported)', 'active length-matching filler', 'same-run (reported)'),
    ('controls_four', 'controls_sota.jsonl', 'controls', '2026-07 (reported)', 'active length-matching filler', 'same-run (reported)'),
    ('controls_t0', 'controls_t0.jsonl', 'controls', '2026-07 (reported)', 'active length-matching filler', 'same-run (reported)'),
    ('clean_api', 'clean_rationale_250_dedup.jsonl', 'controls', '2026-07 (reported)', 'none', 'stored Pass 1; exact generation times unavailable'),
    ('clean_four', 'clean_rationale_sota_20260926.jsonl', 'controls', '2026-09-26', 'none', 'July Pass 1 / September Pass 2'),
    ('temporal', 'temporal_orig_pipeline.jsonl', 'original', '2026-07 (reported)', 'original scaffold (reported)', 'same-run (reported)'),
]


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    return hashlib.sha256(value).hexdigest()


def read_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def flag(value):
    if value is True or value == 'True':
        return True
    if value is False or value == 'False':
        return False
    raise ValueError(f'Invalid boolean flag: {value!r}')


def write_csv(path, records):
    pd.DataFrame(records).to_csv(path, index=False, float_format='%.12g')


def document(item):
    meta = item['metadata']
    key = {'Medicine': 'pubmed_id', 'Science': 'doc_id', 'Law': 'contract_file'}[item['domain']]
    if not meta.get(key):
        raise ValueError(f'Missing document ID: {item["question_id"]}')
    return item['source'] + ':' + str(meta[key])


def canonical_request(row):
    if not row.get('request'):
        return None
    request = json.loads(json.dumps(row['request']))
    count = 0
    if row['condition'] == 'A3_rat_clean':
        for msg in request['messages']:
            msg['content'], changes = re.subn(r' This assistant is specialized in [^.\n]+\.', '', msg['content'])
            count += changes
        if count != 1:
            raise ValueError('Expert source-description sentence not uniquely identified')
    return digest(request)


def rationale_body(text):
    """Match the archived stimulus renderer, which strips the leading answer."""
    s = (text or '').strip()
    for label in ['Yes', 'No']:
        if s[:len(label)].lower() == label.lower():
            rest = s[len(label):].lstrip(' .,:;-—')
            if rest:
                return rest
    return s


def load_cohort(cohort, filename, inputs, rationales):
    path = DATA / filename
    if path.suffix == '.csv':
        with path.open() as stream:
            raw = list(csv.DictReader(stream))
        for row in raw:
            row['status'] = 'retained_in_deduplicated_csv'
    else:
        raw = read_jsonl(path)
    records = []
    seen = set()
    for line, source in enumerate(raw, 1):
        row = dict(source)
        model, qid, condition = row.get('model_id'), row.get('question_id'), row.get('condition')
        if not model or not qid:
            raise ValueError(f'Missing endpoint/item in {filename}:{line}')
        if qid not in inputs:
            raise ValueError(f'Unknown input in {filename}:{line}: {qid}')
        item = inputs[qid]
        key = (model, qid, condition)
        duplicate = key in seen
        seen.add(key)
        valid = row.get('status') in ['ok', 'retained_in_deduplicated_csv']
        valid = valid and all(row.get(k) in ['Yes', 'No'] for k in ['pass1', 'pass2', 'ground_truth'])
        if valid:
            if row['ground_truth'] != item['ground_truth'] or row['domain'] != item['domain']:
                raise ValueError(f'Input/label mismatch: {cohort}/{model}/{qid}')
            for name, expected in [('pass1_correct', row['pass1'] == row['ground_truth']),
                                   ('pass2_correct', row['pass2'] == row['ground_truth']),
                                   ('reversal', row['pass1'] != row['pass2'])]:
                if flag(row[name]) != expected:
                    raise ValueError(f'Corrupt outcome flag: {cohort}/{model}/{qid}/{name}')
        response = row.get('response') or {}
        opponent = row.get('opponent_answer')
        stimulus = rationales.get(qid, {}).get(opponent)
        request = row.get('request')
        texts = '\n'.join(m['content'] for m in request['messages']) if request else None
        # Hashes identify the saved analysis record, NOT a missing generation ID.
        records.append(dict(
            cohort=cohort, model_id=model, question_id=qid, condition=condition,
            serialized_record_sha256=digest(source), source_file=filename, source_line=line,
            input_sha256=item.get('input_sha256') or digest((item['prompt_base'] + '\n' + item['excerpt']).encode()),
            input_version='historical_original' if cohort in ['main', 'temporal'] else 'historical_control_pool',
            domain=item['domain'], document_id=document(item), ground_truth=item['ground_truth'],
            pass1=row.get('pass1'), pass2=row.get('pass2'), opponent_answer=opponent,
            status=row['status'], valid=valid, duplicate=duplicate,
            upstream_duplicate_removed=row.get('duplicate_removed'),
            pass1_record_id=row.get('pass1_record_id'), replicate_id=row.get('replicate_id'),
            served_model=response.get('model'), provider=response.get('provider'),
            started_utc=row.get('started_utc'), finished_utc=row.get('finished_utc'),
            finish_reason=row.get('finish_reason'),
            request_sha256=digest(request) if request else None,
            request_normalized_sha256=canonical_request(row) if condition in ['A1_rat_clean', 'A3_rat_clean'] else None,
            saved_rationale_sha256=digest(stimulus.encode()) if isinstance(stimulus, str) else None,
            delivered_rationale_verified=bool(texts is not None and isinstance(stimulus, str) and rationale_body(stimulus) in texts),
            delivered_input_verified=bool(texts is not None and item['excerpt'] in texts and item['prompt_base'] in texts),
            prior_assistant_message_present=any(m['role']=='assistant' for m in request['messages']) if request else None,
            known_incomplete=item['_known_incomplete'],
        ))
    return records


def pair_rows(records, conditions):
    """Filter complete pairs without outcome-dependent retry selection."""
    items, exclusions = {}, []
    for row in records:
        if row['condition'] in conditions and not row['duplicate']:
            items.setdefault(row['question_id'], {})[row['condition']] = row
    paired = []
    for qid, branches in sorted(items.items()):
        reason = None
        if not all(c in branches for c in conditions):
            reason = 'missing_condition_record'
        elif not all(branches[c]['valid'] for c in conditions):
            reason = 'invalid_branch_output'
        else:
            values = [branches[c] for c in conditions]
            if len({r['pass1'] for r in values}) != 1:
                reason = 'mismatched_initial'
            elif len({r['input_sha256'] for r in values}) != 1:
                reason = 'mismatched_input'
            elif len({r['ground_truth'] for r in values}) != 1:
                reason = 'mismatched_ground_truth'
            for c, row in branches.items():
                if c == 'A0' or row['opponent_answer'] is None:
                    continue
                expected = row['pass1'] if c.endswith('_agree') else ('No' if row['pass1'] == 'Yes' else 'Yes')
                if row['opponent_answer'] != expected:
                    reason = reason or 'mismatched_opponent'
            for field in ['pass1_record_id', 'replicate_id', 'served_model', 'provider']:
                known = {r[field] for r in values if r[field] is not None}
                if len(known) > 1:
                    reason = reason or f'mismatched_{field}'
            if conditions == ['A1_rat_clean', 'A3_rat_clean']:
                requests = [r['request_normalized_sha256'] for r in values]
                if all(requests) and requests[0] != requests[1]:
                    reason = reason or 'unexpected_prompt_difference'
                if all(requests) and not all(r['delivered_input_verified'] and r['delivered_rationale_verified'] for r in values):
                    reason = reason or 'unverified_delivered_input_or_rationale'
        if reason:
            exclusions.append({'question_id': qid, 'exclusion_reason': reason})
        else:
            paired.append(branches)
    return paired, exclusions, items


def moments(pairs, low, high, include_drift=False):
    features = []
    for branches in pairs:
        row = branches[low]
        c = float(row['pass1'] == row['ground_truth'])
        cl = float(row['pass2'] == row['ground_truth'])
        ch = float(branches[high]['pass2'] == row['ground_truth'])
        rl, rh = float(row['pass2'] != row['pass1']), float(branches[high]['pass2'] != row['pass1'])
        values = [c, cl, ch, rl, rh, c*rl, c*rh, (1-c)*rl, (1-c)*rh]
        if include_drift:
            values.append(float(branches['A0']['pass2'] == row['ground_truth']))
        features.append(values)
    return np.asarray(features, dtype=float)


def statistics(m):
    m = np.atleast_2d(m)
    c, cl, ch, rl, rh, hr1, hr3, br1, br3 = m[:, :9].T
    with np.errstate(divide='ignore', invalid='ignore'):
        result = dict(a=c, h_A1=hr1/c, h_A3=hr3/c, b_A1=br1/(1-c), b_A3=br3/(1-c),
                      P_R_pp=100*(rh-rl), P_Acc_pp=100*(ch-cl),
                      delta_acc_A1_pp=100*(cl-c), delta_acc_A3_pp=100*(ch-c),
                      harmful_A1_pp=100*hr1, harmful_A3_pp=100*hr3,
                      beneficial_A1_pp=100*br1, beneficial_A3_pp=100*br3)
    assert np.allclose(ch-c, br3-hr3, atol=1e-12)
    assert np.allclose(ch-cl, (br3-br1)-(hr3-hr1), atol=1e-12)
    if m.shape[1] == 10:
        drift = m[:, 9]
        result.update(reflection_accuracy_pp=100*(drift-c),
                      disagreement_accuracy_pp=100*(cl-drift),
                      source_accuracy_pp=100*(ch-cl))
        assert np.allclose(ch-c, (drift-c)+(cl-drift)+(ch-cl), atol=1e-12)
    return result


def seed_for(*parts):
    return (PLAN['seed'] + int(digest(parts)[:8], 16)) % (2**32)


def bootstrap_means(features, rng, samples, clusters=None, domains=None):
    """Multinomial multiplicities are exactly equivalent to resampling units."""
    if clusters is None:
        patterns, counts = np.unique(features, axis=0, return_counts=True)
        draws = rng.multinomial(len(features), counts/counts.sum(), size=samples)
        return draws @ patterns / len(features)
    # Fixed domain weights; every sampled document carries all its questions.
    result = np.zeros((samples, features.shape[1]))
    clusters, domains = np.asarray(clusters), np.asarray(domains)
    for domain in sorted(set(domains)):
        mask = domains == domain
        ids, idx = np.unique(clusters[mask], return_inverse=True)
        sums = np.zeros((len(ids), features.shape[1]))
        np.add.at(sums, idx, features[mask])
        counts = np.bincount(idx)
        weight = mask.sum()/len(features)
        for start in range(0, samples, 250):
            k = min(250, samples-start)
            draws = rng.multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=k)
            result[start:start+k] += weight*(draws @ sums)/(draws @ counts)[:, None]
    return result


def estimate(pairs, low, high, key, cluster=False, include_drift=False):
    features = moments(pairs, low, high, include_drift)
    original = statistics(features.mean(axis=0))
    documents = [r[low]['document_id'] for r in pairs]
    domains = [r[low]['domain'] for r in pairs]
    seed = seed_for(*key, 'cluster' if cluster else 'item')
    boot = statistics(bootstrap_means(features, np.random.default_rng(seed), PLAN['bootstrap_samples'],
                                     documents if cluster else None, domains if cluster else None))
    result = dict(n=len(pairs), n_initially_correct=int(features[:, 0].sum()),
                  n_initially_wrong=int(len(pairs)-features[:, 0].sum()),
                  n_documents=len(set(documents)), seed=seed,
                  bootstrap_samples=PLAN['bootstrap_samples'],
                  resampling='document_stratified_by_domain' if cluster else 'paired_item')
    for metric, values in original.items():
        result[metric] = values[0]
        valid = np.isfinite(boot[metric])
        lo, hi = np.quantile(boot[metric][valid], [.025, .975]) if valid.any() else (np.nan, np.nan)
        result.update({metric+'_lo': lo, metric+'_hi': hi, metric+'_undefined_draws': int((~valid).sum())})
    for outcome, left, right in [('P_R', 3, 4), ('P_Acc', 1, 2)]:
        delta = features[:, right]-features[:, left]
        b, c = int((delta > 0).sum()), int((delta < 0).sum())
        result.update({outcome+'_positive_discordant': b, outcome+'_negative_discordant': c,
                       outcome+'_exact_p': binomtest(b, b+c, .5).pvalue if b+c else 1.0})
    # a=0/1 and h=0 are explicitly undefined for the ratio criterion.
    a, h, b = result['a'], result['h_A3'], result['b_A3']
    result['selectivity_A3'] = b/h if np.isfinite(b) and h > 0 else np.nan
    result['break_even_selectivity'] = a/(1-a) if 0 < a < 1 else np.nan
    return result


def agreement(pairs, key):
    values = []
    for b in pairs:
        reversal = {c: float(r['pass1'] != r['pass2']) for c, r in b.items()}
        pd_ = reversal['A3']-reversal['A1']
        pa = reversal['A3_agree']-reversal['A1_agree']
        values.append([pd_, pa, pd_-pa, pd_+pa])
        # Independently verify the common source-following outcome.
        f = {c: float(r['pass2'] == r['opponent_answer']) for c, r in b.items()}
        assert np.isclose(pd_+pa, (f['A3']-f['A1'])-(f['A3_agree']-f['A1_agree']))
    values = np.array(values)
    seed = seed_for(*key, 'agreement')
    boot = 100*bootstrap_means(values, np.random.default_rng(seed), PLAN['bootstrap_samples'])
    result = dict(n=len(pairs), seed=seed, bootstrap_samples=PLAN['bootstrap_samples'])
    for i, name in enumerate(['P_disagreement_pp', 'P_agreement_reversal_pp', 'I_R_pp', 'I_F_pp']):
        lo, hi = np.quantile(boot[:, i], [.025, .975])
        result.update({name:100*values[:, i].mean(), name+'_lo':lo, name+'_hi':hi})
    return result


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    original = {r['question_id']:r for r in read_jsonl(REPAIR/'original_benchmark_recovered.jsonl')}
    controls = {r['question_id']:r for r in read_jsonl(DATA/'control_items.jsonl')}
    affected = {}
    for key, filename, items in [('original', 'benchmark_evidence_changes.json', original),
                                  ('controls', 'control_evidence_changes.json', controls)]:
        affected[key] = [r['historical_question_id'] for r in json.loads((REPAIR/filename).read_text())]
        for row in items.values():
            row['_known_incomplete'] = row['question_id'] in affected[key]
    assert len(affected['original']) == 48 and len(affected['controls']) == 16
    rationales = json.loads((DATA/'opponent_rationales.json').read_text())
    source_paths = [DATA/c[1] for c in COHORTS] + [DATA/'control_items.jsonl', DATA/'opponent_rationales.json']
    source_paths += sorted(REPAIR.glob('*.json*'))
    hashes = {str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in source_paths}
    frozen = {'plan':PLAN, 'plan_sha256':digest((HERE/'analysis_plan.json').read_bytes()),
              'inputs':hashes, 'affected_item_ids':affected}
    freeze_path = out/'frozen_inputs.json'
    if freeze_path.exists():
        if json.loads(freeze_path.read_text()) != frozen:
            raise ValueError('Inputs or plan changed; use a NEW output directory.')
    else:
        freeze_path.write_text(json.dumps(frozen, indent=2)+'\n')
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        commit = None
    pdf = ROOT.parent/'neurips_2026_camera_ready.pdf'
    run = {'script_sha256':digest(Path(__file__).read_bytes()), 'workspace_git_commit':commit,
           'workspace_commit_alone_does_not_identify_uncommitted_analysis':True,
           'paper_sha256':digest(pdf.read_bytes()) if pdf.exists() else None,
           'numpy':np.__version__, 'pandas':pd.__version__, 'paid_calls':0}
    (out/'run_manifest.json').write_text(json.dumps(run, indent=2)+'\n')
    inventory, audits, exclusions, results, sensitivity, clusters, agreements, registry, domain_counts, bounds, gates = ([] for _ in range(11))
    for cohort, filename, dataset, date, padding, period in COHORTS:
        rows = load_cohort(cohort, filename, original if dataset == 'original' else controls, rationales)
        inventory.extend(rows)
        for model in sorted({r['model_id'] for r in rows}):
            group = [r for r in rows if r['model_id'] == model]
            relevant = [r for r in group if r['condition'] in CONDITIONS]
            registry.append(dict(cohort=cohort, model_id=model, collection_period=date,
                                 input_version=dataset+'_historical', padding=padding, pass1_pass2_period=period,
                                 observed_item_ids=len({r['question_id'] for r in group}),
                                 raw_records=len(group), observed_relevant_branch_records=len(relevant),
                                 recorded_requests=sum(bool(r['request_sha256']) for r in relevant),
                                 requests_with_prior_assistant_message=sum(r['prior_assistant_message_present'] is True for r in relevant),
                                 permissible_interpretation='within-cohort; historical input coverage limitations; padded contrasts include filler' if padding != 'none' else 'within-cohort source-description contrast; no claim of immunity from nonsignificance'))
            available = {r['condition'] for r in group}
            clean = 'A1_rat_clean' in available
            low, high = ('A1_rat_clean','A3_rat_clean') if clean else ('A1','A3')
            conditions = [low,high]
            paired, excluded, candidate = pair_rows(group, conditions)
            tag = dict(cohort=cohort, model_id=model, contrast=f'{high}-{low}')
            for row in excluded:
                exclusions.append(dict(tag, **row))
            audits.append(dict(tag, candidate_item_ids=len(candidate), paired=len(paired),
                candidate_pair_denominator_is_attempted_total=False,
                upstream_attempts='unknown' if cohort in ['main','clean_api'] else 'see retained statuses; no guarantee every transport attempt was logged',
                duplicate_branch_records=sum(r['duplicate'] for r in group if r['condition'] in conditions),
                invalid_branch_records=sum(not r['valid'] for r in group if r['condition'] in conditions and not r['duplicate']),
                mismatched_initial=sum(r['exclusion_reason']=='mismatched_initial' for r in excluded),
                request_verified_pairs=sum(all(r[c]['request_normalized_sha256'] for c in conditions) for r in paired),
                pass1_record_identity_verified_pairs=sum(all(r[c]['pass1_record_id'] for c in conditions) for r in paired),
                missing_branch_pairs=sum(r['exclusion_reason']=='missing_condition_record' for r in excluded)))
            if not paired:
                continue
            if clean:
                gp, _, _ = pair_rows(group, ['A3_rat_clean', 'A3_rat_gated_clean'])
                for initially_correct in [True, False]:
                    subset_gate = [b for b in gp if (b['A3_rat_clean']['pass1'] == b['A3_rat_clean']['ground_truth']) == initially_correct]
                    if not subset_gate:
                        continue
                    a = np.array([b['A3_rat_clean']['pass1'] != b['A3_rat_clean']['pass2'] for b in subset_gate], dtype=float)
                    b = np.array([b['A3_rat_gated_clean']['pass1'] != b['A3_rat_gated_clean']['pass2'] for b in subset_gate], dtype=float)
                    delta = b-a
                    seed = seed_for(cohort,model,'gate',initially_correct)
                    samples = 100*bootstrap_means(delta[:,None],np.random.default_rng(seed),PLAN['bootstrap_samples'])[:,0]
                    lo, hi = np.quantile(samples,[.025,.975])
                    discordant = int((delta!=0).sum())
                    gates.append(dict(cohort=cohort,model_id=model,stratum='initially_correct' if initially_correct else 'initially_wrong',
                                      n=len(delta),ungated_rate=a.mean(),gated_rate=b.mean(),change_pp=100*delta.mean(),
                                      ci_lo_pp=lo,ci_hi_pp=hi,seed=seed,
                                      exact_p=binomtest(int((delta>0).sum()),discordant,.5).pvalue if discordant else 1,
                                      interpretation='exploratory difference; not a non-inferiority or preservation test'))
            for subset in PLAN['sensitivity_sets']:
                sub = [b for b in paired if subset == 'historical' or
                       (subset == 'exclude_known_incomplete' and not b[low]['known_incomplete']) or
                       (subset == 'medicine_science' and b[low]['domain'] != 'Law')]
                if not sub:
                    continue
                key = [cohort,model,subset,'pair']
                result = dict(tag, subset=subset, intersection='/'.join(conditions), **estimate(sub,low,high,key))
                sensitivity.append(result)
                clusters.append(dict(tag, subset=subset, **estimate(sub,low,high,key,cluster=True)))
                for domain in ['Medicine','Science','Law']:
                    d = [b[low] for b in sub if b[low]['domain']==domain]
                    domain_counts.append(dict(tag,subset=subset,domain=domain,n=len(d),n_documents=len({r['document_id'] for r in d})))
                if subset == 'historical':
                    results.append(result)
            tri_conditions = ['A0','A1','A3']
            if set(tri_conditions).issubset(available):
                tri, tri_excluded, _ = pair_rows(group, tri_conditions)
                if tri:
                    results.append(dict(tag,subset='historical',intersection='/'.join(tri_conditions),
                                        **estimate(tri,'A1','A3',[cohort,model,'triple'],include_drift=True)))
                exclusions.extend(dict(cohort=cohort,model_id=model,contrast='A0/A1/A3',**r) for r in tri_excluded)
            agree_conditions = ['A1','A3','A1_agree','A3_agree']
            if set(agree_conditions).issubset(available):
                ap, ae, _ = pair_rows(group, agree_conditions)
                if ap:
                    agreements.append(dict(cohort=cohort,model_id=model,**agreement(ap,[cohort,model])))
                exclusions.extend(dict(cohort=cohort,model_id=model,contrast='agreement_four_way',**r) for r in ae)
            # Conservatively bound every absent/excluded pair's difference by [-1,1].
            missing = len(candidate)-len(paired)
            features = moments(paired,low,high)
            for name, left, right in [('P_R',3,4),('P_Acc',1,2)]:
                total = (features[:,right]-features[:,left]).sum()
                bounds.append(dict(tag,metric=name,n_observed_candidate_pairs=len(candidate),n_complete=len(paired),
                                   n_unknown=missing,lower_pp=100*(total-missing)/len(candidate),
                                   upper_pp=100*(total+missing)/len(candidate),
                                   interpretation='worst-case range, NOT a confidence interval; excludes entirely unobserved items'))
            print(cohort,model,'paired',len(paired),'excluded',len(excluded),flush=True)
    for filename, rows in [('record_inventory.csv',inventory),('pairing_audit.csv',audits),
                            ('pairing_exclusions.csv',exclusions),('accuracy_decomposition.csv',results),
                            ('evidence_coverage_sensitivity.csv',sensitivity),('document_cluster_sensitivity.csv',clusters),
                            ('agreement_conformity_interaction.csv',agreements),('protocol_registry.csv',registry),
                            ('domain_sample_counts.csv',domain_counts),('missingness_bounds.csv',bounds),
                            ('clean_gate_effects.csv',gates)]:
        write_csv(out/filename,rows)
    assert hashes == {str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in source_paths}
    cards = [r for r in sensitivity if r['subset']=='exclude_known_incomplete' and r['cohort'] in ['main','clean_api','clean_four']]
    write_csv(out/'audit_cards.csv',cards)
    # Compare source-document/hypothesis keys, not incompatible q/rq namespaces.
    def law_key(item):
        return (document(item),item['metadata']['hypothesis_id'])
    original_keys = {law_key(original[q]) for q in affected['original']}
    control_keys = {law_key(controls[q]) for q in affected['controls']}
    (out/'evidence_overlap.json').write_text(json.dumps({
        'affected_original_items':48,'affected_control_items':16,
        'same_document_and_hypothesis':len(original_keys & control_keys),
        'overlapping_documents':len({x[0] for x in original_keys}&{x[0] for x in control_keys}),
        'note':'Overlap does not imply independent samples; versions remain separate.'},indent=2)+'\n')
    # No request for fresh inference is constructed by this script.
    print('Inputs unchanged. Output:', os.path.relpath(out, ROOT), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=HERE/'outputs')
    main(parser.parse_args().out)
