#!/usr/bin/env python3
"""Reconstruct and verify the experiment inputs, and export separate evidence-complete inputs.

No API calls. Existing output files are never overwritten. The captured sources
in sources/ are checked against their checksums, and the reconstruction uses only
Python's standard library, entirely offline.
"""
import argparse
import collections
import copy
import csv
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def h16(text):
    return sha(text.encode('utf-8'))[:16]


def json_bytes(obj):
    return (json.dumps(obj, ensure_ascii=False, indent=2) + '\n').encode()


def jsonl_bytes(rows):
    return ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode()


def immutable_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'Refusing to overwrite a different artifact: {path}')
    else:
        path.write_bytes(data)


def read_bytes(path):
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == '.gz' else data


def read_jsonl(path, allow_truncated_tail=False):
    # JSONL separates records with LF, not Unicode paragraph separators that
    # may occur inside an evidence string.
    lines = read_bytes(path).decode().split('\n')
    if lines and lines[-1] == '':
        lines.pop()
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if not (allow_truncated_tail and index == len(lines) - 1):
                raise
    return rows


def load_sources(sources):
    manifest = json.loads((sources / 'manifest.json').read_text())
    for name, info in manifest.items():
        if sha((sources / name).read_bytes()) != info['stored_sha256']:
            raise ValueError(f'Source checksum mismatch: {name}')
    spec = importlib.util.spec_from_file_location('frozen_normalizers', sources / 'normalizers.py')
    normalizers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(normalizers)
    docs = {}
    contract = []
    for split in ('train', 'dev', 'test'):
        blob = json.loads(read_bytes(sources / f'contract_{split}.json.gz'))
        for doc in blob['documents']:
            if doc['file_name'] in docs:
                raise ValueError('Non-unique ContractNLI file identity')
            docs[doc['file_name']] = (split, doc, blob['labels'])
            contract.append((split, doc, blob['labels']))
    return normalizers, docs, contract


def evidence(doc, hid):
    ann = doc['annotation_sets'][0]['annotations'][hid]
    indices = ann['spans']
    if not indices:
        raise ValueError('No annotated evidence')
    offsets = [doc['spans'][i] for i in indices]
    for a, b in offsets:
        if not 0 <= a < b <= len(doc['text']):
            raise ValueError('Invalid evidence offsets')
    nonempty = [(a, b, doc['text'][a:b].strip()) for a, b in offsets if doc['text'][a:b].strip()]
    if not nonempty:
        raise ValueError('All evidence spans are empty')
    return ann, [[a, b] for a, b, t in nonempty], [t for a, b, t in nonempty]


def law_metadata(split, doc, hid, labels):
    ann, offsets, texts = evidence(doc, hid)
    return {
        'contract_file': doc['file_name'], 'hypothesis_id': hid,
        'source_split': split, 'source_document_sha256': sha(doc['text'].encode()),
        'original_hypothesis': labels[hid]['hypothesis'], 'original_label': ann['choice'],
        'evidence_offsets': offsets, 'evidence_span_texts': texts,
        'evidence_span_count': len(texts), 'b2_evidence': ' '.join(texts),
        'ignored_whitespace_only_spans': len(ann['spans']) - len(texts),
    }


def candidates(sources, norm, contract):
    for raw in read_jsonl(sources / 'pubmedqa.jsonl.gz'):
        rec = norm.normalize_pubmedqa_to_binary(raw)
        if rec:
            rec['_recovery_source'] = 'PubMedQA pqa_labeled/train'
            yield rec
    corpus = {}
    for raw in read_jsonl(sources / 'scifact_corpus.jsonl.gz'):
        sentences = [s.strip() for s in raw['abstract'] if s.strip()]
        corpus[str(raw['doc_id'])] = {'abstract': ' '.join(sentences), 'sentences': sentences}
    for split in ('train', 'dev'):
        for raw in read_jsonl(sources / f'scifact_claims_{split}.jsonl.gz'):
            for entries in raw.get('evidence', {}).values():
                for entry in entries:
                    if entry['label'] == 'CONTRADICT':
                        entry['label'] = 'REFUTES'
            rec = norm.normalize_scifact_to_binary(raw, corpus)
            if rec:
                rec['metadata']['claim_id'] = raw['id']
                rec['metadata']['source_split'] = split
                rec['_recovery_source'] = f'SciFact {split}'
                yield rec
    for split, doc, labels in contract:
        for hid, ann in doc['annotation_sets'][0]['annotations'].items():
            if ann['choice'] not in ('Entailment', 'Contradiction') or not ann['spans']:
                continue
            # Historical loader stripped the document before applying its
            # original offsets. Retain that behavior ONLY for exact recovery.
            text = doc['text'].strip()
            chunks = [text[max(0, doc['spans'][i][0] - 200):doc['spans'][i][1] + 200].strip()
                      for i in ann['spans']]
            excerpt = '\n\n...\n\n'.join(chunks)[:2000].strip()
            rec = norm.normalize_contractnli_to_binary({
                'premise': excerpt, 'hypothesis': labels[hid]['hypothesis'], 'label': ann['choice']})
            if rec:
                rec['metadata'] = law_metadata(split, doc, hid, labels)
                rec['_recovery_source'] = f'ContractNLI {split}; historical character-window construction'
                yield rec


def recover(sources, norm, contract):
    targets = {}
    for row in csv.DictReader(io.StringIO(read_bytes(sources / 'original_trials.csv.gz').decode())):
        value = (row['domain'], row['ground_truth'], row['confidence_tier'])
        qid = row['question_id']
        if qid in targets and targets[qid] != value:
            raise ValueError(f'Conflicting trial labels: {qid}')
        targets[qid] = value
    logged = {}
    for row in read_jsonl(sources / 'original_log.jsonl.gz'):
        qid = row['question_id']
        value = (row['excerpt_hash'], row['question_hash'])
        if qid in logged and logged[qid] != value:
            raise ValueError(f'Conflicting original hashes: {qid}')
        logged[qid] = value
    joint = collections.defaultdict(list)
    screening = collections.defaultdict(list)
    for rec in candidates(sources, norm, contract):
        excerpt_hash = h16(rec['excerpt'])
        joint[(excerpt_hash, h16(rec['prompt_base_experiment'] + '|' + rec['excerpt']))].append(rec)
        screening[(excerpt_hash, h16(rec['prompt_base']))].append(rec)
    frozen = {r['question_id']: r for r in read_jsonl(sources / 'frozen_prefix.jsonl.gz', True)}
    selection = [r for r in read_jsonl(sources / 'selection_log.jsonl.gz', True)
                 if r.get('record_type') == 'candidate']
    # Confirm order against ALL logged rows in the surviving selection prefix,
    # not just the one missing item that needs this fallback.
    for index, row in enumerate(selection, 1):
        qid = f'q{index:04d}'
        matches = screening[(row['excerpt_hash'], row['prompt_hash'])]
        valid = [r for r in matches if r['domain'] == row['domain'] and r['ground_truth'] == row['ground_truth']]
        if not valid:
            raise ValueError(f'Cannot verify selection record {qid}')
        if qid in logged and not any(
                (h16(r['excerpt']), h16(r['prompt_base_experiment'] + '|' + r['excerpt'])) == logged[qid]
                for r in valid):
            raise ValueError(f'Selection order disagrees with original log at {qid}')
    recovered = []
    for qid, (domain, label, tier) in sorted(targets.items()):
        if qid in logged:
            options = joint[logged[qid]]
            basis = 'original_generation_excerpt_and_question_hashes'
        else:
            entry = selection[int(qid[1:]) - 1]
            options = screening[(entry['excerpt_hash'], entry['prompt_hash'])]
            basis = 'surviving_frozen_input' if qid in frozen else 'selection_prompt_and_excerpt_hashes_verified_order'
        options = [r for r in options if r['domain'] == domain and r['ground_truth'] == label]
        identities = {(r['prompt_base_experiment'], r['excerpt'], r['ground_truth']) for r in options}
        if len(identities) != 1:
            raise ValueError(f'Unresolved or ambiguous input: {qid} ({len(identities)} candidates)')
        chosen = options[0]
        rec = {k: copy.deepcopy(chosen[k]) for k in ('domain', 'source', 'excerpt', 'ground_truth', 'metadata')}
        rec.update(question_id=qid, prompt_base=chosen['prompt_base_experiment'], confidence_tier=tier)
        if qid in frozen:
            old = frozen[qid]
            for field in ('prompt_base', 'excerpt', 'domain', 'ground_truth'):
                if old[field] != rec[field]:
                    raise ValueError(f'Surviving frozen input disagrees: {qid}/{field}')
        rec['excerpt_hash'] = h16(rec['excerpt'])
        rec['hash'] = h16(rec['prompt_base'] + '|' + rec['excerpt'])
        rec['input_sha256'] = sha((rec['prompt_base'] + '|' + rec['excerpt']).encode())
        rec['provenance'] = {'verification': basis, 'source': chosen['_recovery_source'],
                             'metadata_reconstructed': True, 'historical_input': True}
        recovered.append(rec)
    if len(recovered) != 1989:
        raise ValueError('Unexpected historical benchmark size')
    return recovered, len(logged), len(frozen), len(selection)


def repair_law(rec, docs):
    """Keep intact excerpts; otherwise use a contiguous, untruncated window."""
    split, doc, labels = docs[rec['metadata']['contract_file']]
    hid = rec['metadata']['hypothesis_id']
    meta = law_metadata(split, doc, hid, labels)
    expected = 'Yes' if meta['original_label'] == 'Entailment' else 'No'
    if rec['ground_truth'] != expected or meta['original_hypothesis'] not in rec['prompt_base']:
        raise ValueError('Law question or label disagrees with source')
    before = [i for i, t in enumerate(meta['evidence_span_texts']) if t not in rec['excerpt']]
    out = copy.deepcopy(rec)
    if before:
        start = max(0, min(a for a, b in meta['evidence_offsets']) - 200)
        end = min(len(doc['text']), max(b for a, b in meta['evidence_offsets']) + 200)
        out['excerpt'] = doc['text'][start:end].strip()
        meta['repaired_document_window'] = [start, end]
    if not all(t in out['excerpt'] for t in meta['evidence_span_texts']):
        raise ValueError('Repair dropped an annotated evidence span')
    out['metadata'] = meta
    return out, before


def versioned(rows, docs, namespace):
    result, changes = [], []
    for old in rows:
        out, missing = repair_law(old, docs) if old['domain'] == 'Law' else (copy.deepcopy(old), [])
        changed = out['excerpt'] != old['excerpt']
        old_id = old['question_id']
        out['question_id'] = f'{namespace}:{old_id}'
        out['hash'] = h16(out['prompt_base'] + '|' + out['excerpt'])
        out['excerpt_hash'] = h16(out['excerpt'])
        out['input_sha256'] = sha((out['prompt_base'] + '|' + out['excerpt']).encode())
        out['dataset_version'] = namespace
        out['provenance'] = {
            'historical_question_id': old_id, 'historical_input_sha256':
            sha((old['prompt_base'] + '|' + old['excerpt']).encode()),
            'evidence_changed': changed, 'requires_fresh_pass1_and_pass2': True,
            'historical_rationales_reusable': not changed,
            'not_a_replacement_for_historical_results': True,
        }
        if 'candidate_id' in out:
            out['provenance']['historical_candidate_id'] = out['candidate_id']
            out['candidate_id'] = 'v2_' + out['input_sha256'][:16]
        if changed and 'confidence_tier' in out:
            out['provenance']['historical_confidence_tier'] = out['confidence_tier']
            out['confidence_tier'] = 'Unscreened'
        for key in ('yes_proportion_llama', 'yes_proportion_mistral'):
            if changed and key in out:
                out['provenance'][f'historical_{key}'] = out.pop(key)
        if changed:
            changes.append({'historical_question_id': old_id, 'question_id': out['question_id'],
                            'missing_annotated_span_indices_before': missing,
                            'before_excerpt_sha256': sha(old['excerpt'].encode()),
                            'after_excerpt_sha256': sha(out['excerpt'].encode()),
                            'before_chars': len(old['excerpt']), 'after_chars': len(out['excerpt']),
                            'all_annotated_spans_present_after': True,
                            'historical_rationales_must_be_regenerated': True})
        result.append(out)
    return result, changes


def build(sources, output):
    norm, docs, contract = load_sources(sources)
    recovered, logged_n, frozen_n, selection_n = recover(sources, norm, contract)
    original_fixed, original_changes = versioned(recovered, docs, 'benchmark-evidence-complete-v2')
    controls = read_jsonl(sources / 'historical_controls.jsonl.gz')
    fixed_controls, control_changes = versioned(controls, docs, 'controls-evidence-complete-v2')
    previous = {r['question_id']: r for r in read_jsonl(sources / 'prior_recovered.jsonl.gz')}
    corrections = []
    for row in recovered:
        old = previous.get(row['question_id'])
        if old and old['prompt_base'] != row['prompt_base']:
            corrections.append({'question_id': row['question_id'], 'domain': row['domain'],
                                'same_excerpt': old['excerpt'] == row['excerpt'],
                                'previous_prompt': old['prompt_base'], 'verified_prompt': row['prompt_base']})
    artifacts = {
        'original_benchmark_recovered.jsonl': jsonl_bytes(recovered),
        'benchmark_evidence_complete_v2.jsonl': jsonl_bytes(original_fixed),
        'controls_evidence_complete_v2.jsonl': jsonl_bytes(fixed_controls),
        'benchmark_evidence_changes.json': json_bytes(original_changes),
        'control_evidence_changes.json': json_bytes(control_changes),
        'previous_recovery_prompt_corrections.json': json_bytes(corrections),
    }
    report = {
        'historical_items_recovered': len(recovered),
        'domains': dict(collections.Counter(r['domain'] for r in recovered)),
        'verification_levels': dict(collections.Counter(r['provenance']['verification'] for r in recovered)),
        'original_generation_hash_items': logged_n, 'surviving_frozen_items_crosschecked': frozen_n,
        'selection_order_records_crosschecked': selection_n,
        'missing_original_items': [], 'previous_recovery_wrong_prompt_count': len(corrections),
        'historical_benchmark_law_excerpts_requiring_repair': len(original_changes),
        'historical_control_law_excerpts_requiring_repair': len(control_changes),
        'repaired_control_items': len(fixed_controls),
        'all_law_annotated_spans_present_in_v2': True,
        'historical_results_changed': False, 'paid_inference_calls': 0,
        'inputs_manifest_sha256': sha((sources / 'manifest.json').read_bytes()),
        'outputs': {name: {'sha256': sha(data), 'bytes': len(data)} for name, data in artifacts.items()},
    }
    for name, data in artifacts.items():
        immutable_write(output / name, data)
    immutable_write(output / 'audit_report.json', json_bytes(report))
    print(json.dumps({k: v for k, v in report.items() if k != 'outputs'}, indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--sources', type=Path, default=HERE / 'sources')
    parser.add_argument('--out', type=Path, default=HERE / 'outputs')
    args = parser.parse_args()
    build(args.sources, args.out)


if __name__ == '__main__':
    main()
