"""Offline regression tests for input recovery and evidence repair."""
import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import repair_inputs as repair


class InputRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = repair.HERE / 'sources'
        cls.outputs = repair.HERE / 'outputs'
        cls.original = repair.read_jsonl(cls.outputs / 'original_benchmark_recovered.jsonl')
        cls.original_by_id = {r['question_id']: r for r in cls.original}
        cls.controls = repair.read_jsonl(cls.sources / 'historical_controls.jsonl.gz')
        cls.norm, cls.docs, _ = repair.load_sources(cls.sources)

    def test_complete_benchmark_and_original_hashes(self):
        self.assertEqual(len(self.original), 1989)
        self.assertEqual(len(self.original_by_id), 1989)
        rows = repair.read_jsonl(self.sources / 'original_log.jsonl.gz')
        for row in rows:
            recovered = self.original_by_id[row['question_id']]
            self.assertEqual(row['excerpt_hash'], repair.h16(recovered['excerpt']))
            self.assertEqual(row['question_hash'], repair.h16(recovered['prompt_base'] + '|' + recovered['excerpt']))
        for row in csv.DictReader(io.StringIO(repair.read_bytes(self.sources / 'original_trials.csv.gz').decode())):
            recovered = self.original_by_id[row['question_id']]
            self.assertEqual((row['domain'], row['ground_truth'], row['confidence_tier']),
                             (recovered['domain'], recovered['ground_truth'], recovered['confidence_tier']))

    def test_surviving_frozen_inputs_and_two_missing_medicine_items(self):
        rows = repair.read_jsonl(self.sources / 'frozen_prefix.jsonl.gz', True)
        self.assertEqual(len(rows), 552)
        for row in rows:
            for key in ('prompt_base', 'excerpt', 'ground_truth'):
                self.assertEqual(row[key], self.original_by_id[row['question_id']][key])
        self.assertEqual(self.original_by_id['q0405']['provenance']['verification'], 'surviving_frozen_input')
        self.assertEqual(self.original_by_id['q0828']['provenance']['verification'],
                         'selection_prompt_and_excerpt_hashes_verified_order')

    def test_science_recovery_does_not_join_on_excerpt_alone(self):
        changes = json.loads((self.outputs / 'previous_recovery_prompt_corrections.json').read_text())
        self.assertEqual(len(changes), 96)
        self.assertTrue(all(r['domain'] == 'Science' and r['same_excerpt'] for r in changes))
        science = [r for r in self.original if r['domain'] == 'Science']
        self.assertEqual(len({(r['prompt_base'], r['excerpt']) for r in science}), 299)

    def check_versioned(self, file, historical, expected_changes):
        rows = repair.read_jsonl(self.outputs / file)
        self.assertEqual(len(rows), len(historical))
        old_by_id = {r['question_id']: r for r in historical}
        changed = 0
        self.assertFalse(set(r['question_id'] for r in rows) & set(old_by_id))
        for row in rows:
            old = old_by_id[row['provenance']['historical_question_id']]
            self.assertEqual(row['ground_truth'], old['ground_truth'])
            self.assertEqual(row['prompt_base'], old['prompt_base'])
            self.assertEqual(row['hash'], repair.h16(row['prompt_base'] + '|' + row['excerpt']))
            if row['excerpt'] != old['excerpt']:
                changed += 1
                self.assertEqual(row['domain'], 'Law')
                self.assertFalse(row['provenance']['historical_rationales_reusable'])
            if row['domain'] == 'Law':
                _, doc, labels = self.docs[row['metadata']['contract_file']]
                hid = row['metadata']['hypothesis_id']
                ann = doc['annotation_sets'][0]['annotations'][hid]
                for index in ann['spans']:
                    a, b = doc['spans'][index]
                    text = doc['text'][a:b].strip()
                    if text:
                        self.assertIn(text, row['excerpt'])
                self.assertIn(labels[hid]['hypothesis'], row['prompt_base'])
        self.assertEqual(changed, expected_changes)

    def test_all_800_law_inputs_evidence_complete(self):
        self.check_versioned('benchmark_evidence_complete_v2.jsonl', self.original, 48)

    def test_control_fix_changes_exactly_16_excerpts(self):
        self.check_versioned('controls_evidence_complete_v2.jsonl', self.controls, 16)

    def test_complete_regeneration_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            repair.build(self.sources, Path(directory))
            for path in self.outputs.iterdir():
                if path.is_file():
                    self.assertEqual(path.read_bytes(), (Path(directory) / path.name).read_bytes())

    def test_no_overwrite_of_different_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'output'
            repair.immutable_write(path, b'original')
            with self.assertRaises(ValueError):
                repair.immutable_write(path, b'changed')
            self.assertEqual(path.read_bytes(), b'original')

    def test_jsonl_unicode_and_truncated_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.jsonl'
            path.write_bytes(repair.jsonl_bytes([{'excerpt': 'x\u2028y\u0085z'}]))
            self.assertEqual(repair.read_jsonl(path), [{'excerpt': 'x\u2028y\u0085z'}])
            path.write_bytes(path.read_bytes() + b'{"broken":')
            self.assertEqual(len(repair.read_jsonl(path, True)), 1)
            with self.assertRaises(json.JSONDecodeError):
                repair.read_jsonl(path)


if __name__ == '__main__':
    unittest.main()
