"""Offline end-to-end checks of the experiment code with a stand-in API client.

No network access, API key or GPU is needed. Each runner is executed on a few
items with a deterministic stand-in for client.Client, and its records are read
back with the analysis code where the paper's analysis consumes them. Two
tests compare against released files: the control pool is rebuilt
byte-for-byte, and the Claude-Sonnet-4 rows of the main trial table are rebuilt
from the bundled generation log. Runners
that length-match prompts with the gpt2 tokenizer are skipped when that
tokenizer cannot be loaded. The local-model path of run_experiment.py requires
a CUDA device and the model weights and is not exercised here.

Run from the artifact root:
    python -m unittest discover -s experiment/tests -v
"""
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1]
ARTIFACT = EXPERIMENT.parent
INPUTS = ARTIFACT / "data-repair" / "outputs" / "original_benchmark_recovered.jsonl"
CONTROL_POOL = ARTIFACT / "results" / "data" / "control_items.jsonl"
MAIN_TABLE = ARTIFACT / "results" / "data" / "main_audit" / "neurips_dedup_trials_FULL.csv"


class StubClient:
    """Deterministic stand-in for client.Client: answers Yes or No from a hash of the request."""

    def __init__(self, *args, **kwargs):
        pass

    def chat(self, model, messages, **kwargs):
        digest = hashlib.sha256(json.dumps([model, messages], sort_keys=True).encode()).digest()
        answer = "Yes" if digest[0] % 2 else "No"
        text = (f"{answer}\nThe excerpt is read as support for this answer. "
                "This sentence makes the explanation long enough to parse.")
        return {"content": text, "finish_reason": "stop", "model": model,
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


class StubResponse:
    def __init__(self, payload):
        self.status_code = 200
        self.ok = True
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


class StubRequests:
    """Stand-in for the requests calls of run_budgeted_clean.py."""

    def __init__(self, models):
        self.models = models

    def get(self, url, timeout=None):
        return StubResponse({"data": [{"id": m, "supported_parameters": ["temperature", "top_p"],
                                       "pricing": {"prompt": "0.000001", "completion": "0.000002"}}
                                      for m in self.models]})

    def post(self, url, headers=None, json=None, timeout=None):
        content = StubClient().chat(json["model"], json["messages"])["content"]
        return StubResponse({"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                             "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.000001}})


stub_client = types.ModuleType("client")
stub_client.Client = StubClient
sys.modules["client"] = stub_client
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(EXPERIMENT / "controls"))
os.environ.setdefault("OPENROUTER_API_KEY", "offline-test")
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(module, *argv):
    saved = sys.argv
    sys.argv = [module.__name__] + [str(a) for a in argv]
    try:
        module.main()
    finally:
        sys.argv = saved


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def gpt2_available():
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained("gpt2")
        return True
    except Exception:
        return False


GPT2 = gpt2_available()


class OfflinePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="offline-pipeline-"))
        cls.controls = cls.tmp / "controls.jsonl"
        run(load("controls/run_controls.py", "run_controls"),
            "--models", "openai/gpt-4o-mini", "--limit", 3, "--max-workers", 2, "--out", cls.controls)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_released_inputs_load_and_verify(self):
        from src.data_structures import load_questions
        self.assertEqual(len(load_questions(INPUTS)), 1989)  # raises on any checksum mismatch
        self.assertEqual(len(read_jsonl(CONTROL_POOL)), 500)

    def test_control_suite_feeds_the_analysis(self):
        rows = read_jsonl(self.controls)
        self.assertEqual(len(rows), 3 * 10)
        self.assertTrue(all(r["status"] == "ok" for r in rows))
        sys.path.insert(0, str(ARTIFACT / "results"))
        import analyze_controls
        frame = analyze_controls.load(self.controls)
        self.assertFalse(analyze_controls.paired_contrasts(frame).empty)

    def test_rationale_reruns_and_reliability_cues(self):
        fixed = self.tmp / "controls_rat_fixed.jsonl"
        run(load("controls/rerun_rationale.py", "rerun_rationale"),
            "--source", self.controls, "--out", fixed, "--max-workers", 2)
        self.assertEqual({r["condition"] for r in read_jsonl(fixed)}, {"A3_rat", "A3_rat_gated"})
        gradient = self.tmp / "gradient.jsonl"
        run(load("controls/run_credibility_gradient.py", "run_credibility_gradient"),
            "--source", self.controls, "--out", gradient, "--max-workers", 2)
        self.assertEqual({r["condition"] for r in read_jsonl(gradient)}, {"A3_unc", "A3_track"})

    def test_no_padding_rationale_run_and_dedupe(self):
        raw = self.tmp / "clean_raw.jsonl"
        run(load("controls/run_clean_rationale_controls.py", "run_clean_rationale_controls"),
            "--source", self.controls, "--out", raw, "--limit", 3, "--max-workers", 2)
        dedup = self.tmp / "clean_dedup.jsonl"
        run(load("controls/dedupe_clean_rationale.py", "dedupe_clean_rationale"), raw, dedup)
        sys.path.insert(0, str(ARTIFACT / "results"))
        import analyze_clean_rationale_controls
        frame = analyze_clean_rationale_controls.load([dedup])
        self.assertEqual(set(frame.condition), {"A1_rat_clean", "A3_rat_clean", "A3_rat_gated_clean"})

    def test_budgeted_runner_records_costs(self):
        module = load("controls/run_budgeted_clean.py", "run_budgeted_clean")
        module.requests = StubRequests(module.MODELS)
        out = self.tmp / "clean_sota.jsonl"
        run(module, "--out", out, "--pilot-items", 1, "--workers", 2)
        rows = read_jsonl(out)
        self.assertEqual(len(rows), len(module.MODELS) * 3)
        self.assertTrue(all(r["charged_cost_usd"] is not None for r in rows))

    def test_rationale_generation(self):
        generate = load("controls/gen_opponent_rationales.py", "gen_opponent_rationales")
        fix = load("controls/fix_opponent_rationales.py", "fix_opponent_rationales")
        generate.OUT = fix.OUT = self.tmp / "opponent_rationales.json"
        run(generate, "--limit", 2, "--max-workers", 2)
        run(fix, "--max-workers", 2)
        data = json.loads(generate.OUT.read_text(encoding="utf-8"))
        self.assertEqual(len(data), 2)
        self.assertTrue(all("compliant" in record for record in data.values()))

    def test_trial_table_rebuilds_released_claude_rows(self):
        import gzip
        raw = self.tmp / "claude_main_audit.jsonl"
        with gzip.open(ARTIFACT / "data-repair" / "sources" / "original_log.jsonl.gz", "rb") as f:
            raw.write_bytes(f.read())
        table = self.tmp / "claude_table.csv"
        run(load("build_trial_table.py", "build_trial_table"), raw, "--out", table)
        with table.open(newline="") as f:
            rebuilt = list(csv.DictReader(f))
        with MAIN_TABLE.open(newline="") as f:
            released = [r for r in csv.DictReader(f) if r["model_id"] == "anthropic/claude-sonnet-4"]
        self.assertEqual(rebuilt, released)

    def test_control_pool_rebuilds_byte_for_byte(self):
        run(load("controls/build_control_items.py", "build_control_items"), "--out", self.tmp / "items.jsonl")
        self.assertEqual((self.tmp / "items.jsonl").read_bytes(), CONTROL_POOL.read_bytes())

    @unittest.skipUnless(GPT2, "gpt2 tokenizer unavailable")
    def test_main_audit_and_trial_table(self):
        questions = self.tmp / "questions.jsonl"
        questions.write_text("".join(INPUTS.read_text(encoding="utf-8").splitlines(True)[:3]), encoding="utf-8")
        trials = self.tmp / "main_audit_trials.jsonl"
        run(load("run_experiment.py", "run_experiment"), "--questions", questions,
            "--models-to-run", "openai/gpt-4o-mini", "--output", trials, "--no-parallel")
        records = read_jsonl(trials)
        self.assertEqual(sorted(r["reputation_factor"] for r in records), ["A0", "A0", "A0", "A1", "A1", "A1",
                                                                           "A3", "A3", "A3"])
        table = self.tmp / "main_audit_trials.csv"
        run(load("build_trial_table.py", "build_trial_table"), trials, "--out", table)
        with table.open() as f, MAIN_TABLE.open() as g:
            self.assertEqual(next(csv.reader(f)), next(csv.reader(g)))

    @unittest.skipUnless(GPT2, "gpt2 tokenizer unavailable")
    def test_mitigation_and_low_prestige_subsets(self):
        mitigation = load("subsets/run_mitigation_subset.py", "run_mitigation_subset")
        records = self.tmp / "mitigation_subset.jsonl"
        summary = self.tmp / "mitigation_subset_summary.csv"
        run(mitigation, "--limit", 3, "--models", "openai/gpt-4o-mini", "--max-workers", 2,
            "--output", records, "--summary", summary)
        combined = self.tmp / "mitigation_summary.csv"
        by_domain = self.tmp / "mitigation_by_domain.csv"
        run(load("subsets/summarize_mitigation.py", "summarize_mitigation"), summary, "--records", records,
            "--out", combined, "--domain-out", by_domain)
        self.assertTrue(combined.exists() and by_domain.exists())
        prestige = load("subsets/run_prestige_control_subset.py", "run_prestige_control_subset")
        prestige.SOURCE_FILES = [records]
        out = self.tmp / "prestige_control_subset.jsonl"
        run(prestige, "--models", "openai/gpt-4o-mini", "--max-workers", 2,
            "--output", out, "--summary", self.tmp / "prestige_control_summary.csv")
        self.assertEqual({r["condition"] for r in read_jsonl(out) if r.get("status") == "ok"}, {"A1", "A2"})

    @unittest.skipUnless(GPT2, "gpt2 tokenizer unavailable")
    def test_original_pipeline_replication(self):
        out = self.tmp / "temporal.jsonl"
        run(load("controls/verify_temporal_original_pipeline.py", "verify_temporal_original_pipeline"),
            "--out", out, "--max-workers", 4)
        rows = read_jsonl(out)
        self.assertEqual({r["condition"] for r in rows}, {"A0", "A1", "A3"})


if __name__ == "__main__":
    unittest.main()
