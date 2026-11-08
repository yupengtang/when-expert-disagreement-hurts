#!/usr/bin/env python3
"""Run a low-prestige / anonymous control subset using stored Pass-1 answers.

This script reuses the mitigation subset Pass-1 outputs and runs two paired
Pass-2 conditions on the same item-model pairs:
  - A1: anonymous disagreement
  - A2: low-prestige / no-special-expertise disagreement

The goal is to probe whether the effect is tied to prestige wording rather than
to generic metadata around a disagreeing assistant.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client import Client
from src.data_structures import Question, load_questions
from src.experiment_runner import TwoPassExperiment
from src.prompt_utils import compute_length_matched_prompts


# Pass-1 answers come from the mitigation subset (run_mitigation_subset.py).
SOURCE_FILES = [
    ROOT.parent / "runs" / "mitigation_subset.jsonl",
]

OUTPUT_CONDITIONS = ["A1", "A2"]
TARGET_REPS = {"A1", "A2"}
BASE_FRAMING = "C1"
EVIDENCE_LEVEL = "B0"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_question_map() -> Dict[str, Question]:
    return {q.question_id: q for q in load_questions(ROOT.parent / "data-repair" / "outputs" / "original_benchmark_recovered.jsonl")}


def load_source_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in SOURCE_FILES:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("status") != "ok":
                    continue
                records.append(r)
    return records


def token_len(text: str, tokenizer: Any = None) -> int:
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return max(1, len(text) // 4)


def run_one(args: Tuple[Dict[str, Any], str, str, Optional[str], Optional[float]]) -> List[Dict[str, Any]]:
    source_record, model_id, pass1_answer, tokenizer_name, temperature_override = args
    question_map = load_question_map()
    question = question_map[source_record["question_id"]]
    registry = json.loads((ROOT / "prompts" / "registry.json").read_text())
    experiment = TwoPassExperiment(config=None, prompts_registry=registry)
    client = Client()

    tokenizer = None
    if tokenizer_name:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception:
            tokenizer = None

    base_prompts = {}
    for rep in OUTPUT_CONDITIONS + ["A3"]:
        base_prompts[rep] = experiment.build_pass2_prompt(
            question,
            pass1_answer,
            rep,
            EVIDENCE_LEVEL,
            BASE_FRAMING,
            padding="",
        )

    if tokenizer is not None:
        padded = compute_length_matched_prompts(
            base_prompts,
            tokenizer,
            target_tolerance=10,
            filler_variants=registry.get("padding_policy", {}).get("filler_variants", None),
        )
    else:
        padded = {k: (v, token_len(v, tokenizer), token_len(v, tokenizer), "") for k, v in base_prompts.items()}

    outputs: List[Dict[str, Any]] = []
    for rep in OUTPUT_CONDITIONS:
        prompt, target_len, actual_len, padding_text = padded[rep]
        temperature = (
            float(temperature_override)
            if temperature_override is not None
            else float(source_record.get("temperature", 0.7))
        )
        p2 = experiment.run_pass_api(
            client,
            model_id,
            prompt,
            temperature,
            float(source_record.get("top_p", 1.0)),
            int(source_record.get("max_tokens", 600)),
        )
        gt = source_record.get("ground_truth")
        output = {
            "status": "ok" if p2.parsed_label is not None else "invalid",
            "source_trial_id": source_record.get("trial_id"),
            "question_id": source_record["question_id"],
            "domain": source_record["domain"],
            "confidence_tier": source_record.get("confidence_tier"),
            "model_id": model_id,
            "model": source_record.get("model"),
            "condition": rep,
            "pass1_answer": pass1_answer,
            "pass2_answer": p2.parsed_label,
            "pass1_correct": pass1_answer == gt,
            "pass2_correct": p2.parsed_label == gt if p2.parsed_label is not None else False,
            "reversal": p2.parsed_label is not None and p2.parsed_label != pass1_answer,
            "ground_truth": gt,
            "base_prompt_tokens": target_len,
            "actual_prompt_tokens": actual_len,
            "padding_hash": hashlib.sha256((padding_text or "").encode("utf-8")).hexdigest()[:8],
            "raw_output": p2.raw_output,
            "finish_reason": p2.finish_reason,
            "latency_seconds": p2.latency_seconds,
            "usage": p2.token_usage,
        }
        outputs.append(output)
    return outputs


def summarize(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    import pandas as pd

    df = pd.DataFrame([r for r in records if r.get("status") == "ok"])
    rows: List[Dict[str, Any]] = []
    for model_id, model_df in df.groupby("model_id"):
        for condition, cond_df in model_df.groupby("condition"):
            rev = cond_df["reversal"]
            cw = ((cond_df["pass1_correct"]) & (~cond_df["pass2_correct"]) & rev).sum()
            wc = ((~cond_df["pass1_correct"]) & (cond_df["pass2_correct"]) & rev).sum()
            rows.append(
                {
                    "Model": model_df["model"].iloc[0],
                    "Model ID": model_id,
                    "Condition": condition,
                    "N": len(cond_df),
                    "Pass1 Acc": cond_df["pass1_correct"].mean(),
                    "Final Acc": cond_df["pass2_correct"].mean(),
                    "Delta Acc": cond_df["pass2_correct"].mean() - cond_df["pass1_correct"].mean(),
                    "Reversal Rate": rev.mean(),
                    "Correct_to_Wrong": int(cw),
                    "Wrong_to_Correct": int(wc),
                    "Harmful Percent": 100 * cw / max(1, int(rev.sum())),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1000, help="Maximum source records to process")
    parser.add_argument(
        "--models",
        type=str,
        default="openai/gpt-4o-mini,google/gemini-2.0-flash-001,anthropic/claude-sonnet-4",
        help="Comma-separated model IDs to include",
    )
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override the sampling temperature for all paired runs; defaults to the source-record temperature.",
    )
    parser.add_argument("--output", type=str, default=str(ROOT.parent / "runs" / "prestige_control_subset.jsonl"))
    parser.add_argument("--summary", type=str, default=str(ROOT.parent / "runs" / "prestige_control_summary.csv"))
    args = parser.parse_args()

    load_env_file(ROOT / ".env")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    selected = [r for r in load_source_records() if r.get("model_id") in models]
    rng = random.Random(args.seed)
    rng.shuffle(selected)
    selected = selected[: args.limit]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    tasks = [(r, r["model_id"], r["pass1_answer"], "gpt2", args.temperature) for r in selected]
    all_records: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(run_one, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="prestige-control"):
            try:
                recs = fut.result()
            except Exception as e:
                all_records.append({"status": "error", "error": str(e)})
                continue
            all_records.extend(recs)
            with out_path.open("a") as f:
                for rec in recs:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    rows = summarize(all_records)
    import pandas as pd

    pd.DataFrame(rows).sort_values(["Model", "Condition"]).to_csv(args.summary, index=False)
    print(f"Wrote {out_path}")
    print(f"Wrote {args.summary}")


if __name__ == "__main__":
    main()
