#!/usr/bin/env python3
"""No-padding rationale-bearing conditions with anonymous and expert sources.

Compared with the padded rationale conditions of run_controls.py, this script

1. adds an anonymous+rationale condition, so the expert-label effect under
   rationale-bearing disagreement is identified by A3_rat_clean - A1_rat_clean;
2. does not append length-matching instructions after the question. The same
   rationale, question, excerpt, Pass-1 answer, decoding settings, and message
   structure are reused across the anonymous and expert conditions.

The rationale target is recorded as benchmark-matching or benchmark-conflicting.
The compliance field only verifies that the generated text starts in the
assigned answer direction; it is not treated as factual validation.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from client import Client  # noqa: E402
from run_controls import (  # noqa: E402
    GATE,
    ITEMS,
    PASS1_SYSTEM,
    RATIONALES,
    SOURCE_BLOCKS,
    approx_tokens,
    load_env,
    parse_answer,
    question_block,
    strip_leading_label,
)


CONDITIONS = {
    "A1_rat_clean": {"source": "A1", "gate": False},
    "A3_rat_clean": {"source": "A3", "gate": False},
    "A3_rat_gated_clean": {"source": "A3", "gate": True},
}

_write_lock = threading.Lock()


def build_prompt(item: dict, pass1: str, rationale: str, condition: str) -> str:
    spec = CONDITIONS[condition]
    opponent = "No" if pass1 == "Yes" else "Yes"
    source = SOURCE_BLOCKS[spec["source"]].format(
        ans=opponent, domain=item["domain"].lower()
    )
    source += f" Their stated reason: {strip_leading_label(rationale)}"
    parts = [
        "Please reconsider the question carefully using only the provided excerpt.",
        source,
    ]
    if spec["gate"]:
        parts.append(GATE)
    parts.append(question_block(item))
    return "\n\n".join(parts)


def load_shared_pass1(source_paths: list[Path], keep_models: set[str]) -> dict:
    """Recover one existing Pass-1 answer for every model-item pair."""
    pass1 = {}
    for path in source_paths:
        for line in path.open():
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            model = row.get("model_id")
            qid = row.get("question_id")
            answer = row.get("pass1")
            if row.get("status") == "ok" and model and qid and answer:
                if not keep_models or model in keep_models:
                    pass1.setdefault((model, qid), answer)
    return pass1


def run_one(item: dict, model: str, pass1: str, rationale_record: dict,
            selected: list[str], temperature: float, max_tokens: int,
            out_path: Path) -> list[dict]:
    opponent = "No" if pass1 == "Yes" else "Yes"
    rationale = rationale_record.get(opponent)
    compliant = (rationale_record.get("compliant") or {}).get(opponent)
    if not rationale or compliant is False:
        return []

    relation = (
        "benchmark_matching" if opponent == item["ground_truth"]
        else "benchmark_conflicting"
    )
    client = Client()
    rows = []
    for condition in selected:
        prompt = build_prompt(item, pass1, rationale, condition)
        result = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": PASS1_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0,
        )
        raw = result.get("content") or ""
        pass2 = parse_answer(raw)
        rows.append({
            "status": "ok" if pass2 else "pass2_invalid",
            "question_id": item["question_id"],
            "domain": item["domain"],
            "source": item.get("source"),
            "model_id": model,
            "condition": condition,
            "opponent_agrees": False,
            "gated": CONDITIONS[condition]["gate"],
            "has_rationale": True,
            "rationale_target_relation": relation,
            "rationale_compliant": compliant,
            "rationale_factually_validated": False,
            "opponent_answer": opponent,
            "pass1": pass1,
            "pass2": pass2,
            "ground_truth": item["ground_truth"],
            "pass1_correct": pass1 == item["ground_truth"],
            "pass2_correct": (pass2 == item["ground_truth"]) if pass2 else None,
            "reversal": (pass2 is not None and pass2 != pass1),
            "temperature": temperature,
            "prompt_tokens_approx": approx_tokens(prompt),
            "padding_policy": "none",
            "raw_pass2": raw[:1200],
            "usage": result.get("usage"),
        })

    with _write_lock, out_path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Existing control JSONL containing the shared Pass-1 answers; repeatable.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--models", default="")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Use the first N items of the control pool (0 uses every item in the source).",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = sorted(set(selected) - set(CONDITIONS))
    if unknown:
        parser.error(f"unknown conditions: {', '.join(unknown)}")

    load_env(ROOT / ".env")
    item_rows = [json.loads(line) for line in ITEMS.open() if line.strip()]
    if args.limit > 0:
        item_rows = item_rows[:args.limit]
    items = {row["question_id"]: row for row in item_rows}
    rationales = json.loads(RATIONALES.read_text())
    keep_models = {m.strip() for m in args.models.split(",") if m.strip()}
    pass1 = load_shared_pass1([Path(p) for p in args.source], keep_models)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out_path.exists():
        for line in out_path.open():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((row.get("model_id"), row.get("question_id"), row.get("condition")))

    tasks = []
    calls = 0
    for (model, qid), answer in pass1.items():
        if qid not in items or qid not in rationales:
            continue
        needed = [c for c in selected if (model, qid, c) not in done]
        if needed:
            tasks.append((items[qid], model, answer, rationales[qid], needed))
            calls += len(needed)

    print(f"model-item pairs={len(tasks)} conditions={selected} calls={calls}")
    if args.dry_run:
        if tasks:
            item, _, answer, record, _ = tasks[0]
            opponent = "No" if answer == "Yes" else "Yes"
            for condition in selected:
                prompt = build_prompt(item, answer, record[opponent], condition)
                print(f"{condition}: chars={len(prompt)} approx_tokens={approx_tokens(prompt)}")
        return

    if out_path.exists() and not args.resume:
        parser.error(f"output exists: {out_path}; use --resume or choose a new path")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                run_one, item, model, answer, record, needed,
                args.temperature, args.max_tokens, out_path,
            )
            for item, model, answer, record, needed in tasks
        ]
        errors = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="clean-rationale"):
            try:
                future.result()
            except Exception as exc:
                errors += 1
                if errors <= 5:
                    print(f"error: {type(exc).__name__}: {exc}")
    print(f"done errors={errors}; wrote {out_path}")


if __name__ == "__main__":
    main()
