#!/usr/bin/env python3
"""Run the January protocol with the original pipeline, byte-for-byte.

The July control suite (run_controls.py) rebuilt the prompts and differs from
the main-audit runner in four respects:

  * it sends a system message ("... based ONLY on the provided evidence excerpt");
    the original sends a single user message and no system role
  * it length-matches with a chars/4 approximation; the original uses the gpt2
    tokenizer via compute_length_matched_prompts
  * it lower-cases {domain} in the source block
  * it passes no stop sequences

Any of these could by itself reduce reversal under unsupported disagreement, so
a January-to-July difference cannot be attributed to provider drift without
re-running the original construction. This script uses the main-audit
TwoPassExperiment, so the collection date is the only difference from January.
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
sys.path.insert(0, str(ROOT))

from client import Client  # noqa: E402
from src.data_structures import Question  # noqa: E402
from src.experiment_runner import TwoPassExperiment  # noqa: E402
from src.prompt_utils import compute_length_matched_prompts  # noqa: E402

CONDS = ["A0", "A1", "A3"]
_lock = threading.Lock()


def load_env(path: Path) -> None:
    if not path.exists():
        return
    import os
    for line in path.open():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def run_item(item: dict, model: str, registry: dict, tok, temperature: float,
             max_tokens: int, out: Path):
    exp = TwoPassExperiment(config=None, prompts_registry=registry)
    client = Client()
    q = Question(question_id=item["question_id"], domain=item["domain"],
                 prompt_base=item["prompt_base"], excerpt=item["excerpt"],
                 ground_truth=item["ground_truth"],
                 confidence_tier=item.get("confidence_tier", "High"),
                 source=item.get("source"), metadata=item.get("metadata") or {},
                 hash=item.get("hash", ""))

    p1 = exp.run_pass_api(client, model, exp.build_pass1_prompt(q),
                          temperature, 1.0, max_tokens)
    if p1.parsed_label is None or p1.truncated or p1.format_violation:
        return []

    prompts = {c: exp.build_pass2_prompt(q, p1.parsed_label, c, "B0", "C1", padding="")
               for c in CONDS}
    if tok is not None:
        fillers = registry.get("padding_policy", {}).get("filler_variants")
        padded = compute_length_matched_prompts(prompts, tok, target_tolerance=10,
                                                filler_variants=fillers)
    else:
        padded = {k: (v, 0, 0, "") for k, v in prompts.items()}

    recs = []
    for c in CONDS:
        prompt = padded[c][0]
        p2 = exp.run_pass_api(client, model, prompt, temperature, 1.0, max_tokens)
        gt = item["ground_truth"]
        recs.append({
            "status": "ok" if p2.parsed_label else "pass2_invalid",
            "question_id": q.question_id, "domain": q.domain, "model_id": model,
            "condition": c, "pass1": p1.parsed_label, "pass2": p2.parsed_label,
            "ground_truth": gt, "pass1_correct": p1.parsed_label == gt,
            "pass2_correct": (p2.parsed_label == gt) if p2.parsed_label else None,
            "reversal": p2.parsed_label is not None and p2.parsed_label != p1.parsed_label,
            "pipeline": "original",
        })
    with _lock, out.open("a") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--out", required=True)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--max-workers", type=int, default=16)
    args = ap.parse_args()

    load_env(ROOT / ".env")

    registry = json.loads((ROOT / "prompts" / "registry.json").read_text())
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
    except Exception as e:
        raise SystemExit(f"The gpt2 tokenizer is required for length matching: {e}")

    # the Medicine items shared with the January run
    items = [json.loads(l) for l in
             (ROOT.parent / "results" / "data" / "control_items.jsonl").open() if l.strip()]
    items = [i for i in items if i["domain"] == "Medicine"]
    print(f"{len(items)} shared Medicine items, model {args.model}, "
          f"{len(items)*(1+len(CONDS))} calls")

    out = Path(args.out)
    if out.exists():
        out.unlink()
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(run_item, it, args.model, registry, tok,
                          args.temperature, args.max_tokens, out) for it in items]
        errs = 0
        for f in tqdm(as_completed(futs), total=len(futs), desc="orig-pipeline"):
            try:
                f.result()
            except Exception as e:
                errs += 1
                if errs <= 5:
                    print(f"error: {type(e).__name__}: {e}")
    print(f"done errors={errs}; wrote {out}")


if __name__ == "__main__":
    main()
