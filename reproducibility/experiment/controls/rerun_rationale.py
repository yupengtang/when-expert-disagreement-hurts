#!/usr/bin/env python3
"""Run the two padded rationale conditions with the compliant opponent rationales.

The Pass-1 answer is not re-sampled: it is read from the existing control
records for the same (model, item), so these rows remain branch-paired with
every other condition in that run. Prompts are rebuilt and length-matched
against the full 10-condition set exactly as in run_controls.py.

Output rows carry the condition names A3_rat / A3_rat_gated and are written to
a separate file (results/data/*_rat_fixed.jsonl in the release); the analysis
uses them in place of the A3_rat / A3_rat_gated rows of the control file. For
a no-padding A1+rationale/A3+rationale contrast, use
run_clean_rationale_controls.py instead.
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
    CONDITIONS, ITEMS, RATIONALES, PASS1_SYSTEM, approx_tokens,
    build_pass2, length_match, load_env, parse_answer,
)

TARGETS = {"A3_rat", "A3_rat_gated"}
_lock = threading.Lock()


def run_one(item, model, pass1, temperature, max_tokens, rationales, out_path):
    client = Client()
    gt = item["ground_truth"]
    opp = "No" if pass1 == "Yes" else "Yes"
    rec_r = rationales.get(item["question_id"]) or {}
    rat = rec_r.get(opp)
    quality = (rec_r.get("quality") or {}).get(opp)
    compliant = (rec_r.get("compliant") or {}).get(opp)

    prompts = {c[0]: build_pass2(item, pass1, c, rat) for c in CONDITIONS}
    prompts = length_match(prompts)

    out = []
    for spec in CONDITIONS:
        cond = spec[0]
        if cond not in TARGETS:
            continue
        p2 = client.chat(
            model=model,
            messages=[{"role": "system", "content": PASS1_SYSTEM},
                      {"role": "user", "content": prompts[cond]}],
            temperature=temperature, max_tokens=max_tokens, top_p=1.0,
        )
        t2 = p2.get("content") or ""
        pass2 = parse_answer(t2)
        out.append({
            "status": "ok" if pass2 else "pass2_invalid",
            "question_id": item["question_id"], "domain": item["domain"],
            "source": item.get("source"), "model_id": model, "condition": cond,
            "opponent_agrees": spec[2], "gated": spec[3],
            "has_rationale": bool(rat), "rationale_quality": quality,
            "rationale_compliant": compliant,
            "opponent_answer": opp, "pass1": pass1, "pass2": pass2,
            "ground_truth": gt, "pass1_correct": pass1 == gt,
            "pass2_correct": (pass2 == gt) if pass2 else None,
            "reversal": (pass2 is not None and pass2 != pass1),
            "temperature": temperature,
            "prompt_tokens_approx": approx_tokens(prompts[cond]),
            "raw_pass2": t2[:1200], "usage": p2.get("usage"),
        })
    with _lock, out_path.open("a") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="existing controls jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    load_env(ROOT / ".env")

    items = {json.loads(l)["question_id"]: json.loads(l)
             for l in ITEMS.open() if l.strip()}
    rationales = json.loads(RATIONALES.read_text())

    # read the shared Pass-1 answer for each (model, item) from the control run
    pass1 = {}
    for line in Path(args.source).open():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("status") == "ok" and r.get("pass1"):
            pass1.setdefault((r["model_id"], r["question_id"]), r["pass1"])

    out_path = Path(args.out)
    done = set()
    if args.resume and out_path.exists():
        for line in out_path.open():
            try:
                r = json.loads(line)
            except Exception:
                continue
            done.add((r.get("model_id"), r.get("question_id")))
    elif out_path.exists():
        out_path.unlink()

    tasks = [(items[q], m, p) for (m, q), p in pass1.items()
             if q in items and (m, q) not in done]
    print(f"re-running {len(TARGETS)} conditions for {len(tasks)} (model,item) pairs "
          f"= {len(tasks) * len(TARGETS)} calls")

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(run_one, it, m, p, args.temperature, args.max_tokens,
                          rationales, out_path) for it, m, p in tasks]
        errs = 0
        for fut in tqdm(as_completed(futs), total=len(futs), desc="rerun-rat"):
            try:
                fut.result()
            except Exception as e:
                errs += 1
                if errs <= 5:
                    print(f"error: {type(e).__name__}: {e}")
    print(f"done errors={errs}; wrote {out_path}")


if __name__ == "__main__":
    main()
