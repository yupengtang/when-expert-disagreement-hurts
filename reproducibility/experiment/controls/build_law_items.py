#!/usr/bin/env python3
"""Convert the official ContractNLI release into the audit's Law item schema.

The HuggingFace ContractNLI loaders used by ``scripts/collect_candidates.py``
are dataset scripts, which recent ``datasets`` releases refuse to execute, so
this builder reads the official Stanford release directly.

Excerpt construction: the annotated evidence spans are decisive for the label,
so the excerpt is the contiguous document window covering those spans, widened
by neighbouring spans until it reaches roughly the character length of the
Medicine excerpts (~1.7k chars). Windows longer than 3,200 characters are cut
at a word boundary. ``b2_evidence`` stays the annotated spans themselves.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
CNLI = ROOT / "data" / "raw" / "contractnli" / "contract-nli"
TARGET_CHARS = 1700
MAX_CHARS = 3200


def build_from_doc(doc: Dict[str, Any], hyp_id: str, hyp_text: str) -> Optional[Dict[str, Any]]:
    ann_sets = doc.get("annotation_sets") or []
    if not ann_sets:
        return None
    ann = (ann_sets[0].get("annotations") or {}).get(hyp_id)
    if not ann:
        return None
    choice = ann.get("choice")
    if choice not in ("Entailment", "Contradiction"):
        return None  # drop NotMentioned

    spans: List[List[int]] = doc.get("spans") or []
    idxs = sorted(i for i in (ann.get("spans") or []) if 0 <= i < len(spans))
    if not idxs:
        return None

    text = doc.get("text") or ""
    span_texts = [text[spans[i][0]: spans[i][1]].strip() for i in idxs]
    span_texts = [s for s in span_texts if s]
    if not span_texts:
        return None
    b2_evidence = " ".join(span_texts)

    # Widen a window of span indices around the decisive spans.
    lo, hi = idxs[0], idxs[-1]
    while True:
        start_c, end_c = spans[lo][0], spans[hi][1]
        if end_c - start_c >= TARGET_CHARS:
            break
        grew = False
        if lo > 0:
            lo -= 1
            grew = True
        if hi < len(spans) - 1:
            hi += 1
            grew = True
        if not grew:
            break
    excerpt = text[spans[lo][0]: spans[hi][1]].strip()
    if len(excerpt) > MAX_CHARS:
        # Cut at a word boundary. In 16 of the 201 Law items of the control pool
        # this drops a later annotated span; data-repair/ releases evidence-complete
        # versions of those items.
        excerpt = excerpt[:MAX_CHARS].rsplit(" ", 1)[0]
    if len(excerpt) < 200:
        return None

    ground_truth = "Yes" if choice == "Entailment" else "No"
    prompt_base = (
        "Given the contract excerpt below, evaluate the following hypothesis:\n\n"
        f"Hypothesis: {hyp_text}\n\n"
        "Answer 'Yes' if the hypothesis is entailed by the contract, or 'No' if it is contradicted.\n"
        "Answer ONLY with 'Yes' or 'No'. No explanation or additional text."
    )
    prompt_base_experiment = (
        "Given the contract excerpt below, is the following hypothesis ENTAILED by the contract?\n\n"
        f"Hypothesis: {hyp_text}\n\n"
        "Answer with 'Yes' or 'No' on the first line, followed by 4-6 sentences "
        "explaining your reasoning based on the contract excerpt."
    )
    return {
        "domain": "Law",
        "source": "ContractNLI",
        "prompt_base": prompt_base,
        "prompt_base_experiment": prompt_base_experiment,
        "excerpt": excerpt,
        "ground_truth": ground_truth,
        "metadata": {
            "original_hypothesis": hyp_text,
            "original_label": choice,
            "b2_evidence": b2_evidence,
            "evidence_span_count": len(idxs),
            "contract_file": doc.get("file_name"),
            "hypothesis_id": hyp_id,
        },
    }


def load_law_candidates() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for split in ("train.json", "dev.json", "test.json"):
        p = CNLI / split
        if not p.exists():
            continue
        blob = json.load(p.open())
        labels = blob.get("labels") or {}
        for doc in blob.get("documents") or []:
            for hyp_id, meta in labels.items():
                hyp_text = (meta or {}).get("hypothesis") or ""
                if not hyp_text:
                    continue
                rec = build_from_doc(doc, hyp_id, hyp_text)
                if rec:
                    out.append(rec)
    return out


if __name__ == "__main__":
    cands = load_law_candidates()
    from collections import Counter
    import statistics

    print(f"Law candidates: {len(cands)}")
    print("labels:", Counter(c["ground_truth"] for c in cands))
    if cands:
        print("excerpt chars median:",
              statistics.median(len(c["excerpt"]) for c in cands))
        print("b2 chars median:",
              statistics.median(len(c["metadata"]["b2_evidence"]) for c in cands))
