#!/usr/bin/env python3
"""Build the 500-item control pool used by the control experiments.

The pool follows the domain mix of the main benchmark (224 Medicine, 75
Science and 201 Law items) with balanced Yes/No labels inside each domain.
Medicine items are benchmark inputs (data-repair/sources/frozen_prefix.jsonl.gz).
Science items are normalized from the official SciFact release with the rules
of benchmark/collect_candidates.py, and Law items from the official
ContractNLI release with build_law_items.py. Every source is read from
data-repair/sources, so the build runs offline.

The pool is written to runs/control_items.jsonl and compared byte-for-byte
with the released results/data/control_items.jsonl.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_law_items  # noqa: E402
from collect_candidates import (  # noqa: E402
    hash_text,
    load_scifact_from_tarball,
    normalize_scifact_to_binary,
)

SOURCES = ARTIFACT / "data-repair" / "sources"
RELEASED = ARTIFACT / "results" / "data" / "control_items.jsonl"
SEED = 20260724

# Domain mix mirrors the benchmark proportions (890/299/800).
TARGETS = {"Medicine": 224, "Science": 75, "Law": 201}


def load_medicine(sources: Path) -> list:
    items = []
    with gzip.open(sources / "frozen_prefix.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # the stored prefix ends inside a record
    return items


def materialize_sources(sources: Path, work: Path) -> None:
    """Lay out the bundled SciFact and ContractNLI files as the loaders expect them."""
    scifact = work / "data" / "raw" / "scifact"
    (scifact / "data").mkdir(parents=True)
    (scifact / "data.tar.gz").touch()  # the release is already extracted below
    for name in ("corpus", "claims_train", "claims_dev"):
        with gzip.open(sources / f"scifact_{name}.jsonl.gz", "rb") as f:
            (scifact / "data" / f"{name}.jsonl").write_bytes(f.read())
    contract = work / "contract-nli"
    contract.mkdir()
    for split in ("train", "dev", "test"):
        with gzip.open(sources / f"contract_{split}.json.gz", "rb") as f:
            (contract / f"{split}.json").write_bytes(f.read())


def to_item(cand: dict, qid: str) -> dict:
    """Project a candidate record onto the benchmark question schema."""
    meta = dict(cand.get("metadata") or {})
    return {
        "question_id": qid,
        "candidate_id": cand.get("candidate_id", qid),
        "domain": cand["domain"],
        "source": cand["source"],
        "prompt_base": cand.get("prompt_base_experiment") or cand["prompt_base"],
        "excerpt": cand["excerpt"],
        "ground_truth": cand["ground_truth"],
        "confidence_tier": "Unscreened",
        "metadata": meta,
        "hash": cand.get("hash") or hash_text(cand["prompt_base"] + "|" + cand["excerpt"]),
    }


def build(sources: Path) -> list:
    rng = random.Random(SEED)
    pool: dict[str, list] = {"Medicine": [], "Science": [], "Law": []}

    med = load_medicine(sources)
    pool["Medicine"] = med
    print(f"Medicine benchmark items: {len(med)}")

    # The official SciFact release spells the refuting class "CONTRADICT"; the
    # normalizer recognizes "REFUTE(S)", the spelling of the HuggingFace build.
    def _relabel(item: dict) -> dict:
        ev = item.get("evidence") or {}
        for lst in ev.values():
            if isinstance(lst, list):
                for e in lst:
                    if str(e.get("label", "")).upper() == "CONTRADICT":
                        e["label"] = "REFUTES"
        return item

    with tempfile.TemporaryDirectory(prefix="control-pool-sources-") as tmp:
        work = Path(tmp)
        materialize_sources(sources, work)
        cwd = Path.cwd()
        os.chdir(work)  # load_scifact_from_tarball reads data/raw/scifact relative to the working directory
        try:
            claims, corpus = load_scifact_from_tarball()
        finally:
            os.chdir(cwd)
        print(f"SciFact raw claims={len(claims)} corpus_docs={len(corpus)}")
        pool["Science"] = [n for n in (normalize_scifact_to_binary(_relabel(it), corpus) for it in claims) if n]
        print(f"SciFact binary candidates: {len(pool['Science'])}")

        build_law_items.CNLI = work / "contract-nli"
        pool["Law"] = build_law_items.load_law_candidates()
        print(f"ContractNLI binary candidates: {len(pool['Law'])}")

    # Stratified sample with balanced Yes/No inside each domain.
    selected = []
    for domain, target in TARGETS.items():
        cands = pool[domain]
        by_label = {"Yes": [], "No": []}
        for c in cands:
            if c.get("ground_truth") in by_label:
                by_label[c["ground_truth"]].append(c)
        for lab in by_label:
            rng.shuffle(by_label[lab])
        half = target // 2
        take = by_label["Yes"][:half] + by_label["No"][: target - half]
        # top up if one label is short
        if len(take) < target:
            rest = [c for c in cands if c not in take]
            rng.shuffle(rest)
            take += rest[: target - len(take)]
        rng.shuffle(take)
        selected.extend(take[:target])
        print(f"{domain}: selected {len(take[:target])} "
              f"(Yes={sum(1 for c in take[:target] if c.get('ground_truth') == 'Yes')})")

    rng.shuffle(selected)
    items = []
    for i, cand in enumerate(selected, start=1):
        if "question_id" in cand and cand.get("domain") == "Medicine":
            item = cand  # already in the benchmark schema
            item["confidence_tier"] = item.get("confidence_tier", "High")
        else:
            item = to_item(cand, f"rq{i:04d}")
        items.append(item)
    print("domain:", Counter(c.get("domain") for c in items))
    print("label :", Counter((c.get("domain"), c.get("ground_truth")) for c in items))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sources", type=Path, default=SOURCES)
    parser.add_argument("--out", type=Path, default=ARTIFACT / "runs" / "control_items.jsonl")
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"output exists: {args.out}")
    items = build(args.sources)
    data = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items).encode("utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(data)
    print(f"Wrote {args.out} with {len(items)} items")
    if RELEASED.exists():
        same = RELEASED.read_bytes() == data
        print("Identical to results/data/control_items.jsonl:", same)
        if not same:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
