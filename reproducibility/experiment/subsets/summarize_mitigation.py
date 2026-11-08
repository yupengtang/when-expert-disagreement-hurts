#!/usr/bin/env python3
"""Summarize the mitigation subset.

Combines the per-run summaries into the model-level table released as
results/data/main_audit/neurips_mitigation_summary.csv and, given the raw
records, counts Correct-to-Wrong and Wrong-to-Correct reversals by model and
domain for A3 and evidence-gated A3 (appendix table "Mitigation Breakdown by
Domain and Model").

Example, from the experiment/ directory:
    python subsets/summarize_mitigation.py ../runs/mitigation_subset_summary.csv \
        --records ../runs/mitigation_subset.jsonl
"""

import argparse
import json
from pathlib import Path

import pandas as pd

ARTIFACT = Path(__file__).resolve().parents[2]
ORDER = ["GPT-4o-mini", "Gemini-2.0-Flash", "Claude-Sonnet-4"]
DOMAIN_ORDER = ["GPT-4o-mini", "Claude-Sonnet-4", "Gemini-2.0-Flash"]
DOMAINS = ["Medicine", "Science", "Law"]


def by_domain(paths):
    """Reversal-direction counts for records with valid outputs in both branches."""
    stats = {}
    for path in paths:
        for line in Path(path).open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("base_valid", True) or not r.get("gated_valid", True) or "domain" not in r:
                continue
            model = r["model"]
            cell = stats.setdefault(model, {}).setdefault(
                r["domain"], {"A3": {"cw": 0, "wc": 0}, "A3-gated": {"cw": 0, "wc": 0}})
            for branch, prefix in [("A3", "base"), ("A3-gated", "gated")]:
                if r[f"{prefix}_reversal"]:
                    if r["pass1_correct"] and not r[f"{prefix}_correct"]:
                        cell[branch]["cw"] += 1
                    elif not r["pass1_correct"] and r[f"{prefix}_correct"]:
                        cell[branch]["wc"] += 1
    rows = []
    for model in DOMAIN_ORDER:
        for domain in DOMAINS:
            if domain in stats.get(model, {}):
                c = stats[model][domain]
                rows.append({"Model": model, "Domain": domain,
                             "A3 C_to_W": c["A3"]["cw"], "A3 W_to_C": c["A3"]["wc"],
                             "A3-gated C_to_W": c["A3-gated"]["cw"], "A3-gated W_to_C": c["A3-gated"]["wc"]})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("summaries", nargs="+", type=Path, help="CSV summaries written by run_mitigation_subset.py")
    parser.add_argument("--records", nargs="*", type=Path, default=[],
                        help="JSONL records written by run_mitigation_subset.py")
    parser.add_argument("--out", type=Path, default=ARTIFACT / "runs" / "neurips_mitigation_summary.csv")
    parser.add_argument("--domain-out", type=Path, default=ARTIFACT / "runs" / "mitigation_by_domain.csv")
    args = parser.parse_args()
    df = pd.concat([pd.read_csv(path) for path in args.summaries], ignore_index=True)
    df["Model"] = pd.Categorical(df["Model"], ORDER, ordered=True)
    df = df.sort_values(["Model", "Condition"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")
    if args.records:
        table = by_domain(args.records)
        table.to_csv(args.domain_out, index=False)
        print(table.to_string(index=False))
        print(f"Wrote {args.domain_out}")


if __name__ == "__main__":
    main()
