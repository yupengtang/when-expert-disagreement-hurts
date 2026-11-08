#!/usr/bin/env python3
"""Build the de-duplicated main-audit trial table from raw run_experiment.py output.

The table is produced by plot_source.load_trials, the function that produced
the released results/data/main_audit/neurips_dedup_trials_FULL.csv. It keeps
B0 evidence and the A0/A1/A3 conditions, requires parseable, untruncated and
format-compliant answers in both passes, and keeps the first record for each
model, condition and question.

Example, from the experiment/ directory:
    python build_trial_table.py ../runs/main_audit_trials.jsonl
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARTIFACT = HERE.parent
sys.path.insert(0, str(ARTIFACT))

import plot_source  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="+", type=Path, help="raw JSONL files written by run_experiment.py")
    parser.add_argument("--questions", type=Path,
                        default=ARTIFACT / "data-repair" / "outputs" / "original_benchmark_recovered.jsonl")
    parser.add_argument("--out", type=Path, default=ARTIFACT / "runs" / "main_audit_trials.csv")
    args = parser.parse_args()
    plot_source.QUESTIONS = args.questions
    plot_source.RESULTS = args.results
    table, duplicates = plot_source.load_trials()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"Wrote {args.out}: {len(table)} rows; duplicates removed: {sum(duplicates.values())}")
    print(table.groupby(["model", "condition"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
