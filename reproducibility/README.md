# When Expert Disagreement Hurts: Supplementary Material

Code and data for the NeurIPS 2026 paper "When Expert Disagreement Hurts: Prestige and Selectivity in LLM Decision Revision" by Yupeng Tang and Mingfeng Lin (Georgia Institute of Technology). The code is also available at https://github.com/yupengtang/when-expert-disagreement-hurts.

The archive supports two levels of reproduction. The reported numbers, tables and figures can be regenerated from the released model outputs on a CPU, without credentials or network access (Section 1). Every experiment can also be run again from the released inputs with the code in `experiment/` (Section 2). Running the experiments requires an OpenRouter API key and, for the two open-weight models of the main audit, a CUDA GPU.

## Contents

| Path | Contents |
|---|---|
| `experiment/` | Code that queries the models: the main audit, the mitigation and low-prestige subsets, the control experiments, benchmark construction, and offline tests |
| `results/data/` | Released model outputs, the 500-item control pool and the opponent arguments |
| `results/analysis/`, `tables/`, `figs/` | Analysis outputs, LaTeX tables and figures generated from the released outputs |
| `reproduce.py`, `make_figures.py` | Entry points of the analysis; the other top-level modules and `results/*.py` are imported by them |
| `reanalysis/` | Paired accuracy analysis and data checks (appendix "Paired Accuracy Analysis and Data Checks") |
| `data-repair/` | Reconstruction and verification of the 1,989 benchmark inputs from the official dataset releases, and evidence-complete Law inputs (`data-repair/README.md`) |
| `source_manifest.json` | Size and SHA-256 of every released data file |

## 1. Reproducing the reported results

The analysis requires Python 3.9 or later and the packages in `requirements.txt`. From the archive root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s data-repair -p 'test_*.py' -v
python reproduce.py
python make_figures.py
python -m unittest discover -s reanalysis -p 'test_*.py' -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python reanalysis/audit.py
python reanalysis/render.py
```

`reproduce.py` writes the CSV files in `results/analysis/` and the LaTeX tables in `tables/`, and `make_figures.py` writes the figures in `figs/` with embedded TrueType fonts. `reanalysis/audit.py` and `reanalysis/render.py` write the paired accuracy analysis, its tables and its figures to `reanalysis/outputs/`. The commands make no network or API calls and take a few minutes on a CPU. On an unmodified copy of the archive they regenerate all 88 CSV and LaTeX outputs byte-for-byte; only the creation timestamps of the PDF and SVG figures change.

Before computing anything, `reanalysis/audit.py` compares the SHA-256 of its analysis plan (`reanalysis/analysis_plan.json`) and of every input file with `reanalysis/outputs/frozen_inputs.json`, and it stops if any of them differ. Its intervals use 20,000 bootstrap draws with seeds derived from the plan; they describe item or document sampling, not variation across repeated model runs. The relative-effect and interaction intervals of `reproduce.py` use 50,000 paired item-bootstrap resamples with the seeds recorded in the scripts.

## 2. Re-running the experiments

Set up the experiment environment and run the offline tests first:

```bash
cd experiment
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
export OPENROUTER_API_KEY=<your key>
```

The tests run every experiment script on a few items with a deterministic stand-in for the API client, so they need neither a key nor a network connection. Two of them compare with released files: `results/data/control_items.jsonl` is rebuilt byte-for-byte from the dataset releases in `data-repair/sources/`, and the Claude-Sonnet-4 rows of the main trial table are rebuilt from the complete generation log in `data-repair/sources/original_log.jsonl.gz`. The key can also be stored in `experiment/.env`. The commands below are run from `experiment/`. Every script writes to `runs/` at the archive root by default and never modifies the released files.

API behavior depends on the endpoint and the date, and some endpoints may be retired, so new outputs need not match the released ones. Section 3 describes how to analyze them with the released analysis code. `experiment/requirements.txt` lists the package versions of the environment that ran the main audit.

### 2.1 Main audit

```bash
python run_experiment.py \
    --models-to-run openai/gpt-4o-mini,anthropic/claude-sonnet-4,google/gemini-2.0-flash-001 \
    --output ../runs/main_audit_api.jsonl
python run_experiment.py \
    --models-to-run meta-llama/Llama-3.1-8B-Instruct,mistralai/Mistral-7B-Instruct-v0.3 \
    --device cuda --output ../runs/main_audit_local.jsonl
python build_trial_table.py ../runs/main_audit_api.jsonl ../runs/main_audit_local.jsonl
```

`configs/main_audit.json` sets the reported protocol: conditions A0 (neutral reflection), A1 (anonymous disagreement) and A3 (expert-attributed disagreement); T=0.7, top-p=1 and a 600-token limit; one replicate; and one Pass 1 per item and model, shared by all Pass-2 conditions. Prompt blocks come from `prompts/registry.json`, and Pass-2 prompts are length-matched across conditions (`src/prompt_utils.py`). The inputs are the 1,989 items of `data-repair/outputs/original_benchmark_recovered.jsonl`; the loader verifies their checksums. Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3 run locally, loaded in 8-bit precision with bitsandbytes, and record teacher-forced Yes/No log-odds in both passes; the released runs used one NVIDIA RTX PRO 6000 Blackwell GPU per model. `build_trial_table.py` applies the filtering and de-duplication that produced `results/data/main_audit/neurips_dedup_trials_FULL.csv` and writes `runs/main_audit_trials.csv` in the same format.

### 2.2 Mitigation and low-prestige subsets

```bash
python subsets/run_mitigation_subset.py
python subsets/summarize_mitigation.py ../runs/mitigation_subset_summary.csv \
    --records ../runs/mitigation_subset.jsonl
python subsets/run_prestige_control_subset.py
python subsets/run_prestige_control_subset.py --temperature 0 \
    --output ../runs/prestige_control_subset_t0.jsonl --summary ../runs/prestige_control_summary_t0.csv
```

The mitigation subset draws 500 domain-balanced benchmark items (seed 454) and runs A3 and evidence-gated A3 from a shared Pass 1 for GPT-4o-mini, Claude-Sonnet-4 and Gemini-2.0-Flash. The low-prestige control reuses those Pass-1 answers and runs A1 and A2 on up to 1,000 sampled item-model pairs (seed 20260505), at the temperature of the source records or at T=0. The outputs correspond to the released summaries `neurips_mitigation_summary.csv`, `prestige_control_summary.csv` and `prestige_control_summary_t0.csv` in `results/data/main_audit/`; `--records` also writes the reversal-direction counts by model and domain of the appendix "Mitigation Breakdown by Domain and Model".

### 2.3 Control experiments

The control suite runs ten conditions from one shared Pass 1 per item and model: A0, A1, A2, A4 (neutral source metadata), A3, A3g (evidence gate), A1_agree, A3_agree, A3_rat (expert argument) and A3_rat_gated. All control experiments use the 500-item pool `results/data/control_items.jsonl` and a 600-token limit; the open-weight models are queried through OpenRouter endpoints.

```bash
python controls/run_controls.py --out ../runs/controls.jsonl
python controls/run_controls.py --temperature 0 --limit 250 --out ../runs/controls_t0.jsonl
python controls/run_controls.py --limit 250 \
    --models openai/gpt-5.6-terra,anthropic/claude-opus-4.8,google/gemini-3.6-flash,x-ai/grok-4.5 \
    --out ../runs/controls_sota.jsonl
python controls/run_controls.py \
    --models meta-llama/llama-3.1-8b-instruct,meta-llama/llama-3.3-70b-instruct \
    --out ../runs/controls_open.jsonl
```

The rationale conditions are then run with arguments that state the assigned answer, once for each control file:

```bash
python controls/rerun_rationale.py --source ../runs/controls.jsonl --out ../runs/controls_rat_fixed.jsonl
```

`--source` names the control file whose Pass-1 answers are reused; a released file in `../results/data/` starts from the released Pass 1. The analysis uses these rows in place of the A3_rat and A3_rat_gated rows of the control file.

Asserted reliability (A3_unc and A3_track) and the no-padding rationale experiment:

```bash
python controls/run_credibility_gradient.py --source ../runs/controls.jsonl --out ../runs/gradient_main.jsonl
python controls/run_credibility_gradient.py --source ../runs/controls_sota.jsonl \
    --models openai/gpt-5.6-terra,x-ai/grok-4.5 --out ../runs/gradient_sota.jsonl
python controls/run_clean_rationale_controls.py --source ../runs/controls.jsonl --limit 250 \
    --out ../runs/clean_rationale_250.jsonl
python controls/dedupe_clean_rationale.py ../runs/clean_rationale_250.jsonl ../runs/clean_rationale_250_dedup.jsonl
python controls/run_budgeted_clean.py --out ../runs/clean_rationale_sota.jsonl --budget 50
```

`dedupe_clean_rationale.py` keeps the earliest record for each model, item and condition. `run_budgeted_clean.py` runs the no-padding experiment for the four newer models from the released July Pass-1 answers in `results/data/controls_sota.jsonl`. It queries the OpenRouter HTTP API directly, logs the charged cost of every call and stops at the budget.

The temporal replication reruns A0, A1 and A3 with the main-audit pipeline (prompt construction, gpt2 length matching and stop sequences) for GPT-4o-mini on the 224 Medicine items of the control pool, which are also benchmark items:

```bash
python controls/verify_temporal_original_pipeline.py --out ../runs/temporal_orig_pipeline.jsonl
```

### 2.4 Stimuli and the control pool

```bash
python controls/gen_opponent_rationales.py
python controls/fix_opponent_rationales.py
python controls/build_control_items.py
```

The first two scripts regenerate the opponent arguments with GPT-4o-mini into `runs/opponent_rationales.json`. The control-experiment runners read the released arguments in `results/data/opponent_rationales.json`, so a rerun presents the same stimuli as the reported experiments. `build_control_items.py` rebuilds the 500-item pool offline from `data-repair/sources/` and checks it byte-for-byte against the released file.

### 2.5 Benchmark construction

The 1,989 inputs of the main audit are released in `data-repair/outputs/original_benchmark_recovered.jsonl`. `data-repair/repair_inputs.py` rebuilds them offline from the official dataset releases and verifies them against the hashes recorded at generation time. The code that built the benchmark is in `experiment/benchmark/` and `experiment/src/beat300_builder.py`, which refer to the benchmark by its working name, BEAT:

```bash
python benchmark/collect_candidates.py --output ../runs/candidate_pool.jsonl
python -m src.beat300_builder --candidates ../runs/candidate_pool.jsonl \
    --selection-log ../runs/selection_log.jsonl --device cuda
python benchmark/generate_beat_full.py --candidates ../runs/candidate_pool.jsonl \
    --profiling-log ../runs/selection_log.jsonl --output ../runs/benchmark_questions.jsonl
```

Candidate collection downloads PubMedQA, SciFact and ContractNLI. Screening answers each candidate ten times at T=0.7 with Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3, and the last step keeps the screened items that pass the validity threshold (`--min-valid-ratio`), together with their confidence tiers.

## 3. Analyzing new outputs

To analyze new outputs with the released code, work on a copy of the archive. Place the new files in `results/data/` under the released file names, and the new trial table, written by `build_trial_table.py`, as `results/data/main_audit/neurips_dedup_trials_FULL.csv`. Remove `reanalysis/outputs/frozen_inputs.json` from the copy, so that `reanalysis/audit.py` records the hashes of the new inputs, and run the commands of Section 1.

## Released records

The main audit is released as its de-duplicated trial table: 29,778 rows over five models and the conditions A0, A1 and A3, with parsed Pass-1 and Pass-2 answers, ground truth, domain, confidence tier, correctness, reversal and, for the two open-weight models, Yes/No log-odds. The complete generation log of the Claude-Sonnet-4 run is included in `data-repair/sources/`. The mitigation and low-prestige subsets are released as their summary tables in `results/data/main_audit/`. In the mitigation summary, the Claude-Sonnet-4 rows cover the 306 items of its run with valid outputs in both branches.

The records of the control experiments keep every request outcome, including invalid outputs, with its status:

| File in `results/data/` | Contents |
|---|---|
| `controls.jsonl`, `controls_t0.jsonl` | Control suite for GPT-4o-mini, Claude-Sonnet-4 and Gemini-2.5-Flash: 500 items at T=0.7 and 250 items at T=0 |
| `controls_sota.jsonl` | Control suite for the four newer models on 250 items |
| `controls_open.jsonl` | Control suite for Llama-3.1-8B-Instruct and Llama-3.3-70B-Instruct on 500 items |
| `*_rat_fixed.jsonl` | Rationale conditions with arguments that state the assigned answer, used in place of the corresponding rows of each control file |
| `gradient_main.jsonl`, `gradient_sota.jsonl` | Fallibility and track-record conditions |
| `clean_rationale_250_dedup.jsonl` | No-padding rationale experiment for the three proprietary models on 249 items |
| `clean_rationale_sota_20260926.jsonl` and `.manifest.json` | No-padding rationale experiment for the four newer models, run on September 26, 2026, with requests, complete responses, charged costs, timestamps and endpoint metadata |
| `temporal_orig_pipeline.jsonl` | Rerun of the main-audit pipeline for GPT-4o-mini on 224 Medicine items |
| `control_items.jsonl`, `opponent_rationales.json` | The 500-item control pool and the opponent arguments |

`controls_open.jsonl` also contains records from a qwen/qwen3.6-27b endpoint that returned mostly empty responses; that endpoint is excluded from all reported results. The run and capture manifests, `clean_rationale_sota_20260926.manifest.json` and `data-repair/sources/manifest.json`, identify each input by its path in this archive or by its origin, together with its SHA-256.

## Interpretation notes

The padded suites of `run_controls.py` append active filler instructions to approximate a common prompt length. Changing the rationale, gate or source block can change the filler, so the rationale and gate contrasts of these suites compare prompt bundles. The no-padding suite removes the filler and reuses the same argument in the anonymous and expert branches; its expert prompt remains one sentence longer. Rationale compliance checks only that an argument states its assigned answer, not that each of its sentences is true, and the `rationale_quality` values `valid` and `misleading` mark arguments for the benchmark answer and for the other answer.

Each interaction is computed on the items valid in all four of its conditions, so per-condition rates, paired two-condition rates and interactions can have different denominators. Missing outputs need not be random, which matters most for the newer endpoints.

In the September run, temperature 0.7 and top-p 1 were sent only where the endpoint lists them as supported. GPT-5.6-terra received neither and Claude-Opus-4.8 received no top-p, so those settings are provider defaults. All calls requested at most 600 tokens without a reasoning-effort setting. A response is kept when its leading Yes or No is parseable, including length-stopped responses. This run reuses the July Pass-1 answers, so its accuracy changes span two collection dates.

Per-model exact McNemar tests are exploratory and not adjusted for multiplicity. Pooled intervals resample questions and keep the responses of all models to a question together. The four September models are pooled only with each other (`results/analysis/pooled_cluster_september.csv`, seed 20261104). Records collected in January, July and September 2026 are analyzed separately and never pooled.

## Assets and license

The code is released under the MIT License (`LICENSE`). The license does not cover third-party data, model weights or provider terms. PubMedQA is distributed under the MIT License, SciFact under CC BY-NC 2.0 according to its dataset card, and ContractNLI under CC BY 4.0; `data-repair/sources/` includes the ContractNLI license and terms. The original dataset papers are cited in the paper. No model weights are distributed. Generated text is research output of the named API services and does not imply endorsement by their providers.
