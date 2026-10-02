# Saved-output reproduction artifact

When Expert Disagreement Hurts: Auditing Prestige-Sensitive Revision in LLM Decision Pipelines

Authors: Yupeng Tang and Mingfeng Lin, Georgia Institute of Technology.

This archive accompanies the camera-ready paper. It supports numerical verification from saved outputs without credentials, GPU access, internet access at analysis time, or paid API calls. All original benchmark inputs are now recovered and independently verifiable in `data-repair/`; obtaining new model outputs still requires model access and a generation environment.

Original experiment code: https://github.com/yupengtang/when-expert-disagreement-hurts (public access verified September 26, 2026). This supplementary archive separately supplies the camera-ready saved-output analyses and follow-up records; the repository link does not replace it.

## Run

With Python 3.10 or later and the dependencies installed:

```bash
python -m pip install -r requirements.txt
python reproduce.py
python make_figures.py
```

`reproduce.py` regenerates CSVs in `rebuttal/analysis/` and LaTeX tables in `tables/`. `make_figures.py` recreates the numerical main-paper and appendix figures in `figs/`, embedding TrueType fonts. Analysis takes minutes on CPU. Dependencies are pinned to the versions used for this release. No generation script is executed by these commands.

The two main result figures also export SVGs with selectable text. To regenerate the project page's figures and interactive data from the same numerical source, run `python make_figures.py --web-output ../docs/assets` in a repository checkout. The reversal-direction plot uses all valid expert-labeled trials for both directions, and its net accuracy change is checked against their difference. The effect plot preserves the paired confidence intervals and separates the two prompt protocols. Changing the display does not recompute or pool the reported estimates.

## Included evidence and exclusions

- Complete de-duplicated main table: `rebuttal/data/recovered/neurips_dedup_trials_FULL.csv`, 29,778 rows over five models, with parsed answers, ground truth, condition, domain and available log-odds. Assertions verify that reversal and accuracy flags agree with the stored answers.
- Complete historical inputs: `data-repair/outputs/original_benchmark_recovered.jsonl` contains all 1,989 question–excerpt pairs, with verification levels and source provenance. Both original generation hashes match for 1,987 items; the two remaining Medicine inputs are verified against frozen/selection records and source data. The older `beat_full_questions_RECOVERED.jsonl` is retained only as historical recovery evidence: its excerpt-only join assigned incorrect questions to 96 Science entries. Do not use it for generation. Original saved trial outputs and the reported numerical results are unaffected.
- Evidence-complete inputs: `data-repair/outputs/benchmark_evidence_complete_v2.jsonl` repairs 48 original Law excerpts; `controls_evidence_complete_v2.jsonl` repairs 16 of the 201 Law excerpts in the 500-item control pool. All annotated nonempty spans are preserved and checked against original source offsets. These separately named inputs require fresh model calls and new rationale stimuli for changed excerpts; no results on them are claimed here.
- Historical follow-up pool: 500 inputs plus generated opponent arguments, preserved unchanged for the saved-output analyses. Their evidence-coverage limitation is documented above, not silently repaired underneath existing results.
- Legacy controls at T=.7, newer-model controls, open-weight endpoint controls, T=0 API controls, repaired rationale conditions, source-reliability controls, and the original-pipeline temporal replication. Invalid statuses are retained. Repaired rationale rows supersede originals; residual direction-noncompliant stimuli are excluded by the analysis.
- Clean no-padding rationale data: 249 items per model, three models and three conditions. `clean_rationale_250_dedup.jsonl` retains the earliest record for each model-item-condition. This removes 1,191 redundant successful rows from the un-deduplicated original file; those duplicates are not independent observations.
- `clean_rationale_sota_20260926.jsonl` adds 2,937 calls on September 26, 2026: three no-padding conditions on 249 GPT-5.6-terra, 249 Grok-4.5, 233 Gemini-3.6-Flash, and 248 Claude-Opus-4.8 eligible items. These reuse July Pass-1 answers and saved rationales; the three new branches are contemporaneous. No automatic retries were made. Raw requests, complete responses, usage, timestamps, invalid statuses, and endpoint metadata are retained. `new_clean_run_audit.json` gives exact cost and retention; `clean_retention.tex` reports parseability and length-stop counts separately.
- Original low-prestige and mitigation summary CSVs are preserved in `recovered/`. These support direct checks of the original aggregate results; their full raw generation logs and the model-domain breakdown are not re-created by this package.
- `source_manifest.json` records the original workspace-relative locations and SHA-256 of each selected input/source file. This does not publish or identify an independently hosted repository.

## Interpretation details

The legacy suite appends active filler instructions to approximate prompt length. Changing rationale/gate/source blocks can change filler, so those comparisons describe prompt bundles. The clean suite removes filler and reuses the same rationale across anonymous/expert branches. Its source-label contrast still changes the description length. Rationale compliance checks the initial assigned answer, not the truth of every sentence or an independently validated expert argument. Legacy `rationale_quality` values such as `valid` and `misleading` mean benchmark-matching and benchmark-conflicting target answers only.

Each interaction uses a common four-condition item intersection. Per-condition marginal rates, two-condition paired rates, and four-condition interactions can have different denominators, especially for Gemini-3.6. Missing outputs may be nonrandom. Qwen's mostly empty-response endpoint is excluded from behavioral claims.

The newer-model detail figure (now in the appendix) is generated by `newer_models.py` from the saved July four-model suite. `rebuttal/analysis/newer_models_figure.csv` contains exact point estimates, panel-specific sample sizes, seeds, and 95% percentile intervals from 50,000 paired question resamples. Prestige compares A3 with A1 on their shared intersection; accuracy compares Pass 2 with Pass 1 on valid A3 rows.

`core_display.py` adds all four newer models directly to main Figures 2–3 and Tables 2–4, separated without informal collection-period headings. `core_nine_models.csv` supplies transition shares, conditional rates, paired reversal rates, effect intervals, and retained N. Newer risk-ratio and odds-ratio intervals use 50,000 paired resamples; undefined 0/0 draws are omitted with counts recorded, while positive-over-zero ratios remain infinite. The January and July records are not pooled. Table 5 now reports seven models in the no-padding identical-rationale experiment, including the September calls. `clean_display.py` generates its rows, paired bootstrap intervals, and the corresponding appendix figure. The original three-model pooled estimate is preserved separately rather than mixing collection periods.

For the September calls, temperature=.7 and top_p=1 are sent only when listed as supported. GPT-5.6-terra omits both, and Claude-Opus-4.8 omits top_p; omitted settings use provider defaults. All calls request max_tokens=600, with no reasoning override. Parseable leading Yes/No answers are retained even if the explanation is length-stopped. Cross-date accuracy changes against stored Pass 1 can include endpoint drift and are not same-session accuracy effects. No cross-generation ranking is claimed.

Per-model exact McNemar tests are exploratory and unadjusted. Relative-effect and interaction CIs use 50,000 paired item-bootstrap resamples with seeds in the scripts. The pooled clean-rationale CI resamples 249 question blocks while retaining all three model responses in each block, rather than treating 747 responses as independent questions. The historical `clean_paired_contrasts.csv` includes naive pooled exact tests for traceability; the paper uses per-model tests and the question-cluster CI for inference.

`prompt_source/` preserves the actual source blocks, padding algorithms, rationale generation and repair code, and the budgeted September runner. It is provided to document stimuli, not as a self-contained API generation application. No credentials are supplied and importing these scripts is not required for reproduction. Identifiers and requested parameters are stored in the outputs; January/July/September 2026 are recorded run periods, not immutable provider snapshots.

The camera-ready does not rely on the original manuscript's unreported GLMM fit or on a full-benchmark open-weight T=0 table for which complete supporting records could not be located. Its T=0 section uses the saved July API data instead.

## Asset terms

The existing software license is retained verbatim in `LICENSE`, including its original copyright notice. That license does not relicense third-party data, model weights, or provider terms. PubMedQA's repository uses MIT terms; SciFact's dataset card lists CC BY-NC 2.0; ContractNLI uses CC BY 4.0. Attribute the original dataset papers cited in the accompanying paper. No model weights are distributed. Follow-up generated text is research output from the named API services and does not imply endorsement by the providers.
