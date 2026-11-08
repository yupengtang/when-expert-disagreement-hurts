# Benchmark inputs: reconstruction, verification and evidence-complete versions

This directory reconstructs the 1,989 question-excerpt pairs used in every main-audit run from the official dataset releases and verifies them against the hashes recorded at generation time. It also exports evidence-complete versions of the Law inputs for new experiments. It never modifies a released model output.

## Files

| File in `outputs/` | Contents |
|---|---|
| `original_benchmark_recovered.jsonl` | The experiment inputs: 890 Medicine, 299 Science and 800 Law items, each with its verification level and source provenance. This is the input file of `experiment/run_experiment.py`. |
| `benchmark_evidence_complete_v2.jsonl` | The same questions and labels, with 48 Law excerpts widened to contain every nonempty annotated evidence span. For new experiments only. |
| `controls_evidence_complete_v2.jsonl` | The 500 items of the control pool, with 16 Law excerpts widened in the same way; the other 484 question-excerpt pairs are unchanged. For new experiments only. |
| `audit_report.json` | Coverage, verification levels, output checksums and change counts |
| `benchmark_evidence_changes.json`, `control_evidence_changes.json` | Per-item changes: old and new hashes, missing evidence spans and lengths |
| `previous_recovery_prompt_corrections.json` | The 96 Science questions that share an abstract with another question; for each, the question an excerpt-only key would assign and the verified question |

The evidence-complete inputs use a separate identifier namespace. A changed excerpt invalidates earlier model responses and generated arguments for that item, so experiments on these inputs need a new Pass 1, new revision branches and new arguments for the changed items. Confidence tiers are cleared for changed excerpts. The paper reports no results on these inputs.

## Offline reconstruction

Python 3.9 or later suffices. No third-party package, account, network connection, GPU or API credit is needed.

```bash
python repair_inputs.py
python -m unittest discover -s . -p 'test_*.py' -v
```

The script verifies the checksums of all files in `sources/` before running. Existing output files must be byte-identical to the regenerated ones, and a differing file is never overwritten. To regenerate into a new directory:

```bash
python repair_inputs.py --out /path/to/new-output-directory
```

## Verification

For 1,987 items, both the excerpt hash and the combined question-excerpt hash match the generation log of the Claude-Sonnet-4 run. Item q0405 is verified directly against its frozen input record, and the de-duplicated trial table supplies its label and confidence tier. Item q0828 matches the full question hash and the excerpt hash in the selection log and the label in the trial table; its identifier follows the selection order, which is cross-checked against all 1,371 records of the stored selection-log prefix. All 552 complete frozen input records match the reconstructed question, excerpt and label, and all 29,778 rows of the trial table agree on question identifier, domain, label and confidence tier.

The selection log stores full questions but only excerpt prefixes. Full excerpts are therefore taken from the public dataset releases and validated with the saved hashes. Because 96 Science questions share an abstract with another question, items are keyed by question as well as by excerpt.

The Law inputs of the main audit were built by forming a 200-character context window around each annotated span, joining the windows with `\n\n...\n\n`, and truncating the result to 2,000 characters. The loader also stripped the document before applying its original character offsets. Reproducing this construction matches all 800 Law inputs on both hashes. The reconstruction keeps these bytes; evidence completion is a separate step.

## Evidence completion

For every Law item, the original source document and hypothesis identifier are used, the label is validated, and each annotated evidence span is checked separately; a test on the concatenation of all spans would not detect a missing span when the excerpt retains text between them. Whitespace-only annotations carry no content and are counted in the metadata.

An excerpt is kept if it contains every nonempty annotated span. Otherwise it is replaced by the contiguous source interval that covers all annotated spans, with 200 characters of context on each side where available and without a length cap. Offsets, document hashes and source identifiers are preserved. This establishes evidence-span coverage, not an independent expert validation of every label or argument.

The check finds 48 of the 800 Law items of the benchmark and 16 of the 201 Law items of the control pool with at least one missing span. Whether the answers to these items would change can only be determined with new model calls.

## Sources and terms

`sources/` holds the captured inputs of the reconstruction: the de-duplicated trial table, the complete generation log of the Claude-Sonnet-4 main-audit run, the stored prefix of the frozen input file, the selection log, an excerpt-keyed reconstruction used to identify the 96 shared-abstract questions, the control pool, the dataset normalizer of the benchmark build, and the official dataset releases. `sources/manifest.json` records the origin and the checksums of each captured file; a file that is also released elsewhere in this archive is identified by its path here. Only the pure conversion functions of the captured normalizer are used; its download functions are not called.

| Dataset | Source |
|---|---|
| PubMedQA | `qiaojin/PubMedQA`, `pqa_labeled/train` (1,000 records), exported from the cached Arrow dataset; https://github.com/pubmedqa/pubmedqa |
| SciFact | Official corpus and train/dev claims from https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz; https://github.com/allenai/scifact |
| ContractNLI | Official train/dev/test release, 607 documents, with its license and terms; https://github.com/stanfordnlp/contract-nli |

Third-party datasets retain their original terms, and the code license does not relicense them. No model weights or credentials are included.
