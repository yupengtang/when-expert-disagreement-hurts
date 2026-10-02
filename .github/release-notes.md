This release accompanies the accepted NeurIPS 2026 paper **“When Expert Disagreement Hurts: Auditing Prestige-Sensitive Revision in LLM Decision Pipelines.”**

It contains:

- the NeurIPS camera-ready PDF (`neurips-2026-camera-ready.pdf`);
- the public preprint-style PDF (`paper.pdf` and `arxiv-preprint.pdf`);
- portable LaTeX source packages for both versions;
- both vector architecture figures and their two-slide editable PowerPoint source; and
- the saved-output reproducibility package used for the camera-ready analyses.

The reproducibility package regenerates the reported numerical analyses, tables, and figures without API credentials, paid model calls, or GPU access. The same files are browsable under [`reproducibility/`](https://github.com/yupengtang/when-expert-disagreement-hurts/tree/main/reproducibility), and the repository CI reruns the analysis.

The November 4 revision adds a selectivity-versus-break-even figure and revises the text and table placement. The PDFs contain 10 content pages and 39 pages overall. Figure 3 (page 7) shows selectivity; Figure 4 (page 8) separates reversal and accuracy effects. The two architecture diagrams remain on pages 2 and 14. `audit-figures-editable.pptx` contains their editable sources; the earlier single-diagram PowerPoint is retained as a historical source.

The analyses use historical saved outputs, not new model calls. `reproducibility/reanalysis/` contains compatible-pair accuracy effects, evidence-coverage and document-cluster sensitivity, and source-following agreement contrasts. `reproducibility/selectivity_display.py` generates the new figure from the released conditional reversal rates.

The later November 4 update adds the pooled identical-rationale effect for the four September models: +2.0 percentage points (95% question-cluster bootstrap CI [0.7, 3.3]). The estimate uses 953 pairs from 249 questions; individual-model exact tests remain nonsignificant. `reproduce.py` regenerates `rebuttal/analysis/pooled_cluster_september.csv` from the saved responses.

All 1,989 original question–excerpt pairs are now recovered. The included input audit verifies the recovered inputs and corrects 96 Science question associations in an older reconstructed input file; original saved trial outputs are unchanged. Separately versioned datasets repair incomplete evidence coverage in 48 original Law excerpts and 16 control Law excerpts. Those repaired inputs require fresh evaluation and must not be paired with historical model outputs. See [`data-repair/`](https://github.com/yupengtang/when-expert-disagreement-hurts/tree/main/reproducibility/data-repair) for the data, provenance, per-item changes and offline regression tests.
