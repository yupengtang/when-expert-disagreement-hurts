# When Expert Disagreement Hurts

**Auditing Prestige-Sensitive Revision in LLM Decision Pipelines**<br>
Yupeng Tang and Mingfeng Lin · NeurIPS 2026

[Project page](https://yupengtang.github.io/when-expert-disagreement-hurts/) · [Paper](https://yupengtang.github.io/when-expert-disagreement-hurts/assets/paper.pdf) · [Release](https://github.com/yupengtang/when-expert-disagreement-hurts/releases/latest)

Language models often revise an answer after another agent disagrees. We use a branch-paired audit to separate ordinary reconsideration, sensitivity to anonymous disagreement, and the incremental effect of labeling the same disagreeing source as an expert. The question, evidence, and initial answer are held fixed across branches.

## Findings

- On the full 1,989-item benchmark, disagreement induces substantially more revision than neutral reconsideration.
- Revision improves accuracy only when its selectivity for initial errors exceeds the break-even ratio implied by initial accuracy. All five main-benchmark models fall short under expert disagreement; most API models would need to be three to five times more selective.
- Expert labels have heterogeneous effects: four of five full-benchmark models show an additional 2.6–6.9 percentage-point reversal effect, while one shows a negative aggregate effect.
- More reversal does not necessarily imply lower accuracy. On compatible anonymous/expert pairs, expert-label accuracy effects range from −5.2 to +2.4 points across the five main models; beneficial and harmful transitions are reported separately.
- In the identical-rationale control, the four September models have a pooled source-label effect of +2.0 points (95% question-cluster bootstrap CI [0.7, 3.3]), although none of their individual exact tests is significant. They are pooled within their protocol, not ranked against the main five models.

## Reproduce the results

The saved-output analysis runs on CPU and does not require API credentials:

```bash
git clone https://github.com/yupengtang/when-expert-disagreement-hurts.git
cd when-expert-disagreement-hurts/reproducibility
python -m pip install -r requirements.txt
python reproduce.py
python make_figures.py
python reanalysis/audit.py
python reanalysis/render.py
```

These commands regenerate the historical analyses and the revised paired reversal and accuracy estimates, sensitivity checks, tables, and figures from released records. The repository [CI workflow](.github/workflows/reproduce.yml) runs both analyses and checks the tracked numerical outputs for changes. See the [analysis documentation](reproducibility/README.md) for the retrospective rules and the distinction between historical and revised estimates.

All 1,989 original question–excerpt pairs are now recovered. The [input audit](reproducibility/data-repair/README.md) verifies their provenance and provides separately versioned evidence-complete inputs: 48 original Law excerpts and 16 control Law excerpts required expansion. Historical results are preserved; repaired inputs require fresh model calls and must not be paired with old outputs. The same audit corrects 96 Science question associations in an earlier reconstructed input file, without changing the original saved trial outputs.

The experiment loader accepts all three released input files, preserves their provenance, and rejects changed content with inconsistent checksums. Its offline smoke tests run with `python -m unittest discover -s experiment/tests -p 'test_*.py' -v` from the repository root. This check loads inputs only; it does not query a model.

## Repository structure

| Path | Contents |
|---|---|
| [`reproducibility/`](reproducibility/) | Saved outputs, analysis code, generated tables, and figure data used in the paper |
| [`experiment/`](experiment/) | Original experiment pipeline and frozen study materials |
| [`docs/`](docs/) | Project-page source and paper figures |

The [camera-ready release](https://github.com/yupengtang/when-expert-disagreement-hurts/releases/latest) includes the final paper, public preprint, LaTeX source archives, [both editable architecture figures](https://github.com/yupengtang/when-expert-disagreement-hurts/releases/download/neurips-2026-camera-ready/audit-figures-editable.pptx), and a standalone reproduction package.

## Citation

```bibtex
@inproceedings{tang2026expert,
  title     = {When Expert Disagreement Hurts: Auditing Prestige-Sensitive Revision in LLM Decision Pipelines},
  author    = {Tang, Yupeng and Lin, Mingfeng},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

The analysis code is released under the [MIT License](LICENSE). Dataset excerpts and model outputs remain subject to their original source and provider terms; no model weights are distributed. For questions, open an issue or contact Yupeng Tang at `yupengtang@gatech.edu`.
