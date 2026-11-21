# CLAUDE.md — Shapley-Analyse eines MLP (paper and slides)

Project context. Code and results: [heatrisk-shap-analysis](https://github.com/amk0098/heatrisk-shap-analysis)
(tag `v1.0-talk`).

## What this is
Paper, slides and documentation of the seminar talk „Shapley-Analyse eines MLP“ (Bergische Universität Wuppertal,
1 December 2026). Central question: how can the predictions of an MLP be explained with Shapley values, and where
are the limits of these explanations? The talk is in German; the paper is in English and is educational (theory
with exercises), a case study and the documentation of the project.

## Content
- `paper/`: Markdown chapters `00_…`–`11_…`, figures, `build.sh` (pandoc + xelatex) and `paper.pdf`.
  Part I theory (MLP, Shapley value, axioms, value function, exact/KernelSHAP/DeepSHAP, logits vs. embeddings,
  runtime model), Part II case study (7 → 16 → 8 → 1 MLP on DWD heat-day data), Part III discussion, appendices
  (solutions, project documentation, formula summary, self-test).
- `slides/`: deck (`.pptx` with speaker notes, `.pdf`).

## Rules
- Every number in the paper must come from a run in `docs/experiment_log.md` of heatrisk-shap-analysis; update
  both together. Rebuild `paper.pdf` after every change of the Markdown files.
- Never invent citations or results; every reference in the paper has been checked against its source.
- Exercises are solved in Appendix A; keep exercise numbers and solution numbers in sync.
- Shapley values are reported in logit units, always with the background used; SHAP is not causal inference.
- Talk and slides German; paper English. Chair rules for slides: font ≥ 18 pt, 3–5 points per slide, sources on
  the slide, page numbers.

## Literature (verified)
[Sha53] Shapley 1953 · [You85] Young 1985 · [ŠK14] Štrumbelj & Kononenko 2014 · [LL17] Lundberg & Lee 2017 ·
[SGK17] Shrikumar et al. 2017 · [SN20] Sundararajan & Najmi 2020 · [JMB20] Janzing et al. 2020 · [Kum20] Kumar
et al. 2020 · [AJL21] Aas et al. 2021 · [Mol22] Molnar 2022 · [GBC16] Goodfellow et al. 2016 · [Pas19] PyTorch.
Full entries: `paper/09_discussion.md` (References).
