# Explaining a Multi-Layer Perceptron with Shapley Values

**Paper, slides and documentation of the seminar talk „Shapley-Analyse eines MLP“** — Bergische Universität
Wuppertal, 1 December 2026.

> **Central question:** How can the predictions of an MLP be explained with Shapley values — and where are the
> limits of these explanations?

📄 **[Read the paper (PDF)](paper/paper.pdf)** · [Markdown chapters](paper/) · [Slides (German)](slides/)

## Abstract

Shapley values distribute the output of a model among its input features in the only way that satisfies a
short list of fairness axioms. The paper explains, step by step and with exercises, how they are defined, why a
neural network forces a choice of value function and background data, and how KernelSHAP and DeepSHAP
approximate them. The theory is applied to a small MLP (7 → 16 → 8 → 1, 273 parameters) that predicts whether
the next day reaches 30 °C at five German weather stations. With seven features, exact Shapley values (all 128
coalitions) are cheap and serve as ground truth: default KernelSHAP reproduces them to 10⁻⁶, DeepSHAP keeps the
correct sum but deviates by 6.5 % of the mean attribution. Runtime is governed by the number of features rather
than by "kernel or deep", the background data changes attributions by more than a factor of two, and on the
embedding the network signals heat by switching off neurons that represent normal weather.

## Contents of the paper

| Part | Sections |
|---|---|
| **I — Theory** (with 19 exercises) | [MLPs, logits, embeddings](paper/02_mlp.md) · [Shapley value and axioms](paper/03_shapley.md) · [Value function and background](paper/04_value_function.md) · [Exact, KernelSHAP, DeepSHAP](paper/05_approximations.md) · [Logits vs. embeddings, runtime model](paper/06_embeddings_runtime.md) |
| **II — Case study** | [Data and model](paper/07_case_data_model.md) · [Shapley results: logit, embeddings, runtime, background](paper/08_case_shapley.md) |
| **III — Discussion** | [Limitations, misconceptions, conclusion, references](paper/09_discussion.md) |
| **Appendices** | [Solutions](paper/10_appendix_solutions.md) · [Project documentation, formula summary, glossary, self-test](paper/11_appendix_project.md) |

Start with the [introduction](paper/01_introduction.md) and [how to read this paper](paper/00_abstract.md).

## Repository structure

| Path | Content |
|---|---|
| `paper/` | the paper as Markdown chapters, its figures, `build.sh` and the built `paper.pdf` |
| `slides/` | the deck of the talk (`.pptx`, `.pdf`) and its speaker notes |
| `notes/meetings/` | kick-off note with the original requirements |

## Related repositories

| Repository | Role |
|---|---|
| [heatrisk-shap-analysis](https://github.com/amk0098/heatrisk-shap-analysis) | all code, tests and results of the case study; experiment log; tag `v1.0-talk` |
| [heatrisk-deutschland](https://github.com/amk0098/heatrisk-deutschland) | source project: DWD data pipeline and baselines |
| [mlp-shap-pipeline](https://github.com/amk0098/mlp-shap-pipeline) | follow-up: the same analysis automated for any MLP (design phase) |

## Building the PDF

Requires pandoc ≥ 3 and a TeX distribution with `xelatex` and the DejaVu fonts.

```bash
paper/build.sh
```

## Licence

MIT (see `LICENSE`). Weather data in the figures: Deutscher Wetterdienst, open data („Quelle: Deutscher
Wetterdienst“).
