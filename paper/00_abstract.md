---
title: "Explaining a Multi-Layer Perceptron with Shapley Values"
subtitle: "Theory, approximations and a case study on next-day heat prediction"
author: "Arian Mohammadkhani"
date: "Bergische Universität Wuppertal · Seminar talk „Shapley-Analyse eines MLP“, 1 December 2026 · Written November 2026"
abstract: |
  Shapley values distribute the output of a model among its input features in the only way that satisfies a
  short list of fairness axioms. This paper explains, step by step and with exercises, how they are defined, why
  a neural network forces a choice of *value function* and *background data*, and how KernelSHAP and DeepSHAP
  approximate them. The theory is then applied to a small multi-layer perceptron (7 → 16 → 8 → 1, 273
  parameters) that predicts whether the next day reaches 30 °C at five German weather stations. Because the
  model has only seven features, exact Shapley values (all 128 coalitions) are cheap and serve as ground truth.
  With its default settings KernelSHAP reproduces them to 10⁻⁶; DeepSHAP keeps the correct sum but deviates by
  6.5 % of the mean attribution on average. Runtime is governed by the number of features, not by the choice
  "kernel or deep": for seven features the exact computation is the fastest method. The choice of background
  changes the size of the attributions by more than a factor of two for the same prediction. On the embedding
  (last hidden layer) the network signals heat mainly by switching *off* four neurons that are active on normal
  days. The paper also documents the seminar project — requirements, decisions, repositories and how the
  results were produced — so that every number can be traced and reproduced.
---

# How to read this paper {-}

**Part I (Sections 2–8)** is the theory, written for readers who know linear algebra and basic machine learning
but no game theory. Each concept follows the same pattern: intuition, a small worked example, the formula with
every symbol explained, and its meaning for the network of the case study. **Exercises** are placed where they
belong in the text; do them on paper before you read the solutions in Appendix A. Exercises marked *(code)*
need a computer.

**Part II (Sections 9–14)** is the case study: data, model, and all Shapley results with their numbers.
**Part III (Sections 15–16)** discusses limitations and concludes. **Appendix B** documents the seminar project
itself; **Appendix C** is a one-page summary of all formulas, **Appendix D** a self-test.

Readers in a hurry: Section 1, the boxed results at the start of each case-study section, and Section 15.

**Companion material.** The code and all raw results are in
[heatrisk-shap-analysis](https://github.com/amk0098/heatrisk-shap-analysis) (tag `v1.0-talk`); the German
slides of the talk are in [`slides/`](../slides/) of this repository. Figures in Part II are the figures of the
talk and therefore carry German labels; the feature table in Section 9.2 translates them.

**A note on tools.** Drafting, code review and the verification of library behaviour were assisted by an AI
assistant (Claude, Anthropic). Every number in this paper comes from a run recorded in the experiment log of the
code repository, every exercise was checked by computation, and every reference was looked up. The author is
responsible for the content; the assistant is not a source.
