# Project Instructions — Shapley Analysis of an MLP

Academic seminar project. Explain the mathematics, implement a reproducible experiment, and critically interpret feature attributions.

## Research question
How can Shapley values explain MLP predictions, what assumptions do explanations rely on, and what are their limitations?

## Topics
- MLP: layers, weights, biases, activations, forward pass, loss, gradient descent, backpropagation, binary classification, generalisation and overfitting.
- Cooperative game theory: players, coalitions, value functions, marginal contributions; efficiency, symmetry, dummy and additivity axioms.
- SHAP: local/global explanations; model-agnostic/model-specific methods; KernelSHAP, DeepExplainer and GradientExplainer.
- Background data, correlated features, output scale, approximation and runtime.

## Shapley formula
For feature i among M features: phi_i = sum over S subset N without i of [|S|!(M-|S|-1)!/M!] times [v(S union {i})-v(S)]. Explain every term intuitively. The model output and missing-feature/reference mechanism define v(S).

## Experiment requirements
Use a small tabular dataset and reproducible MLP. Define target/features; use train/validation/test splits; fit preprocessing only on training data; record seeds and package versions; specify probability or raw output; select background data deliberately; verify explainer compatibility; include global and two local explanations; check additivity and sensitivity where feasible.

HeatRisk-specific implementation belongs in the separate heatrisk-shap-analysis repository.

## Critical interpretation
SHAP depends on the fitted model, output definition, explainer, feature representation and background data. It is not causal inference. Discuss correlated inputs, baseline dependence, approximation error, model quality and stability.

## AI workflow
Inspect before coding; state objectives and changed files; explain concepts; avoid unnecessary complexity. Report implementation, rationale, run/verification instructions and limitations. Never invent citations, results, APIs or supervisor requirements.

## First task
Prepare a conceptual map, formula explanation, explainer comparison, experiment proposal, seminar/literature plan and supervisor questions before substantial coding.