# 1 Introduction

A multi-layer perceptron (MLP) computes its prediction as a nested, non-linear function of all its inputs.
Even a network with a few hundred weights gives no direct answer to the question *why* it predicted what it
did for one particular input: no single weight "belongs" to a feature, and the effect of a feature depends on
the values of all the others. Methods of explainable AI try to answer this question after training.

One family of such methods borrows an idea from cooperative game theory. Shapley [Sha53] asked how a group of
players should share the payoff they earn together, and showed that exactly one rule satisfies a small set of
fairness requirements. In machine learning the players are the input features and the payoff is the model's
prediction for one input, measured relative to an average prediction [ŠK14, LL17]. The result is a *local
attribution*: one number per feature that says how much this feature moved this prediction up or down. Molnar
[Mol22] gives an accessible overview of these and related methods.

This paper answers one question:

> **How can the predictions of an MLP be explained with Shapley values — and where are the limits of these
> explanations?**

It was written to accompany a 30-minute seminar talk at the Bergische Universität Wuppertal. The supervisor's
requirements for the talk (Appendix B.1) determine its structure: train an MLP on a dataset of one's choice,
explain its architecture and the dataset, explain the mathematics of Shapley values, and apply Shapley values
to the **logit** and to the **embeddings** of the network with **KernelSHAP** and **DeepSHAP**, including
the question of runtime.

## 1.1 Contributions

1. A self-contained, exercise-based introduction to Shapley values for neural networks (Part I): cooperative
   games, the four axioms, the value function and the background data, the cost of exact computation,
   KernelSHAP, DeepLIFT/DeepSHAP, the two possible meanings of "Shapley values on the embeddings", and a cost
   model for the runtime.
2. A case study (Part II) in which **exact** Shapley values are available as ground truth, because the model
   has only seven features. This makes it possible to *measure* — not just state — how far KernelSHAP and
   DeepSHAP are from the values they approximate.
3. Four findings that correct common simplifications: KernelSHAP with default settings is exact for seven
   features; DeepSHAP keeps the sum but not the split; runtime is driven by the number of features rather than
   by "kernel versus deep"; and the background data changes the attributions more than the choice of method.
4. A documentation of the whole project (Appendix B), so that every figure in the talk and every number in this
   paper can be traced to a script, a commit and a run.

## 1.2 Running example

Throughout the paper the same small network is used. It receives seven weather measurements of one day at one
German weather station and outputs a score for "tomorrow will be a heat day" (maximum temperature ≥ 30 °C).
Its architecture is 7 → 16 → 8 → 1 with ReLU activations (Section 2.3). Seven features are few enough for exact
Shapley values (2⁷ = 128 coalitions) and enough to show the typical problems of real data: strongly
correlated inputs and a rare positive class.

## 1.3 Notation

| Symbol | Meaning | In the case study |
|---|---|---|
| $M$ | number of players (features) | 7 inputs; 8 when the embedding neurons are the players |
| $N$ | number of background rows | 100 training days |
| $B$ | number of explained instances | up to 3650 test days |
| $x$ | the instance being explained | one standardised day (7 numbers) |
| $b$ | one background row | one training day |
| $S$ | a coalition, i.e. a subset of the features | e.g. {Max.-Temp., rel. humidity} |
| $v(S)$ | value of coalition $S$ | expected logit when only the features in $S$ are known |
| $\varphi_i$ | Shapley value of feature $i$ | contribution of feature $i$ to this logit |
| $\varphi_0$ | base value | mean logit over the background |
| $f$ | the function being explained | logit of the MLP, or one embedding neuron |

Shapley values are always given in the unit of the explained output; for the logit these are log-odds.
