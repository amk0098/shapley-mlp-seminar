# 6 Computing Shapley values: exact, KernelSHAP, DeepSHAP

## 6.1 The cost of exact computation

Each feature is present or absent, so there are $2^M$ coalitions, and each coalition value $v_x(S)$ needs $N$
forward passes. One more feature doubles the work.

| $M$ | coalitions $2^M$ | model rows per instance ($N = 100$) |
|---:|---:|---:|
| 3 | 8 | 800 |
| 7 (case-study inputs) | 128 | 12 800 |
| 16 | 65 536 | 6.6 million |
| 30 | ≈ 1.07 · 10⁹ | ≈ 1.07 · 10¹¹ |

> **Exercise 11.** The Shapley formula has $M \cdot 2^{M-1}$ terms, and each needs two values,
> $v(S \cup \{i\})$ and $v(S)$. For $M = 7$: (a) how many evaluations of $v$ is that without caching? (b) How
> many *different* coalitions exist? (c) What follows for an implementation?

The case-study implementation evaluates each coalition once: all 128 coalitions are encoded as bit masks, the
128 values $v_x(S)$ are computed in one vectorised pass over the background, and the Shapley values are then
weighted differences of these 128 numbers. For seven features exact computation is cheap, and it serves as the
ground truth for both approximations. They exist for models where $2^M$ is out of reach:

| Idea | Method | What is saved |
|---|---|---|
| Do not visit all coalitions; sample some | permutation sampling [ŠK14], KernelSHAP [LL17] | the factor $2^M$ |
| Do not query the model per coalition at all; propagate contributions backwards through the layers | DeepLIFT [SGK17] → DeepSHAP [LL17] | the factor $2^M$ entirely; cost independent of $M$ |

## 6.2 KernelSHAP

**Intuition.** Think of each feature as a switch: on = present, off = absent. KernelSHAP evaluates the model
for a number of switch patterns (coalitions), records $v(S)$ for each, and fits the simplest possible model to
these records — a sum "base value plus one number per switch that is on". With one particular weighting of the
patterns, the *Shapley kernel*, the fitted numbers are **exactly** the Shapley values; with fewer patterns than
$2^M$ they are an estimate.

**Formula.** With $z'_S \in \{0,1\}^M$ the switch vector of coalition $S$ and the additive explanation model
$g(z') = \varphi_0 + \sum_i \varphi_i z'_i$, KernelSHAP solves the weighted least-squares problem

$$
\min_{\varphi}\; \sum_{S} \pi(|S|)\,\bigl[v(S) - g(z'_S)\bigr]^2, \qquad
\pi(s) = \frac{M-1}{\binom{M}{s}\, s\,(M-s)} \;(s = 1, \dots, M-1), \qquad \pi(0) = \pi(M) = \infty .
$$

The infinite weights for the empty and the full coalition force the fit through these two points, which gives
the two constraints $\varphi_0 = v(\emptyset)$ and $\sum_i \varphi_i = v(\mathcal{N}) - v(\emptyset)$:
efficiency is built in. That this regression returns the Shapley values is Theorem 2 of Lundberg and Lee
[LL17, Sec. 4.1].

**Two weights, two roles.** The *Shapley weight* $|S|!(M-|S|-1)!/M!$ weights a marginal contribution (with
$i \notin S$). The *Shapley kernel* $\pi$ weights a whole coalition in a regression. Different formulas,
different jobs, same result.

> **Exercise 12 (kernel weights).** Compute $\pi(1), \pi(2)$ for $M = 3$ and $\pi(1), \pi(2), \pi(3)$ for
> $M = 4$. Which coalition sizes get the highest weight, and why?

> **Exercise 13 (the regression by hand).** Use the running game of Section 3. The constraints give
> $\varphi_0 = 0$ and $\varphi_A + \varphi_B + \varphi_C = 18$; the six remaining coalitions all have the same
> weight (Exercise 12), so the plain sum of squared errors is minimised. Hint: for a pair such as {A, B} the
> model predicts $\varphi_A + \varphi_B = 18 - \varphi_C$, so each $\varphi_k$ appears in exactly two squared
> terms. Minimise and enforce the sum.

**What the library does.** The following facts were read from the source code of `shap` 0.54.1, the version
used in the case study:

- The number of coalitions per instance is `nsamples`; by default $2M + 2048$. If this exceeds the number of
  non-trivial coalitions $2^M - 2$, it is capped there and **all** coalitions are enumerated. For $M = 7$:
  $2 \cdot 7 + 2048 = 2062 > 126$, so the default KernelSHAP does not sample at all — it computes the exact
  Shapley values of the interventional game. Sampling only starts at $M = 12$ ($2^{12} - 2 = 4094 > 2072$).
- Each coalition costs $N$ model rows; all rows of one instance are passed to the model in one call; instances
  are explained one after another.
- Efficiency holds for every `nsamples`, because it is a constraint, not an estimate.
- The default `l1_reg="num_features(10)"` first selects at most ten features and sets the others to exactly
  zero. For seven inputs or eight embedding neurons this is harmless, for larger $M$ it silently changes the
  result; `l1_reg=False` gives the plain Kernel-SHAP regression and was used for all comparisons in Part II.

To *show* the approximation error one therefore has to reduce `nsamples` deliberately (Section 11.3).

## 6.3 DeepLIFT and DeepSHAP

**Intuition.** KernelSHAP queries the model from outside. DeepLIFT [SGK17] opens it. It runs the network on
the instance $x$ and on a **reference** input $b$, records how much every neuron differs between the two runs,
and then walks backwards from the output, handing each neuron's difference down to the neurons that caused it in
proportion to their share. At the bottom every input holds its share of $f(x) - f(b)$. DeepSHAP [LL17, Sec.
4.2] does this with every background row as the reference and **averages** the results.

**Building blocks.** For any quantity $t$ in the network write $\Delta t = t(x) - t(b)$. A *multiplier*
$m(a \to t)$ is the contribution of $\Delta a$ to $\Delta t$, divided by $\Delta a$.

| Rule | Where | Multiplier |
|---|---|---|
| Linear rule | $z = \sum_i w_i a_i + \text{bias}$ | $m(a_i \to z) = w_i$ |
| Rescale rule | $h = \mathrm{ReLU}(z)$ (any one-input non-linearity) | $m(z \to h) = \Delta h / \Delta z$; the ordinary gradient if $\Delta z \approx 0$ |
| Chain rule | through layers | $m(x_i \to y) = \sum_j m(x_i \to z_j)\, m(z_j \to h_j)\, m(h_j \to y)$ |

The contribution of input $i$ for reference $b$ is $C_i = m(x_i \to y)\,(x_i - b_i)$, and
**summation-to-delta** holds: $\sum_i C_i = f(x) - f(b)$. Sketch: for one neuron $\Delta z = \sum_i w_i \Delta a_i$
exactly (the bias cancels) and $\Delta h = (\Delta h / \Delta z)\,\Delta z$ by definition, so the contributions
add up to $\Delta h$; this repeats layer by layer. DeepSHAP then sets

$$
\varphi_i \approx \frac1N \sum_{b} m_b(x_i \to f)\,(x_i - b_i).
$$

The rescale multiplier is the slope of the **secant** between reference and instance, not the slope of the
tangent at $x$: DeepLIFT is not "gradient times input".

> **Exercise 14 (one ReLU neuron).** $y = \mathrm{ReLU}(2x_1 + x_2 - 1)$, instance $x = (1, 1)$, reference
> $b = (0, 0)$. (a) Compute $z$ and $y$ for $x$ and $b$, and $\Delta z, \Delta y$. (b) Rescale multiplier,
> $m(x_1 \to y)$, $m(x_2 \to y)$ and the contributions $C_1, C_2$. (c) Check summation-to-delta. (d) Compute the
> exact Shapley values of the same game ($M = 2$, absent features taken from $b$). Are they equal to
> $C_1, C_2$?

> **Exercise 15 (averaging = DeepSHAP).** Same neuron and instance, second reference $b' = (1, 0)$. (a) Compute
> $C_1, C_2$ and the exact Shapley values for $b'$. (b) Average both over $\{b, b'\}$. (c) Check that both sum
> to $f(x) - \frac12(f(b) + f(b'))$. Why is DeepLIFT exact for $b'$ but not for $b$?

> **Exercise 16 (two layers).** $z_1 = x_1 + x_2 - 1$, $h_1 = \mathrm{ReLU}(z_1)$; $z_2 = x_1 - x_2$,
> $h_2 = \mathrm{ReLU}(z_2)$; $y = 2h_1 - h_2$. Instance $x = (2, 1)$, reference $b = (0, 0)$. Compute the
> forward passes, the two rescale multipliers, $m(x_1 \to y)$ and $m(x_2 \to y)$ via the chain rule, the
> contributions, and the exact Shapley values. Compare.

All three exercises were checked with `shap.DeepExplainer` on equivalent PyTorch modules; it returns exactly
the hand-computed DeepLIFT values.

**When is DeepSHAP exact?**

- For a **linear** model: all multipliers are weights, so $\varphi_i = w_i(x_i - \bar b_i)$ as in Section 5.
- For one reference $b$, if no ReLU changes its on/off state on any hybrid input between $b$ and $x$: the
  network is then one fixed linear function on all these points (reference $b'$ in Exercise 15).
- **Otherwise it is an approximation.** When a ReLU switches, DeepLIFT distributes its $\Delta h$ in proportion
  to the inputs' shares of $\Delta z$, whereas the Shapley value depends on *which coalitions* switch the
  neuron on. Exercise 16 is a pure interaction: the exact values split equally, DeepLIFT gives most of it to
  one input.
- **Always**: $\sum_i \varphi_i = f(x) - \varphi_0$, by summation-to-delta and averaging. The library checks
  this by default.

Efficiency is therefore a useful **bug check**, not a proof of quality: in Exercise 16 the sum is perfect and
the split is not the Shapley split.

**Practical requirements** (from the `shap` source): activations must be `nn.ReLU()` modules (a functional
`torch.relu` gets no hook and is treated with the plain gradient), one module object per place, a model output
of shape $(B, 1)$; `DeepExplainer((model, layer), background)` explains the *input of that layer* (needed in
Section 7). scikit-learn's `MLPClassifier` is not supported, which is why the case study uses PyTorch.

## 6.4 Which axioms survive the approximations?

Reference point: the exact Shapley values of the interventional game with the chosen background.

| Property | Exact | KernelSHAP | DeepSHAP |
|---|---|---|---|
| Efficiency | ✓ | ✓ for any `nsamples` (constraint) | ✓ (summation-to-delta, then averaging) |
| Symmetry | ✓ | ✓ if all coalitions are enumerated; only approximately when sampling | not guaranteed: the split depends on the path through the network |
| Dummy | ✓ | ✓ if enumerated; under sampling a feature without effect can get a small non-zero estimate | ✓ for a feature with $\Delta x_i = 0$ or without a path to the output |
| Additivity | ✓ | ✓ for fixed coalitions without feature selection (the estimate is linear in the model outputs) | ✓ across outputs joined by a linear layer (Section 7) |
| Equals the Shapley value | by definition | if all coalitions are used; otherwise an estimate that improves with `nsamples` | for linear models or when no ReLU switches; otherwise a different additive attribution |

Two caveats remain for all methods. Even the exact values are exact only **for the chosen background
sample**; another sample defines another game (Section 14). And all of this concerns faithfulness to the
interventional game — whether that game asks the right question for correlated features is the separate issue
of Section 5.3.
