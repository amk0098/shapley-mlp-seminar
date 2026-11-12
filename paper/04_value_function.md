# 5 From the game to the network: value function and background

## 5.1 Simulating an absent feature

A game needs $v(S)$ for *every* subset of features, but a network cannot run with "feature 3 missing": it
always needs all seven numbers. Absence must therefore be **simulated**. The standard approach keeps the
features in $S$ at their values in $x$ and fills in the others with values from real data, the **background**.
This is repeated for many background rows and the outputs are averaged:

$$
v_x(S) = \frac{1}{N} \sum_{b \in \mathcal{B}} f\bigl(z(S, b)\bigr), \qquad
z(S, b)_j = \begin{cases} x_j & j \in S \quad(\text{present: taken from the instance})\\
b_j & j \notin S \quad(\text{absent: taken from the background row})\end{cases}
$$

where $\mathcal{B}$ is the background set with $N$ rows. Two special cases follow directly:

$$
v_x(\emptyset) = \frac{1}{N}\sum_{b} f(b) = \varphi_0 \;(\text{base value}), \qquad v_x(\mathcal{N}) = f(x).
$$

Efficiency (Section 4) therefore reads

$$
\sum_{i=1}^{M} \varphi_i = f(x) - \varphi_0 .
$$

The Shapley values explain the **gap** between this prediction and the average prediction over the background,
not the prediction itself. Note that $\varphi_0$ is the mean of the *outputs*, $\frac1N\sum_b f(b)$, not the
output at the mean input $f(\bar b)$; for a non-linear $f$ these differ.

> **Exercise 8 (linear model).** $f(x_1, x_2) = 3x_1 - 2x_2 + 1$, instance $x = (2, 1)$, background = two rows
> $(0, 0)$ and $(2, 3)$. (a) Compute $v(\emptyset)$, $v(\{1\})$, $v(\{2\})$, $v(\{1,2\})$. (b) Compute
> $\varphi_1, \varphi_2$ (with $M = 2$ both weights are $1/2$). (c) Compare with the shortcut
> $\varphi_i = w_i\,(x_i - \bar b_i)$ and check efficiency.

The shortcut of Exercise 8 holds for every linear model: adding feature $i$ to *any* coalition changes $v$ by
the same amount $w_i(x_i - \bar b_i)$, so the weighted average is that amount.

**Averaging over references.** For each background row $b$ define the single-reference game
$v_b(S) = f(z(S, b))$. Then $v_x$ is the average of the $v_b$, and by additivity

$$
\varphi_i(\text{background of } N \text{ rows}) = \frac1N \sum_b \varphi_i(\text{single reference } b).
$$

## 5.2 Interventional versus conditional

There are two ways to fill in the absent features.

| | Interventional (marginal) | Conditional (observational) |
|---|---|---|
| Absent features are drawn from | the background, **independently** of $x_S$ | the distribution of days that **match** $x_S$ |
| $v(S)$ | $\mathbb{E}_b\bigl[f(x_S, b_{\bar S})\bigr]$ | $\mathbb{E}\bigl[f(X) \mid X_S = x_S\bigr]$ |
| Question answered | "What does the *model* do when these inputs are set?" | "What does *knowing* these inputs say about the output?" |
| Weakness | creates feature combinations that never occur (*off-manifold*) | needs the conditional distribution (hard to estimate); features the model ignores can receive credit |

Lundberg and Lee define SHAP values with the conditional expectation and then approximate it by assuming
feature independence [LL17, Eq. 11] — which turns it into the interventional value function. KernelSHAP and
DeepSHAP in the `shap` library, and the exact computation in the case study, all use the interventional
version, so all three answer the same question. Janzing et al. argue that the unconditional (interventional)
expectation is the conceptually right notion of dropping a feature when the goal is to explain the model
[JMB20]. Sundararajan and Najmi show that the conditional approach can assign non-zero attributions to features
the model does not use [SN20]. Aas et al. show that the independence assumption can make explanations
inaccurate when features are dependent and propose conditional estimators [AJL21].

> **Exercise 9 (the two value functions disagree).** Model $f(x_1, x_2) = x_1$; it does not use $x_2$. In the
> data $x_2 = x_1$ always, both with mean 0. Explain $x = (2, 2)$. (a) Interventional: compute all four $v(S)$
> and $\varphi_1, \varphi_2$. (b) Conditional: $v(\{2\}) = \mathbb{E}[f(X) \mid X_2 = 2]$; compute $\varphi_1,
> \varphi_2$ again. (c) Which answer gives the ignored feature 0?

## 5.3 Why this matters for correlated weather features

In the case study the features are tied together by construction, not only statistically: the 3-day mean is
the mean of the maximum temperatures of the three previous days, and the previous day's maximum is one of these
three days; on every real day the minimum ≤ mean ≤ maximum temperature.

> **Exercise 10 (an impossible day).** Coalition $S$ = {previous day's maximum}, instance with previous-day
> maximum 35 °C, background row from January with 3-day mean 4 °C. The hybrid input claims: yesterday's maximum
> was 35 °C **and** the mean of the last three maxima was 4 °C. What must the other two days' maxima sum to?
> Has the network ever seen such a day?

Consequences, all visible in Part II:

- Interventional Shapley values average the model's output over many such impossible inputs. The network was
  never trained there, so its output is extrapolation. The values remain a correct description of *the model*,
  but partly of behaviour that never occurs.
- The credit for "it has been hot" is **split** among the five temperature features. None of them looks as
  important as "temperature" as a whole, and on single days correlated features can receive opposite signs.
- Small changes of the model or the background can move credit between correlated features [AJL21, Kum20].
