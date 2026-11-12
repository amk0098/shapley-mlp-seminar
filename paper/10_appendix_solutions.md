# Appendix A — Solutions to the exercises {-}

All numerical solutions were checked by computation; the DeepLIFT solutions (14–16) were also reproduced with
`shap.DeepExplainer` on equivalent PyTorch modules.

**A.1 (Exercise 1).** $z_1 = 1\cdot2 + 2\cdot1 - 1 = 3$, $z_2 = -1\cdot2 + 1\cdot1 - 2 = -3$, so $z = (3, -3)$
and $h = (3, 0)$. Neuron 2 is off.

**A.2 (Exercise 2).** (a) Feature 1 first: logit $-3 \to -0.5$, probability $0.047 \to 0.378$ (+0.330); then
feature 2: $0.378 \to 0.731$ (+0.354). (b) Feature 2 first: $-3 \to -1.5$, $0.047 \to 0.182$ (+0.135); then
feature 1: $0.182 \to 0.731$ (+0.549). (c) Feature 1 "adds" 33 or 55 percentage points depending on the order;
in logit space it adds 2.5 in both orders. A logit contribution has no fixed probability equivalent.

**A.3 (Exercise 3).** Layer 1: $16 \cdot 7 + 16 = 128$; layer 2: $8 \cdot 16 + 8 = 136$; output: $8 + 1 = 9$.
Total **273**.

**A.4 (Exercise 4).** Weights for $M = 3$: $|S| = 0$: $0!\,2!/3! = 1/3$; $|S| = 1$: $1/6$; $|S| = 2$: $1/3$.

| $S$ (for A) | ∅ | {B} | {C} | {B,C} |
|---|---|---|---|---|
| marginal contribution | 6 | 12 | 6 | 18 |
| weight | 1/3 | 1/6 | 1/6 | 1/3 |

$\varphi_A = 2 + 2 + 1 + 6 = 11$. For B the marginals are 0, 6, 0, 12, so $\varphi_B = 0 + 1 + 0 + 4 = 5$. For
C they are 0, 0, 0, 6, so $\varphi_C = 2$. Sum $= 18$ ✓.

**A.5 (Exercise 5).**

| Order | A adds | B adds | C adds |
|---|---:|---:|---:|
| A, B, C | 6 | 6 | 6 |
| A, C, B | 6 | 12 | 0 |
| B, A, C | 12 | 0 | 6 |
| B, C, A | 18 | 0 | 0 |
| C, A, B | 6 | 12 | 0 |
| C, B, A | 18 | 0 | 0 |
| **Mean** | **11** | **5** | **2** |

Every row sums to 18: in every order the whole value is paid out — that is efficiency.

**A.6 (Exercise 6).** Only the order B, A, C has exactly {B} in front of A: 1 of 6 orders $= 1/6$, the weight
of $S = \{B\}$. In general the $|S|$ players of $S$ can stand in front of $i$ in $|S|!$ orders and the other
$M - |S| - 1$ players behind $i$ in $(M - |S| - 1)!$ orders. In all these orders $i$'s marginal contribution is
the same number, so the average over orders collapses into the weighted sum over subsets.

**A.7 (Exercise 7).** C never changes $u$ → Dummy: $\varphi_C = 0$. A and B add the same to every coalition
without them (0 to ∅ and 0 to {C}) → Symmetry: $\varphi_A = \varphi_B$. Efficiency: $\varphi_A + \varphi_B +
\varphi_C = u(\mathcal N) - u(\emptyset) = 10$, so $\varphi_A = \varphi_B = 5$.

**A.8 (Exercise 8).** $f(0,0) = 1$, $f(2,3) = 1$ → $v(\emptyset) = 1$. $v(\{1\}) = \frac12(f(2,0) + f(2,3)) =
\frac12(7 + 1) = 4$. $v(\{2\}) = \frac12(f(0,1) + f(2,1)) = \frac12(-1 + 5) = 2$. $v(\{1,2\}) = f(2,1) = 5$.
$\varphi_1 = \frac12(4 - 1) + \frac12(5 - 2) = 3$, $\varphi_2 = \frac12(2 - 1) + \frac12(5 - 4) = 1$. Shortcut:
$\bar b = (1, 1.5)$, $\varphi_1 = 3(2 - 1) = 3$, $\varphi_2 = -2(1 - 1.5) = 1$ ✓. Efficiency: $3 + 1 = 5 - 1$ ✓.

**A.9 (Exercise 9).** Interventional: $v(\emptyset) = 0$, $v(\{1\}) = 2$, $v(\{2\}) = 0$, $v(\{1,2\}) = 2$, so
$\varphi_1 = 2$, $\varphi_2 = 0$. Conditional: knowing $X_2 = 2$ implies $X_1 = 2$, so $v(\{2\}) = 2$ and
$\varphi_1 = \varphi_2 = 1$. Only the interventional answer gives the ignored feature 0. The conditional answer
is not wrong — it answers "what does knowing $x_2$ tell me?", and here that is everything.

**A.10 (Exercise 10).** The three maxima must sum to $3 \cdot 4 = 12$ °C; with 35 °C yesterday the other two
would have to sum to −23 °C, i.e. two days with maxima around −11 °C directly before a 35 °C day. No such
sequence exists in the data; the network's output there is extrapolation.

**A.11 (Exercise 11).** (a) $7 \cdot 2^6 = 448$ terms × 2 = 896 evaluations of $v$. (b) Only $2^7 = 128$
different coalitions. (c) Without a cache every coalition is evaluated seven times; compute each $v(S)$ once
and form the Shapley values as weighted differences — this is what the case-study code does.

**A.12 (Exercise 12).** $M = 3$: $\pi(1) = 2/(3\cdot1\cdot2) = 1/3$, $\pi(2) = 1/3$. $M = 4$: $\pi(1) =
3/(4\cdot1\cdot3) = 1/4$, $\pi(2) = 3/(6\cdot2\cdot2) = 1/8$, $\pi(3) = 1/4$. The smallest and largest
coalitions get the highest weight (for $M = 7$: $1/7$ for sizes 1 and 6, $1/70$ for sizes 3 and 4): a coalition
with one feature present, or one feature missing, isolates the effect of a single feature most directly.

**A.13 (Exercise 13).** With $\varphi_0 = 0$ and $\sum_k\varphi_k = 18$ the six squared errors are, for each
$k$, $(\varphi_k - a_k)^2 + (\varphi_k - c_k)^2$ with $a_k = v(\{k\})$ (single) and $c_k = 18 - v(\mathcal N
\setminus \{k\})$ (pair).

| $k$ | $a_k$ | $c_k$ | unconstrained minimum $(a_k + c_k)/2$ |
|---|---:|---:|---:|
| A | 6 | 18 | 12 |
| B | 0 | 12 | 6 |
| C | 0 | 6 | 3 |

These sum to 21 but must sum to 18. All three terms have the same curvature, so the constraint shifts each by
the same amount $\lambda = (18 - 21)/3 = -1$ (Lagrange multiplier). Result **(11, 5, 2)** — the Shapley values.

**A.14 (Exercise 14).** At $x$: $z = 2$, $y = 2$; at $b$: $z = -1$, $y = 0$; $\Delta z = 3$, $\Delta y = 2$.
Rescale multiplier $2/3$; $m(x_1 \to y) = 4/3$, $m(x_2 \to y) = 2/3$; $C_1 = 1.333$, $C_2 = 0.667$; sum $2 =
\Delta y$ ✓. Exact: $v(\emptyset) = 0$, $v(\{1\}) = \mathrm{ReLU}(1) = 1$, $v(\{2\}) = \mathrm{ReLU}(0) = 0$,
$v(\{1,2\}) = 2$, so $\varphi_1 = 1.5$, $\varphi_2 = 0.5$. Same sum, different split.

**A.15 (Exercise 15).** For $b' = (1, 0)$: $z = 1$, $y = 1$, $\Delta z = \Delta y = 1$, multiplier 1; $C_1 = 0$,
$C_2 = 1$. Exact for $b'$: $v(\emptyset) = 1$, $v(\{1\}) = 1$, $v(\{2\}) = 2$, $v(\{1,2\}) = 2$, so $\varphi =
(0, 1)$ — identical, because between $b'$ and $x$ the neuron stays on and the function is linear there. For $b$
it switches from off to on. Averages: DeepSHAP $(0.667, 0.833)$, exact $(0.75, 0.75)$; both sum to $1.5 = 2 -
\frac12(0 + 1)$ ✓.

**A.16 (Exercise 16).** At $b$: $z_1 = -1$, $h_1 = 0$, $z_2 = 0$, $h_2 = 0$, $y = 0$. At $x$: $z_1 = 2$,
$h_1 = 2$, $z_2 = 1$, $h_2 = 1$, $y = 3$. Multipliers: $\Delta h_1/\Delta z_1 = 2/3$, $\Delta h_2/\Delta z_2 = 1$.
$m(x_1 \to y) = 1\cdot\frac23\cdot2 + 1\cdot1\cdot(-1) = \frac13$; $m(x_2 \to y) = 1\cdot\frac23\cdot2 +
(-1)\cdot1\cdot(-1) = \frac73$. $C_1 = 0.667$, $C_2 = 2.333$, sum 3 ✓. Exact: $v(\emptyset) = 0$, $v(\{1\}) =
f(2,0) = 2\cdot1 - 2 = 0$, $v(\{2\}) = f(0,1) = 0$, $v(\{1,2\}) = 3$, so $\varphi_1 = \varphi_2 = 1.5$. DeepLIFT
says "mostly $x_2$", Shapley says "both equally, and only together": the game is a pure interaction, and the
single backward pass along $(b, x)$ cannot see the hybrid inputs $(2,0)$ and $(0,1)$.

**A.17 (Exercise 17).** Rows scaled by $w$: neuron 1: $2\cdot(0.5, 1.0) = (1, 2)$; neuron 2: $-1\cdot(1.0, -2.0)
= (-1, 2)$. Column sums (logit attributions): input 1: 0, input 2: 4. Row sums (reading b): neuron 1: 3,
neuron 2: 1. Total: 4. Input 1 influences both neurons, but with opposite effects on the logit — invisible in
the logit explanation alone.

**A.18 (Exercise 18).** `return w * (h_x - H_bg.mean(axis=0))`. The case-study script performs both checks;
the closed form agrees with the exact reading-(b) values to $1.3\cdot10^{-6}$ (Section 12).

**A.19 (Exercise 19).** Per day $126 \cdot 100 = 12\,600$ rows; $B = 50$ → 630 000 rows; with $N = 400$ →
2 520 000 rows. At 1 µs per row: 0.63 s and 2.5 s. Measured times are longer because of per-day work in Python:
building the coalition array, the weighted regression and conversions between NumPy and PyTorch.
