# 7 Shapley values on logits and on embeddings

"Shapley values on the embeddings" can mean two different things. Both are worked out here; they answer
different questions and turn out to be two margins of one table.

- **Logit (the standard case).** Players are the 7 inputs, the payout is the logit: *which weather feature
  pushed the score of this day up or down?*
- **Reading (a): inputs → embedding neuron.** Players are still the 7 inputs, but the payout is the activation
  of one neuron of the last hidden layer, for each of the 8 neurons: *which inputs switch this internal neuron
  on?*
- **Reading (b): embedding → logit.** The 8 embedding neurons are the players and the payout is the logit; the
  inputs do not appear: *which internal neurons carried the decision for this day?*

| | Logit | (a) inputs → neuron $k$ | (b) neurons → logit |
|---|---|---|---|
| Players $M$ | 7 inputs | 7 inputs | 8 neurons |
| Explained function | logit$(x)$ | $h_k(x)$, $k = 1, \dots, 8$ | head$(h) = w^\top h + c$ |
| Values per instance | 7 | 8 × 7 | 8 |
| Base value | mean logit over background | mean of $h_k$ over background | mean logit over background |
| Efficiency | $\sum_i \varphi_i = \text{logit}(x) - \varphi_0$ | $\sum_i \varphi_{k,i} = h_k(x) - \overline{h_k}$ | $\sum_k \varphi_k = \text{logit}(x) - \varphi_0$ |
| Unit | log-odds | activation of neuron $k$ | log-odds |
| Interpretable? | yes, features have names | columns yes; neurons only indirectly | neurons have no names |

The background for reading (b) consists of the embeddings of the background days.

## 7.1 Reading (b) has a closed form

The head is linear (Section 2.2), so the linear shortcut of Section 5 applies:

$$
\varphi^{(b)}_k = w_k\,\bigl(h_k(x) - \overline{h_k}\bigr), \qquad \overline{h_k} = \frac1N \sum_b h_k(b).
$$

Consequences: exact, KernelSHAP and DeepSHAP must all agree in reading (b); its base value is
$w^\top \bar h + c$, the **same number** as the mean logit, because averaging commutes with a linear function;
and a neuron that is zero for $x$ and for the whole background gets exactly zero. This holds because the
embedding is the **last** hidden layer, taken **after** its ReLU. For an earlier layer the remaining network
would contain a ReLU and the closed form would be lost.

## 7.2 One table connects all three views

Additivity (Section 4) applied to $\text{logit} = \sum_k w_k h_k + c$ gives

$$
\varphi_i^{\text{logit}} = \sum_k w_k\, \varphi_{k,i}^{(a)}
$$

(a constant game has all Shapley values zero, so $c$ drops out). Define the 8 × 7 table
$T_{k,i} = w_k\,\varphi^{(a)}_{k,i}$. Then

- the **column sums** of $T$ are the logit attributions of the inputs;
- the **row sums** are $w_k \sum_i \varphi^{(a)}_{k,i} = w_k(h_k(x) - \overline{h_k}) = \varphi^{(b)}_k$, the
  reading-(b) values;
- the **total** is $\text{logit}(x) - \varphi_0$.

Reading (a) is the full table, reading (b) its row margin, and the usual logit explanation its column margin.
The identity holds for exact interventional values and for KernelSHAP when it enumerates all coalitions; for
DeepSHAP the column relation also holds, because the chain rule through a linear layer is exact. On the trained
case-study network the identities were checked for all 3650 test days: the largest deviations are
$6 \cdot 10^{-7}$ (columns, exact), $2 \cdot 10^{-6}$ (rows) and $3 \cdot 10^{-6}$ (columns, DeepSHAP), i.e.
float32 rounding.

> **Exercise 17.** Two inputs, two embedding neurons, $w = (2, -1)$, $c = 0$. Reading (a) gave
> $\varphi^{(a)} = (0.5, 1.0)$ for neuron 1 and $(1.0, -2.0)$ for neuron 2. (a) Build $T$. (b) What are the
> logit attributions of the two inputs? (c) What are the reading-(b) values? (d) What is
> $\text{logit}(x) - \varphi_0$?

> **Exercise 18 (code).** Write `linear_head_shapley(w, h_x, H_bg)` that returns the closed-form reading-(b)
> values ($w$, $h_x$ of shape (8,), $H_\text{bg}$ of shape (N, 8)). Check that the result sums to
> $\text{logit}(x) - \varphi_0$ and equals KernelSHAP on the head with `l1_reg=False`.

## 7.3 What makes embeddings hard to interpret

- Embedding neurons have no names and no fixed meaning; after retraining with another seed "neuron 5" is a
  different thing. Reading-(b) values are not comparable across training runs.
- Hidden neurons are often correlated and partly redundant. Under the interventional value function coalitions
  of neurons create embedding vectors that no input could produce. Because the head is linear this does not
  distort the numbers, but it limits their meaning.
- A beeswarm plot of "neuron 7" explains nothing without reading (a) — or another way of saying what neuron 7
  reacts to. Section 12 combines both readings for exactly this reason.

# 8 A cost model for the runtime

All three methods spend their time evaluating the network, so the runtime can be predicted by counting model
rows. Let $c_f$ and $c_b$ be the cost of one forward and one backward pass for one row, $D$ the number of
explained outputs (1 for the logit, 8 for reading (a)), $n_s = \min(2^M - 2, \texttt{nsamples})$.

| Method | Model rows per instance | Total cost | in $M$ | in $N$ | in $B$ |
|---|---|---|---|---|---|
| Exact (cached) | $2^M N$ | $B \cdot 2^M N\, c_f$ | exponential | linear | linear |
| KernelSHAP | $n_s N$ | $B\,(n_s N c_f + \text{regression})$ | $2^M$ up to $M = 11$, then ≈ flat | linear | linear |
| DeepSHAP | $2N$ forward, $N$ backward, per output | $B \cdot D \cdot N\,(2c_f + c_b)$ | none | linear | linear |

**"Batch size".** In both `shap` explainers the explained instances are processed one by one. The number of
explained instances $B$ is therefore not a batch in the GPU sense: nothing is shared between instances, and
the time grows linearly with $B$ — the time *per instance* stays constant. What is processed as one batch is the
block of $n_s N$ rows (KernelSHAP) or $2N$ rows (DeepSHAP) of a single instance.

**Predictions** (written down before measuring; Section 13 compares them with the measurements):

| # | Prediction |
|---|---|
| P1 | Time grows linearly with the number of explained days $B$ for all methods. |
| P2 | Time grows linearly with the background size $N$. |
| P3 | With $M = 7$, $N = 100$: exact 12 800 rows per day, KernelSHAP 12 600, DeepSHAP 200 forward + 100 backward. Exact and KernelSHAP should take similar time; DeepSHAP should be faster. |
| P4 | Exact time doubles with every additional feature; KernelSHAP too until the cap, then flat; DeepSHAP flat. |
| P5 | For a network this small, fixed per-call overhead (Python, hooks, the regression) will make the measured differences smaller than the row counts suggest. |

> **Exercise 19.** With $M = 7$ and $N = 100$, how many model rows does KernelSHAP need for $B = 50$ days? And
> for $N = 400$? At 1 µs per row, how long is that, and why will the measured time be longer?
