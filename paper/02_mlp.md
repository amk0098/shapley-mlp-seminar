# Part I — Theory {-}

# 2 Multi-layer perceptrons, logits and embeddings

## 2.1 One layer

A layer is a row of neurons. Each neuron forms a weighted sum of all its inputs, adds a constant, and passes
the result through a simple non-linear function, the *activation*. With ReLU the activation is "negative
becomes 0, positive stays". Doing this for all neurons at once is a matrix–vector product:

$$
h = \sigma(Wx + b).
$$

| Symbol | Meaning |
|---|---|
| $x \in \mathbb{R}^{n_\text{in}}$ | input vector of the layer |
| $W \in \mathbb{R}^{n_\text{out} \times n_\text{in}}$ | weight matrix; row $j$ holds the weights of neuron $j$ |
| $b \in \mathbb{R}^{n_\text{out}}$ | bias vector, one number per neuron |
| $z = Wx + b$ | pre-activation: a linear map plus a shift |
| $\sigma$ | activation function, applied to each component; here $\mathrm{ReLU}(z) = \max(0, z)$ |
| $h \in \mathbb{R}^{n_\text{out}}$ | output (activation) of the layer |

For a batch of $B$ rows stacked in a matrix $X \in \mathbb{R}^{B \times n_\text{in}}$ the same layer reads
$H = \sigma(XW^\top + b)$. Here $\sigma$ is a placeholder for *any* activation, not necessarily the sigmoid.

**Why the activation is needed.** Without $\sigma$, two layers collapse into one:
$W_2(W_1x + b_1) + b_2 = (W_2W_1)x + (W_2b_1 + b_2)$. The non-linearity is what makes an MLP more expressive
than a linear model [GBC16] — and it is also exactly what makes its Shapley values hard to compute (Section 6).

> **Exercise 1.** Two inputs, two neurons, ReLU: $x = (2, 1)$, $W = \begin{pmatrix} 1 & 2 \\ -1 & 1
> \end{pmatrix}$, $b = (-1, -2)$. Compute $z = Wx + b$ and $h = \mathrm{ReLU}(z)$. Which neuron is "off"?

## 2.2 Logit, probability and embedding

For a binary classification task the last layer has a single neuron and **no** activation. Its output is the
**logit** $z_\text{out}$, an arbitrary real number: positive means "rather class 1", negative "rather class 0".
The **probability** is obtained with the sigmoid function,

$$
p = \frac{1}{1 + e^{-z_\text{out}}}, \qquad z_\text{out} = \log\frac{p}{1-p} \quad (\text{log-odds}).
$$

In PyTorch the sigmoid is usually not part of the network but of the loss function (`BCEWithLogitsLoss`), which
is numerically more stable [Pas19].

The **embedding** is what the network "knows" just before its final decision: the activations of the last
hidden layer. If $h^{(L)}$ denotes this vector and $w, c$ the weights and bias of the output layer, then

$$
z_\text{out} = w^\top h^{(L)} + c.
$$

The logit is a **linear** function of the embedding. Section 7 builds on this fact.

**Why Shapley values are computed on the logit.** Shapley values add up (Section 4): the contributions of all
features sum to the difference between this prediction and the average prediction. The sigmoid compresses the
same logit change into very different probability changes depending on the starting point, so contributions
cannot be stated in percentage points.

> **Exercise 2.** The base logit is $-3$. Two features have Shapley values $\varphi_1 = +2.5$ and
> $\varphi_2 = +1.5$ (logit units), so the logit of this day is $+1$. Useful values:
> $\mathrm{sigmoid}(-3) = 0.047$, $\mathrm{sigmoid}(-1.5) = 0.182$, $\mathrm{sigmoid}(-0.5) = 0.378$,
> $\mathrm{sigmoid}(1) = 0.731$. (a) By how much does the probability rise when feature 1 is added first and
> feature 2 second? (b) And in the opposite order? (c) What does this mean for the sentence "feature 1 adds
> $x$ percentage points"?

Two practical consequences: never apply the sigmoid to individual Shapley values, and never pass a model that
outputs probabilities to an explainer together with a logit link — the base value then becomes the logit of the
*mean probability*, which is not the *mean logit*.

## 2.3 The network of the case study

The case-study network maps 7 standardised weather features to one logit:

$$
\begin{aligned}
h^{(1)} &= \mathrm{ReLU}(W_1 x + b_1), & W_1 &\in \mathbb{R}^{16 \times 7}, & h^{(1)} &\in \mathbb{R}^{16},\\
h^{(2)} &= \mathrm{ReLU}(W_2 h^{(1)} + b_2), & W_2 &\in \mathbb{R}^{8 \times 16}, & h^{(2)} &\in \mathbb{R}^{8}
\quad \text{(the embedding)},\\
z_\text{out} &= w^\top h^{(2)} + c, & w &\in \mathbb{R}^{8}, & z_\text{out} &\in \mathbb{R} \quad \text{(the logit)}.
\end{aligned}
$$

In the code the two hidden layers form a module `body` (its output is the embedding) and the output layer a
module `head` (its output is the logit); `forward` returns the logit, shape $(B, 1)$. The activations are
`nn.ReLU()` *modules*, one object per place — a requirement of SHAP's DeepExplainer, which attaches its hooks
to modules (Section 6.3).

> **Exercise 3.** How many trainable parameters does this network have? (PyTorch stores each weight matrix with
> shape (out, in).)

The network is deliberately small. The task (Section 9) has seven inputs, and the purpose is not the best
possible forecast but a model whose explanations can be checked exactly.
