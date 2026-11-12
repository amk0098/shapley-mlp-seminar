# 3 Cooperative games and the Shapley value

## 3.1 Games and coalitions

Several people work together and earn something as a group. How much of it does each person deserve? Looking
at what each one earns alone is not enough, because people are worth more — or less — in combination.
Shapley's idea [Sha53]: let the players enter the room one after another, pay each newcomer exactly what the
group's value rises by when they enter, and do this for **every possible order of arrival**. A player's
Shapley value is their average pay over all orders.

Formally, a **cooperative game** consists of a set of players $\mathcal{N} = \{1, \dots, M\}$ and a
*value function* $v$ that assigns a number $v(S)$ to every coalition $S \subseteq \mathcal{N}$, usually with
$v(\emptyset) = 0$.

**Running example.** Players A, B and C can do a job alone or in any group; the earnings in euros are:

| Coalition $S$ | ∅ | {A} | {B} | {C} | {A,B} | {A,C} | {B,C} | {A,B,C} |
|---|---|---|---|---|---|---|---|---|
| $v(S)$ | 0 | 6 | 0 | 0 | 12 | 6 | 0 | 18 |

Only A can earn alone. B doubles what A earns. C adds nothing to A alone and nothing to B alone, but adds 6
once A and B are both present. The question is how to split the 18.

## 3.2 The formula

$$
\varphi_i \;=\; \sum_{S \subseteq \mathcal{N} \setminus \{i\}}
\frac{|S|!\,(M - |S| - 1)!}{M!}\,\bigl[v(S \cup \{i\}) - v(S)\bigr].
$$

| Symbol | Meaning |
|---|---|
| $S \subseteq \mathcal{N} \setminus \{i\}$ | any coalition that does **not** contain $i$, including ∅; there are $2^{M-1}$ of them |
| $v(S \cup \{i\}) - v(S)$ | *marginal contribution* of $i$ when joining $S$ |
| $\lvert S\rvert!$ | number of ways to order the players who arrive before $i$ |
| $(M - \lvert S\rvert - 1)!$ | number of ways to order the players who arrive after $i$ |
| $M!$ | number of all orders |
| the fraction | share of all orders in which exactly the set $S$ arrives before $i$ (Exercise 6) |

Because the fraction is a share of orders, the formula is equivalent to the **permutation form**

$$
\varphi_i = \frac{1}{M!} \sum_{\pi} \bigl[v(P_i^\pi \cup \{i\}) - v(P_i^\pi)\bigr],
$$

where $\pi$ runs over all $M!$ orders and $P_i^\pi$ is the set of players before $i$ in $\pi$.

> **Exercise 4 (subset view).** (a) Compute the weight for $|S| = 0, 1, 2$ with $M = 3$. (b) List the four
> coalitions $S \subseteq \{B, C\}$ and A's marginal contribution for each. (c) Compute $\varphi_A$, then
> $\varphi_B$ and $\varphi_C$. (d) Do the three values add up to 18?

> **Exercise 5 (permutation view).** Write down all $3! = 6$ orders of arrival. For each order, note what
> each player adds when entering (in the order B, A, C player A adds $v(\{A,B\}) - v(\{B\})$). Average each
> player's column; you must obtain the values of Exercise 4.

> **Exercise 6 (where the factorials come from).** In how many of the 6 orders does **exactly** {B} stand in
> front of A? Compare with the weight of $S = \{B\}$, and explain in one sentence where $|S|!$ and
> $(M - |S| - 1)!$ come from.

**A second reading of the weight.** One can verify that

$$
\frac{|S|!\,(M-|S|-1)!}{M!} = \frac{1}{M} \cdot \frac{1}{\binom{M-1}{|S|}}.
$$

First the *position* of $i$ is chosen uniformly ($M$ possibilities, which fixes the size of $S$), then
uniformly *which* players of that size stand in front ($\binom{M-1}{|S|}$ possibilities). Every coalition
**size** therefore gets the same total weight $1/M$. For $M = 7$ the weights for $|S| = 0, \dots, 6$ are
$1/7, 1/42, 1/105, 1/140, 1/105, 1/42, 1/7$, and there are $1, 6, 15, 20, 15, 6, 1$ coalitions of these sizes:
each size contributes exactly $1/7$.

**Why all subsets?** A player's contribution depends on who is already present: C is worth 0 or 6. Any
single coalition, or any single order, would give an arbitrary answer. Two common mistakes follow from
ignoring this:

- Taking the *unweighted* mean of the marginal contributions. For A this gives $(6 + 12 + 6 + 18)/4 = 10.5$
  instead of the correct value. Orders are equally likely; coalitions are not.
- Reading $\varphi_i$ as "what $i$ earns alone" or "what is lost when $i$ leaves the full group". These are
  only two of the $2^{M-1}$ terms ($S = \emptyset$ and $S = \mathcal{N} \setminus \{i\}$).

# 4 The four axioms

Shapley did not start from the formula. He asked which properties any fair division must have, wrote them down
and proved that exactly one rule has all of them [Sha53]. The axioms are the *definition* of "fair"; the formula
is the consequence.

| Axiom | Statement | Example in the running game |
|---|---|---|
| **Efficiency** | $\sum_i \varphi_i = v(\mathcal{N}) - v(\emptyset)$: everything is distributed. | The three values of Exercise 4 sum to 18. |
| **Symmetry** | If $i$ and $j$ add the same to every coalition containing neither, then $\varphi_i = \varphi_j$. | In the bonus game $w(S) = 3\lvert S\rvert$ all players are interchangeable and each gets 3. |
| **Dummy (null player)** | If $i$ never changes any coalition's value, then $\varphi_i = 0$. | Add a player D with $v(S \cup \{D\}) = v(S)$ for all $S$: D gets 0, A, B and C keep their values. |
| **Additivity** | For two games on the same players, $\varphi_i(v + w) = \varphi_i(v) + \varphi_i(w)$. | Adding the bonus game raises every value of the running game by 3. |

> **Exercise 7.** Use only the axioms, no formula. In the game $u$ with $u(S) = 10$ if $A \in S$ **and**
> $B \in S$, and $u(S) = 0$ otherwise (players A, B, C): what are $\varphi_A$, $\varphi_B$, $\varphi_C$? Name the
> axiom used in each step.

**How many axioms?** The supervisor's question and most textbooks list four (efficiency, symmetry, dummy,
additivity). Shapley's original paper uses three: symmetry, additivity and a *carrier* axiom that combines
efficiency and the null-player property [Sha53]. Young [You85] showed that additivity and the null player can be
replaced by a single *monotonicity* (marginality) requirement. Lundberg and Lee [LL17, Sec. 3] formulate three
properties for explanation models — *local accuracy*, *missingness* and *consistency* — and show that only one
additive attribution satisfies them; their proof relies on Young's result. All these lists characterise the
same value; they should not be mixed on one slide.

**What the theorem does and does not say.** It proves uniqueness *for a given game* $v$. The axioms say nothing
about how $v$ should be defined for a neural network. That choice (Section 5) is where most of the difficulties
of Shapley explanations arise [SN20, Kum20].

**Why additivity matters most in practice.** Additivity together with scaling implies that the Shapley value
of an *average of games* is the *average of the Shapley values*. This will be used three times: to average
over background rows (Section 5), to justify that DeepSHAP averages over references (Section 6.3), and to
connect the two readings of "Shapley values on the embeddings" (Section 7).
