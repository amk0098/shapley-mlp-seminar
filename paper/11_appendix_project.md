# Appendix B — Documentation of the seminar project {-}

## B.1 Requirements

The seminar is held at the chair of automation technology / computer science of the Bergische Universität
Wuppertal: a 30-minute talk in German for a mixed audience (electrical engineering, computer science,
industrial engineering), roughly 12–18 slides plus reserve slides. The supervisor's requirements, received on
5 November 2026 (in German; paraphrased here), were:

1. Train any MLP on any dataset.
2. Explain the architecture of the MLP and how it is built for the chosen task.
3. Explain the mathematics behind Shapley values.
4. Analyse and explain the chosen dataset, "so that one knows where one is".
5. Apply Shapley values to the embeddings and to the logits, demonstrated with DeepSHAP and KernelSHAP.

and seven questions the speaker must be able to answer:

| # | Question (paraphrased) | Answered in |
|---|---|---|
| 1 | How is one MLP layer written as a matrix–vector operation with an activation? | Section 2.1 |
| 2 | What does cooperative game theory say, and how are features interpreted as players? | Sections 3, 5.1 |
| 3 | How is the Shapley formula built, and why do factorials and all subsets appear? | Section 3.2, Exercise 6 |
| 4 | What is the difference between Shapley values on the logits and on the embeddings? | Sections 7, 12 |
| 5 | How do sampling (KernelSHAP) and DeepSHAP (via DeepLIFT) solve the performance problem? | Sections 6.1–6.3 |
| 6 | Which factors determine the runtime, and how does it scale with the batch size? | Sections 8, 13 |
| 7 | Which four axioms make the distribution "mathematically correct and fair"? | Sections 4, 6.4 |

## B.2 Timeline

| Date (2026) | Step |
|---|---|
| 31 Oct – 5 Nov | Topic and supervisor confirmed; chair guidelines for talks read; requirements received (5 Nov) |
| 5 Nov | Decision: own HeatRisk data and a PyTorch MLP (Section B.3) |
| 10 Nov | Theory guide written (the basis of Part I of this paper) |
| 11 Nov | Audit of the source data pipeline; MLP trained (Run 1); exact Shapley values (Run 2); KernelSHAP and DeepSHAP compared (Run 3) |
| 12 Nov | Runtime (Run 4), background sensitivity (Run 5), embeddings (Run 6), slide figures, full reproduction from a fresh clone (Run 7); slides filled with results; code repository prepared for publication |
| 12 Nov | This paper |
| 19 Nov | Meeting with the supervisor: embeddings approach confirmed; requested a short explanation of how an MLP works with a visual of the trained network, and a clear statement that Shapley values describe the model, not the real world |
| 21 Nov | Final slides: MLP overview, network with real activations (Run 8), "why not the weights", "model, not the world", a prepared audience question |
| 1 Dec | Talk |

## B.3 Decisions

| Decision | Alternatives | Reason |
|---|---|---|
| HeatRisk weather data (own earlier project) | bundled scikit-learn datasets (breast cancer, California housing) | intuitive for a mixed audience; pipeline and baselines already existed; 7 features → exact Shapley values feasible; realistic problems (correlation, rare class) |
| PyTorch, 7 → 16 → 8 → 1 | scikit-learn `MLPClassifier`; larger networks | DeepExplainer supports PyTorch, not scikit-learn; a small network is enough for the task and keeps exact values cheap |
| Explain the **logit** | explain the probability | additivity holds on the logit scale (Section 2.2) |
| Interventional value function, 100 random training days as background | conditional; test-period background; summer-only background | the value function of KernelSHAP and DeepSHAP; no test data in any choice; background effects studied separately (Section 14) |
| Own exact implementation as ground truth | trust the library | makes the approximation error measurable; required understanding of the formula |
| Both readings of "embeddings" | ask and wait for one reading | the meaning was unclear; both are cheap and they form one table (Section 7.2) |
| Runtime measured in days, background size and features | only days | the supervisor's "batch size" is ambiguous; the number of features is what separates the methods |
| Class weight 32, threshold from validation | no weight | higher recall at similar PR-AUC (Section 10) |

## B.4 Repositories and artefacts

| Repository | Content |
|---|---|
| [shapley-mlp-seminar](https://github.com/amk0098/shapley-mlp-seminar) (this one) | this paper and the slides of the talk |
| [heatrisk-shap-analysis](https://github.com/amk0098/heatrisk-shap-analysis) | all code, tests, results and figures of Part II; experiment log (Runs 1–8), methodology, data audit; tag `v1.0-talk` = state used in the talk |
| [heatrisk-deutschland](https://github.com/amk0098/heatrisk-deutschland) | source project: DWD data, loader, features, split and baselines (commit `9974c0e`) |
| [mlp-shap-pipeline](https://github.com/amk0098/mlp-shap-pipeline) | follow-up project (design phase): the same analysis automated for any trained MLP, producing a SHAP report |

**Where each number comes from.** Model metrics: Run 1; exact values and ranking: Run 2; KernelSHAP/DeepSHAP
comparison, convergence and example days: Run 3; runtime: Run 4; background: Run 5; embeddings: Run 6;
reproducibility: Run 7; network figures of the final slides: Run 8 — all in `docs/experiment_log.md` of heatrisk-shap-analysis, each with the code revision
that produced it. The embedding-table identity of Section 7.2 was checked on the saved arrays of Runs 2 and 6.

## B.5 From the talk to this paper

| Talk (German slides) | Paper |
|---|---|
| Motivation, dataset, how an MLP works, the HeatRisk MLP, why weights do not explain | Sections 1, 2, 9, 10 |
| Cooperative games, Shapley formula, worked example, axioms | Sections 3, 4 |
| Value function and background | Section 5 |
| Exact, KernelSHAP, DeepSHAP | Section 6 |
| Results on the logit, waterfalls, beeswarm | Section 11 |
| Embeddings | Sections 7, 12 |
| Runtime | Sections 8, 13 |
| Limitations, "model, not the world", summary | Sections 14–16 (esp. 15.2) |
| Prepared audience question: retraining with another seed | Section 15.3 |
| Reserve slides (convergence, background, derivations) | Sections 11.3, 14, Appendix A |

The talk necessarily simplifies; where they differ, this paper is the more precise version.

# Appendix C — Summary of formulas and glossary {-}

| Concept | Formula |
|---|---|
| MLP layer | $h = \sigma(Wx + b)$, $W \in \mathbb R^{\text{out}\times\text{in}}$; ReLU in hidden layers, none at the output |
| Case-study network | $7 \to 16 \to 8 \to 1$, 273 parameters; embedding $h^{(2)} \in \mathbb R^8$; logit $= w^\top h^{(2)} + c$ |
| Logit / probability | $p = 1/(1 + e^{-z})$, $z = \log(p/(1-p))$; explain $z$, not $p$ |
| Shapley value | $\varphi_i = \sum_{S \subseteq \mathcal N\setminus\{i\}} \frac{\lvert S\rvert!(M-\lvert S\rvert-1)!}{M!}[v(S\cup\{i\}) - v(S)]$ |
| Weight | share of the $M!$ orders with exactly $S$ before $i$ $= \frac1M \binom{M-1}{\lvert S\rvert}^{-1}$ |
| Axioms | efficiency, symmetry, dummy, additivity |
| Value function | $v_x(S) = \frac1N\sum_b f(x_S, b_{\bar S})$ (interventional) |
| Base value, efficiency | $\varphi_0 = \frac1N\sum_b f(b)$, $\sum_i\varphi_i = f(x) - \varphi_0$ |
| Linear model | $\varphi_i = w_i(x_i - \bar b_i)$ |
| Exact cost | $2^M N$ model rows per instance |
| KernelSHAP | $\min\sum_S \pi(\lvert S\rvert)[v(S) - \varphi_0 - \sum_{i\in S}\varphi_i]^2$, $\pi(s) = \frac{M-1}{\binom Ms s(M-s)}$; default `nsamples` $= 2M + 2048$, capped at $2^M - 2$ |
| DeepLIFT | linear: $m = w$; ReLU: $m = \Delta h/\Delta z$; chain rule; $\sum_i C_i = f(x) - f(b)$ |
| DeepSHAP | $\varphi_i \approx \frac1N\sum_b m_b(x_i\to f)(x_i - b_i)$; cost independent of $M$ |
| Embedding (a) | players = inputs, output = neuron $k$: $\sum_i\varphi_{k,i} = h_k(x) - \overline{h_k}$ |
| Embedding (b) | players = neurons, output = logit: $\varphi_k = w_k(h_k(x) - \overline{h_k})$ |
| Table | $T_{k,i} = w_k\varphi^{(a)}_{k,i}$: column sums = logit attributions, row sums = reading (b) |
| Runtime | exact $B\,2^M N$; KernelSHAP $B\,n_s N$; DeepSHAP $B\,D\,N\cdot$const — all linear in $B$ and $N$ |

| German (talk) | English (paper) |
|---|---|
| Merkmal | feature |
| Vorhersage | prediction |
| Koalition | coalition |
| Grenzbeitrag | marginal contribution |
| Wertfunktion | value function |
| Hintergrunddaten | background data |
| Basiswert | base value |
| Merkmalsattribution | feature attribution |
| verdeckte Schicht | hidden layer |
| Aktivierungsfunktion | activation function |
| Hitzetag | heat day (maximum temperature ≥ 30 °C) |
| Fehlalarm | false alarm (false positive) |

# Appendix D — Self-test {-}

Answer without looking; the section in brackets has the answer.

1. Write one MLP layer as a formula and give the shapes of $W$ and $b$ for the second hidden layer of the
   case-study network. (2)
2. For $M = 4$ and $|S| = 1$: what is the Shapley weight, how many such coalitions exist for a given player,
   and what do the weights of one size add up to? (3.2)
3. In the running game, why does C get 2 although C alone earns nothing and adds nothing to A or B alone? (3)
4. State the four axioms in one sentence each. Which one justifies averaging over background rows? (4, 5.1)
5. What are $v(\emptyset)$ and $v(\mathcal N)$ for the MLP, and what do the Shapley values sum to? (5.1)
6. Describe one impossible hybrid input created by the interventional value function in the case study. (5.3)
7. Why does default KernelSHAP return exact Shapley values for 7 features, and from how many features on does
   it sample? (6.2)
8. A ReLU has $z = -1$ at the reference and $z = 2$ at the instance. What is the rescale multiplier, and what
   is the gradient at the instance? (6.3)
9. Explain readings (a) and (b) of "Shapley values on the embeddings" and how they are connected. (7)
10. You double the number of explained days, then the background size, then the number of features from 7 to
    14. What happens to the runtime of each method? (8, 13)
11. Why did the maximum temperature of München on 29 June 2025 get +5.90 with one background and +2.47 with
    another, and which value is correct? (14.1)
