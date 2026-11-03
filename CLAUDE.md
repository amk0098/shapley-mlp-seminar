# CLAUDE.md — Seminar „Shapley-Analyse eines MLP“

> Project context file for Claude. Read this first in every session of this project.
> Last updated: 31 Oct 2026 · Presentation: **Tue, 1 Dec 2026** (year to be confirmed in Moodle)

---

## 1. Project in one sentence

Explain — and if required, demonstrate — how **Shapley values** attribute the prediction of a **Multi-Layer Perceptron (MLP)** to its input features, in a 30-minute German seminar talk at Bergische Universität Wuppertal (Lehrstuhl für Automatisierungstechnik/Informatik).

**Central question:** *Wie lassen sich die Vorhersagen eines MLP mithilfe von Shapley-Werten erklären — und wo liegen die Grenzen dieser Erklärungen?*

---

## 2. Who I am (Arian)

- B.Sc. Informatik, Uni Wuppertal, ~4th semester.
- Talk and slides must be in **German**.
- Background: C (pointers, structs, memory), moving to Python; R (`dplyr`, `ggplot2`); intro ML (regression, logistic regression, trees, clustering, basic neural nets).
- **Not yet proficient** in: Python ML stack, deep learning, probability theory, game theory.

### How Claude should help
- Role: **tutor, research assistant, critical reviewer** — not a ghostwriter.
- One concept at a time, intuition → small numeric example → formula (explain every symbol) → short comprehension check.
- Explain in simple English; produce natural academic German for slides, notes and emails. Keep English technical terms in parentheses.
- Be critical: correct misconceptions directly, challenge weak reasoning.
- **Never invent** citations, results, experiments or requirements. Mark assumptions clearly.
- Code: teach and debug; don't just replace my code with a finished solution.

---

## 3. Status

### Confirmed ✅
- Topic: **Shapley-Analyse eines MLP**, supervisor **Mathias Sure**.
- Presentation date **1 December** agreed by email (book in Moodle).
- Zoom kick-off **Thursday 11:00** accepted.
- Seminar format (from chair slides „Voodoo in wissenschaftlichen Vorträgen?“, April 2026): **30-minute talk**, mixed audience (ET, IT, Wi-Ing).

### Open ❓ (fill in after the Zoom meeting)
| Question | Answer |
|---|---|
| Moodle date booked? | |
| Zoom meeting done? Date? | |
| Theory only, or practical MLP + dataset? | |
| Python implementation mandatory? | |
| Prescribed dataset / model? | |
| Prescribed method (exact, KernelSHAP, DeepSHAP, …)? | |
| Recommended literature? | |
| Mathematical depth expected? | |
| Written Ausarbeitung required? Length? Deadline? | |
| Assessment focus? | |
| Q&A length after the talk? | |

**Rule:** Until this table is filled, treat the scope as *theory + small demo (likely)*, but don't start large implementation work.

---

## 4. Chair guidelines (from provided documents)

### Talk (Hohnhäuser, Musial, Lüdtke, Tutsch — Version April 2026)
- Structure: **Einleitung** (Begrüßung, Problem, Motivation, Terminologie, Fahrplan) → **Hauptteil** (Ansatz, Hauptidee, Bedeutung; small chunks with mini-summaries, no step-by-step detail) → **Zusammenfassung** (Kernaspekte, offene Probleme, Ausblick).
- **1 slide ≈ 1–3 min → ~12–18 slides** for 30 min (+ Reservefolien for Q&A).
- Slides: font ≥ 18 pt, sans-serif, one font, **3–5 points per slide**, page number on every slide, no Fließtext, no source code (exceptions), no large tables, formulas sparingly.
- Sources **directly on the slide**. *AI is a tool, not a source* — every claim needs a real reference.
- KISS, examples, speak (don't read), eye contact, keep time, don't apologise, **don't invent answers** in Q&A.
- Bring a PDF version; check laptop/beamer compatibility.

### Written work (Leitfaden für wissenschaftliche Arbeiten, John & Potthoff 2015) — if an Ausarbeitung is required
- Structure IMRAD-like: Einleitung → Grundlagen (incl. methods/tools) → eigene Arbeit → Analyse/Ergebnisse → Schlussbetrachtung.
- Citation: alphanumeric brackets, e.g. **[LL17]**, **[Sha53]**; paraphrases with „vgl.“; references per DIN 1505-2, sorted alphabetically.
- Figures/tables: „Abbildung <Kap>.<Nr>: Titel“, must be referenced in the text; source in caption; modified figures marked „(nach [XY00])“.
- German quotation marks „…“; 12 pt serif body text, 1.2–1.5 line spacing.
- Code: monospace, documented, full code attached digitally.

---

## 5. Core content

### 5.1 MLP (Multi-Layer Perceptron)
Input layer → hidden layer(s) → output layer; weights, biases, activation functions (ReLU, sigmoid); forward pass; training via loss + gradient descent / backpropagation; regression vs. classification. **Why it's a black box:** the prediction is a nested non-linear function of all inputs — no single weight explains it.

### 5.2 Explainable AI (XAI)
Interpretability vs. explainability · global vs. local explanations · feature importance vs. feature attribution · model-agnostic vs. model-specific methods.

### 5.3 Shapley values (cooperative game theory)
- Players = features, coalition = subset of features, payoff = model output.
- Shapley value of feature *i*:

  φᵢ = Σ_{S ⊆ N∖{i}} [ |S|! (M−|S|−1)! / M! ] · [ v(S ∪ {i}) − v(S) ]

  — the weighted average marginal contribution of *i* over all coalitions (equivalently: averaged over all M! orderings of the features).
- Axioms: **Efficiency** (Σφᵢ = f(x) − E[f(X)]), **Symmetry**, **Dummy/Null player**, **Additivity/Linearity**.
- Key interpretation: φᵢ explains the gap between **this prediction** and the **baseline (expected output over the background data)**.

### 5.4 Applying it to an MLP — the crucial design choice
An MLP can't "drop" a feature, so **v(S) must be defined**: features outside S are replaced by values from a **background/reference dataset**.
- *Interventional* (marginal) vs. *observational* (conditional) value function → different Shapley values ([JLL21], [SN20], [AJL21]).
- Cost: 2^M coalitions → exact computation only feasible for small M.
- Methods:
  - **Exact** (brute force, small M) — good for teaching/verification.
  - **Sampling / permutation** approximations ([ŠK14]).
  - **KernelSHAP** — model-agnostic, weighted linear regression ([LL17]).
  - **DeepSHAP / DeepExplainer** — DeepLIFT-based, fast for neural nets, approximation ([LL17], [SGK17]).
  - (GradientExplainer — expected gradients; optional.)
- SHAP is a **family of methods**, not one exact algorithm.

### 5.5 Limitations (must appear in the talk)
Computational cost · approximation error · correlated features (off-manifold samples) · sensitivity to background data · local ≠ global · instability · **attribution ≠ causation** ([KVS+20]).

---

## 6. Literature (verify every entry before citing)

| Key | Reference | Use |
|---|---|---|
| [Sha53] | Shapley, L. S.: „A Value for n-Person Games“. In: *Contributions to the Theory of Games II*, Princeton University Press, 1953, S. 307–317 | Original definition |
| [ŠK14] | Štrumbelj, E.; Kononenko, I.: „Explaining prediction models and individual predictions with feature contributions“. In: *Knowledge and Information Systems* 41 (2014), Nr. 3 | Shapley for ML, sampling |
| [LL17] | Lundberg, S. M.; Lee, S.-I.: „A Unified Approach to Interpreting Model Predictions“. In: *Advances in Neural Information Processing Systems 30 (NeurIPS)*, 2017 | SHAP, KernelSHAP, DeepSHAP — **main paper** |
| [SGK17] | Shrikumar, A.; Greenside, P.; Kundaje, A.: „Learning Important Features Through Propagating Activation Differences“. In: *Proc. ICML*, 2017 | DeepLIFT (basis of DeepSHAP) |
| [SN20] | Sundararajan, M.; Najmi, A.: „The Many Shapley Values for Model Explanation“. In: *Proc. ICML*, 2020 | Different value functions |
| [KVS+20] | Kumar, I. E.; Venkatasubramanian, S.; Scheidegger, C.; Friedler, S.: „Problems with Shapley-value-based explanations as feature importance measures“. In: *Proc. ICML*, 2020 | Critical discussion |
| [AJL21] | Aas, K.; Jullum, M.; Løland, A.: „Explaining individual predictions when features are dependent“. In: *Artificial Intelligence* 298 (2021) | Correlated features |
| [JLL21] | Janzing, D.; Minorics, L.; Blöbaum, P.: „Feature relevance quantification in explainable AI: A causal problem“. In: *Proc. AISTATS*, 2020 | Interventional vs. conditional (check key/year) |
| [Mol22] | Molnar, C.: *Interpretable Machine Learning*. 2. Aufl., 2022 (christophm.github.io/interpretable-ml-book) | Accessible overview — background, cite sparingly |
| — | SHAP documentation (shap.readthedocs.io) | Implementation |

Supervisor's recommendations override this list. Reading order: Molnar (Shapley + SHAP chapters) → [LL17] → [ŠK14] → [KVS+20] / [AJL21].

---

## 7. Possible practical demo (only if confirmed)

Goal: small, reproducible, explainable — not a big ML project.

1. **Dataset:** small tabular, few features (e.g. scikit-learn *California Housing* — 8 features, regression; or *Breast Cancer* — classification). Standardise features.
2. **Model:** `sklearn.neural_network.MLPRegressor` (1–2 hidden layers) — or PyTorch if DeepSHAP is required. Report test score.
3. **Exact Shapley by hand** (brute force over 2^M coalitions, interventional with background sample) for one prediction — shows I understand the formula.
4. **KernelSHAP** (and DeepSHAP if PyTorch) on the same instance → compare with exact values; check efficiency property (sum = f(x) − baseline).
5. **Plots:** waterfall (one prediction), beeswarm/summary (global view), maybe dependence plot.
6. **Experiments for discussion:** vary background size / content → how stable are the values? effect of correlated features.
7. Fixed random seeds, `requirements.txt`, one notebook or script.

Repo layout:
```
Shapley-analyse-seminar/
├── CLAUDE.md                 # this file (single source of project context)
├── README.md
├── requirements.txt          # pinned versions
├── pytest.ini
├── data/                     # datasets are loaded via sklearn; nothing committed
├── notebooks/demo.ipynb      # demo scaffold (sections only, filled step by step)
├── src/exact_shapley.py      # skeleton with TODOs — Arian implements
├── tests/test_exact_shapley.py  # toy games with known answers to check the implementation
└── figures/                  # exported plots for the slides
```

---

## 8. Draft talk outline (≈ 30 min, ~15 slides) — provisional

| # | Folie | min |
|---|---|---|
| 1 | Titel | 0.5 |
| 2 | Motivation: Black-Box-Modelle, Warum erklären? | 2 |
| 3 | Fahrplan | 0.5 |
| 4 | MLP in einer Folie | 2.5 |
| 5 | XAI: lokal vs. global, Attribution | 2 |
| 6 | Kooperative Spieltheorie: Spieler, Koalitionen, Auszahlung | 2.5 |
| 7 | Shapley-Wert: Formel + Intuition | 3 |
| 8 | Rechenbeispiel mit 3 Merkmalen | 3 |
| 9 | Axiome (Effizienz, Symmetrie, Dummy, Additivität) | 1.5 |
| 10 | Von Spieltheorie zum MLP: Wertfunktion & Hintergrunddaten | 2.5 |
| 11 | Berechnung: exakt vs. KernelSHAP vs. DeepSHAP | 2.5 |
| 12 | Demo / Ergebnisse (Waterfall + Summary-Plot) | 3 |
| 13 | Grenzen & kritische Diskussion | 2.5 |
| 14 | Zusammenfassung & Ausblick | 1.5 |
| 15 | Quellen | — |
| R | Reservefolien: Komplexität, Korrelation, Formelherleitung | — |

---

## 9. Roadmap (31 Oct → 1 Dec 2026)

### Phase 0 — Klärung (Sa 31.10. – Do 05.11.)
- [ ] Book 1 Dec in Moodle; check chair's official schedule (weekly from now on).
- [ ] Confirm Zoom invite (Thursday 11:00) — fill table in §3 right after the meeting.
- [ ] Download Testatbogen.
- [ ] Learn: MLP fundamentals (forward pass, activations, training) + XAI basics.
- ✔ Checkpoint: I can explain an MLP and "local attribution" in 2 minutes in German.

### Phase 1 — Theorie (Fr 06.11. – Do 12.11.)
- [ ] Cooperative game theory + Shapley formula; compute a 3-player example **by hand**.
- [ ] Axioms and the efficiency property.
- [ ] Read Molnar chapters + [LL17] (Sections on SHAP, KernelSHAP, DeepSHAP).
- [ ] Build source list with full bibliographic data (verified).
- ✔ Checkpoint: I can derive every symbol of the formula and explain the baseline.

### Phase 2 — Anwendung & Demo (Fr 13.11. – Do 19.11.)
- [ ] (If practical work confirmed) Python setup, dataset, train MLP.
- [ ] Exact Shapley implementation for small M; KernelSHAP/DeepSHAP comparison.
- [ ] Figures (vector/PNG high-res), background-sensitivity experiment.
- [ ] Read critical papers [KVS+20], [AJL21]/[SN20].
- ✔ Checkpoint: results reproducible; I can explain what each plot does *and does not* mean.
- Optional: short email update/questions to Mathias Sure.

### Phase 3 — Folien (Fr 20.11. – Do 26.11.)
- [ ] Slides per §8 and chair rules (≥18 pt, 3–5 points, sources on slide, page numbers).
- [ ] German speaker notes (keywords, not a script to read).
- [ ] First full dry run with timer → cut to 30 min.

### Phase 4 — Proben & Fragen (Fr 27.11. – Mo 30.11.)
- [ ] 2–3 rehearsals (out loud, standing, timed); record once.
- [ ] Q&A prep: 10–15 likely questions with honest answers (see §10).
- [ ] Export PDF, test on another device, backup on USB.

### Di 01.12. — Vortrag 🎤
Bring Testatbogen, PDF + backup.

### Afterwards (if required)
- [ ] Schriftliche Ausarbeitung per §4 (deadline: ___).

---

## 10. Likely questions to prepare
1. Warum nicht einfach die Gewichte des MLP interpretieren?
2. Was genau ist der Baseline-Wert, und wie wählt man die Hintergrunddaten?
3. Warum ist die exakte Berechnung exponentiell teuer?
4. Unterschied KernelSHAP vs. DeepSHAP — welche Annahmen?
5. Was passiert bei korrelierten Merkmalen?
6. Bedeutet ein hoher Shapley-Wert Kausalität?
7. Sind die Werte stabil, wenn man das Modell neu trainiert?
8. Unterschied zu Permutation Importance / LIME / Gradienten?
9. Was bedeutet ein negativer Shapley-Wert?
10. Warum genau diese Gewichtung |S|!(M−|S|−1)!/M!?

---

## 11. Glossary (DE / EN)
| Deutsch | English |
|---|---|
| Merkmal | feature |
| Vorhersage | prediction |
| Koalition | coalition |
| Grenzbeitrag | marginal contribution |
| Hintergrunddaten | background data |
| Erwartungswert | expected value |
| Merkmalsattribution | feature attribution |
| Erklärbarkeit | explainability |
| verdeckte Schicht | hidden layer |
| Aktivierungsfunktion | activation function |

---

## 12. Session log
| Date | What was done | Next step |
|---|---|---|
| 31.10.2026 | CLAUDE.md created | Book date, Zoom meeting, start MLP basics |
| 01.11.2026 | Repo cleaned: old instruction files replaced by CLAUDE.md; scaffold (src skeleton, tests, notebook, pinned requirements) added | Book date, Zoom meeting, start MLP basics |
