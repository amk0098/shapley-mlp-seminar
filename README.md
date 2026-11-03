# Shapley-Analyse eines MLP

Seminar project — Bergische Universität Wuppertal (supervisor: Mathias Sure), talk on 1 Dec 2026.

**Central question:** Wie lassen sich die Vorhersagen eines MLP mithilfe von Shapley-Werten
erklären — und wo liegen die Grenzen dieser Erklärungen?

Full project context, roadmap, literature and open questions: [CLAUDE.md](CLAUDE.md).

## Structure

| Path | Content |
|---|---|
| `CLAUDE.md` | Project context, roadmap, literature, talk outline |
| `src/exact_shapley.py` | Exact (brute-force) Shapley values — skeleton with TODOs |
| `tests/test_exact_shapley.py` | Toy games with known answers to check the implementation |
| `notebooks/demo.ipynb` | Demo scaffold: data → MLP → exact Shapley → KernelSHAP → plots |
| `figures/` | Exported plots for the slides |
| `data/` | Nothing committed; datasets come from scikit-learn |
| `slides/` | Talk slides (30 min, German) and speaker notes |
| `report/` | Written Ausarbeitung (if required) and `references.bib` |
| `notes/` | Learning notes, literature notes, meeting notes |

## Setup

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
pytest
```

The tests fail with `NotImplementedError` until the functions in `src/exact_shapley.py`
are implemented — that is the first coding exercise.

```bash
jupyter notebook notebooks/demo.ipynb
```

## Status

Scaffold only. No experimental results yet. Scope (theory only vs. practical demo) is still
to be confirmed with the supervisor — see the open questions in CLAUDE.md §3.

Related (separate project): [heatrisk-shap-analysis](https://github.com/amk0098/heatrisk-shap-analysis).
