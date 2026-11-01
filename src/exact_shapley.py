"""Exact (brute-force) Shapley values — skeleton to implement yourself.

Formula (see CLAUDE.md §5.3):

    phi_i = sum_{S ⊆ N \\ {i}}  |S|! (M - |S| - 1)! / M!  *  [ v(S ∪ {i}) - v(S) ]

Two parts:
1. ``shapley_values(v, n_players)`` works for ANY cooperative game v(S).
   The tests in ``tests/test_exact_shapley.py`` check it with toy games whose
   answers are known, so you can verify your understanding before touching an MLP.
2. ``interventional_value_function(...)`` turns a trained model into a game:
   features in S are taken from the instance x, all other features are filled
   in from background rows, and the model outputs are averaged.

Run the tests with:  pytest
Every function currently raises NotImplementedError — that is expected.
"""

from __future__ import annotations

from itertools import combinations  # noqa: F401  (for you to use)
from math import factorial  # noqa: F401  (for you to use)
from typing import Callable, FrozenSet

import numpy as np

Coalition = FrozenSet[int]
ValueFunction = Callable[[Coalition], float]


def shapley_weight(coalition_size: int, n_players: int) -> float:
    """Weight |S|! (M - |S| - 1)! / M! of a coalition S that does not contain i.

    Hint: ``math.factorial`` is already imported.
    Check by hand: for M = 3 the weights for |S| = 0, 1, 2 should sum to 1
    when each is multiplied by the number of coalitions of that size.
    """
    # TODO: implement
    raise NotImplementedError


def shapley_values(v: ValueFunction, n_players: int) -> np.ndarray:
    """Exact Shapley value for every player 0..n_players-1 of the game v.

    Steps:
    - for each player i, iterate over all coalitions S of the OTHER players
      (hint: ``itertools.combinations`` over every size 0..M-1)
    - add  weight(|S|, M) * (v(S ∪ {i}) - v(S))
    - pass coalitions to v as ``frozenset`` objects

    Cost: M * 2^(M-1) evaluations of v -> only feasible for small M.
    """
    # TODO: implement
    raise NotImplementedError


def interventional_value_function(
    predict: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    background: np.ndarray,
) -> ValueFunction:
    """Build v(S) for one instance x of a model with a ``predict`` function.

    v(S) = mean over background rows b of  predict(z),  where
           z_j = x_j  if j in S,  else  b_j.

    So v(∅) = average prediction over the background (the baseline E[f(X)])
    and v(N) = f(x). Think about why that makes the efficiency axiom
    "sum of phi = f(x) - E[f(X)]" work out.

    Args:
        predict: maps an array of shape (n, M) to predictions of shape (n,)
                 (for a classifier choose explicitly: probability or logit!)
        x: the instance to explain, shape (M,)
        background: reference rows, shape (n_background, M)
    """
    # TODO: implement — return an inner function  def v(S): ...
    raise NotImplementedError


__all__ = [
    "shapley_weight",
    "shapley_values",
    "interventional_value_function",
]
