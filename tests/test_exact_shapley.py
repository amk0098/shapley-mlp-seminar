"""Checks for src/exact_shapley.py using games with known Shapley values.

These fail with NotImplementedError until you implement the functions.
"""

import numpy as np
import pytest

from src.exact_shapley import (
    interventional_value_function,
    shapley_values,
    shapley_weight,
)


def test_weights_sum_to_one():
    # sum over sizes k of C(M-1, k) * weight(k, M) must be 1
    from math import comb

    for m in range(1, 7):
        total = sum(comb(m - 1, k) * shapley_weight(k, m) for k in range(m))
        assert total == pytest.approx(1.0)


def test_additive_game():
    # v(S) = sum of w_i  ->  phi_i = w_i
    w = [3.0, -1.0, 0.5, 2.0]
    phi = shapley_values(lambda S: sum(w[i] for i in S), len(w))
    assert phi == pytest.approx(w)


def test_glove_game():
    # player 0 owns a left glove, players 1 and 2 own right gloves.
    # A pair is worth 1.  Known result: phi = (2/3, 1/6, 1/6).
    def v(S):
        return 1.0 if 0 in S and (1 in S or 2 in S) else 0.0

    phi = shapley_values(v, 3)
    assert phi == pytest.approx([2 / 3, 1 / 6, 1 / 6])


def test_dummy_player_gets_zero():
    # player 2 never changes the value
    def v(S):
        return float((0 in S) * 2 + (1 in S) * (0 in S))

    phi = shapley_values(v, 3)
    assert phi[2] == pytest.approx(0.0)


def test_efficiency():
    def v(S):
        return len(S) ** 2 + (1 in S) * 3

    m = 4
    phi = shapley_values(v, m)
    assert phi.sum() == pytest.approx(v(frozenset(range(m))) - v(frozenset()))


def test_linear_model_interventional():
    # For a linear model f(x) = w·x + b with the interventional value function,
    # theory gives phi_i = w_i * (x_i - mean(background_i)).
    rng = np.random.default_rng(0)
    w, b = np.array([1.5, -2.0, 0.3]), 0.7
    background = rng.normal(size=(50, 3))
    x = np.array([1.0, 0.5, -1.0])

    v = interventional_value_function(lambda X: X @ w + b, x, background)
    phi = shapley_values(v, 3)
    assert phi == pytest.approx(w * (x - background.mean(axis=0)))
