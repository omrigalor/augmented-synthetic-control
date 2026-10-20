"""Compact augmented synthetic-control estimator with placebo inference."""

from dataclasses import dataclass
import numpy as np


def simplex_projection(v):
    """Euclidean projection onto nonnegative weights summing to one."""
    v = np.asarray(v, dtype=float)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    rho = np.nonzero(u - cssv / (np.arange(len(u)) + 1) > 0)[0][-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(v - theta, 0)


@dataclass
class SyntheticControlResult:
    weights: np.ndarray
    counterfactual: np.ndarray
    effect: np.ndarray
    pre_rmse: float


def fit(treated, donors, intervention, ridge=1e-4, steps=4000):
    """Fit donor weights on pre-treatment outcomes, then apply a ridge residual correction."""
    y, X = np.asarray(treated, float), np.asarray(donors, float)
    if X.ndim != 2 or X.shape[1] != y.size:
        raise ValueError("donors must have shape (n_donors, n_periods)")
    if not 2 <= intervention < y.size:
        raise ValueError("intervention must leave pre- and post-treatment periods")
    A, b = X[:, :intervention].T, y[:intervention]
    lipschitz = 2 * (np.linalg.norm(A, 2) ** 2 + ridge)
    w = np.repeat(1 / X.shape[0], X.shape[0])
    for _ in range(steps):
        grad = 2 * A.T @ (A @ w - b) + 2 * ridge * w
        w = simplex_projection(w - grad / max(lipschitz, 1e-9))
    base = X.T @ w
    # Outcome-model augmentation: shrink the average pre-period residual and carry it forward.
    residual_shift = (b - base[:intervention]).sum() / (intervention + ridge)
    counterfactual = base + residual_shift
    effect = y - counterfactual
    return SyntheticControlResult(w, counterfactual, effect,
                                  float(np.sqrt(np.mean(effect[:intervention] ** 2))))


def in_space_placebos(panel, treated_index, intervention, ridge=1e-4):
    """Refit every unit as if treated and return post/pre RMSPE ratios."""
    panel = np.asarray(panel, float)
    ratios = []
    for i in range(panel.shape[0]):
        result = fit(panel[i], np.delete(panel, i, axis=0), intervention, ridge)
        pre = max(result.pre_rmse, 1e-9)
        post = np.sqrt(np.mean(result.effect[intervention:] ** 2))
        ratios.append(post / pre)
    treated_ratio = ratios[treated_index]
    p_value = sum(r >= treated_ratio for r in ratios) / len(ratios)
    return np.asarray(ratios), p_value
