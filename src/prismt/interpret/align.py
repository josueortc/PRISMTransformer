"""Matching motifs across independent fits (folds, bootstrap draws) and building consensus motifs.

Components are matched one-to-one by the Hungarian algorithm on the similarity of their
non-trial factors (mean cosine over the spatial, temporal and modality profiles).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from prismt.interpret.cp import CPResult


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine between columns: a [d, K1], b [d, K2] -> [K1, K2]."""
    na = np.linalg.norm(a, axis=0, keepdims=True)
    nb = np.linalg.norm(b, axis=0, keepdims=True)
    return (a.T @ b) / np.maximum(na.T @ nb, 1e-12)


def similarity(ref: CPResult, other: CPResult, modes: list[int] | None = None) -> np.ndarray:
    """[K_ref, K_other] mean cosine over ``modes`` (default: every mode)."""
    modes = list(range(len(ref.factors))) if modes is None else modes
    return np.mean([_cos(ref.factors[m], other.factors[m]) for m in modes], axis=0)


def match(ref: CPResult, other: CPResult, modes: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """For each ref component, the index of its match in ``other`` (-1 if none) and the similarity."""
    S = similarity(ref, other, modes)
    r, c = linear_sum_assignment(-S)
    idx = np.full(ref.rank, -1)
    sim = np.zeros(ref.rank)
    idx[r] = c
    sim[r] = S[r, c]
    return idx, sim


def consensus(ref: CPResult, fits: list[CPResult], modes: list[int] | None = None) -> dict:
    """Align every fit to ``ref`` and average the matched factors.

    Returns the mean factors (per mode [dim, K]), the matched similarity per fit and
    component, and stability (mean similarity per component)."""
    modes_all = list(range(len(ref.factors)))
    sims = np.zeros((len(fits), ref.rank))
    sums = [np.zeros_like(f) for f in ref.factors]
    counts = np.zeros(ref.rank)
    weights = np.zeros(ref.rank)
    for i, fit in enumerate(fits):
        idx, sim = match(ref, fit, modes)
        sims[i] = sim
        for k, j in enumerate(idx):
            if j < 0:
                continue
            for m in modes_all:
                if fit.factors[m].shape[0] == ref.factors[m].shape[0]:
                    sums[m][:, k] += fit.factors[m][:, j]
            weights[k] += fit.weights[j]
            counts[k] += 1
    c = np.maximum(counts, 1)
    return {"factors": [s / c for s in sums], "weights": weights / c, "similarity": sims,
            "stability": sims.mean(0) if len(fits) else np.ones(ref.rank)}
