"""Removing the transformer's own axes (layer, head) without averaging away the experiment.

Each layer-head pair is weighted by how much its attribution differs between the two
classes::

    D[l, h] = sum_i | mean(G[y=b]) - mean(G[y=a]) |[l, h, i],   w = D / sum(D)

and the per-trial tensor keeps the trial axis::

    G_tilde[n, i] = sum_{l,h} w[l, h] G[n, l, h, i]

The discriminative tensor, used when ground truth is defined over channel x time x modality
(the synthetic benchmark), is | mean(G_tilde[y=b]) - mean(G_tilde[y=a]) |.
"""

from __future__ import annotations

import numpy as np


def class_means(G: np.ndarray, y01: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Means over trials of class 0 and class 1 (y01 is 0/1)."""
    return G[y01 == 0].mean(0), G[y01 == 1].mean(0)


def head_weights(G: np.ndarray, y01: np.ndarray) -> np.ndarray:
    """G [n, L, H, ...] -> w [L, H], non-negative, summing to 1."""
    m0, m1 = class_means(G, y01)
    D = np.abs(m1 - m0).reshape(G.shape[1], G.shape[2], -1).sum(-1)
    total = D.sum()
    return D / total if total > 0 else np.full_like(D, 1.0 / D.size)


def reduce_heads(G: np.ndarray, w: np.ndarray) -> np.ndarray:
    """[n, L, H, ...] -> [n, ...]."""
    return np.tensordot(G, w, axes=([1, 2], [0, 1])) if G.ndim > 3 else (G * w).sum((1, 2))


def discriminative(Gt: np.ndarray, y01: np.ndarray) -> np.ndarray:
    m0, m1 = class_means(Gt, y01)
    return np.abs(m1 - m0)


def class_signed(Gt: np.ndarray, y01: np.ndarray) -> np.ndarray:
    """Signed class difference (class 1 minus class 0), for display."""
    m0, m1 = class_means(Gt, y01)
    return m1 - m0
