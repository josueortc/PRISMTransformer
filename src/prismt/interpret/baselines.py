"""The interpretation methods PRISMt is compared with, each reduced to an importance map on the
same channel x time x modality axes (and to K motifs, for motif-level scores).

Model-internal:
* ``rollout``: attention rollout without value messages, class-differenced like PRISMt
  (isolates what the signed messages and the head weighting add);
* ``input_gradient``: gradient x input, class-differenced.

Observation-level (use only the data, never the model):
* ``class_difference``: | mean(X | b) - mean(X | a) |;
* ``pca``: leading principal components of the trials;
* ``nmf``: non-negative matrix factorization of the (shifted) trials;
* ``cp_observations``: non-negative CP of the class difference;
* ``logistic``: absolute coefficients of an L2 logistic regression.
"""

from __future__ import annotations

import warnings

import numpy as np

from prismt.interpret import cp as cpmod
from prismt.interpret.reduce import discriminative

OBSERVATION_METHODS = ("class_difference", "pca", "nmf", "cp_observations", "logistic")
MODEL_METHODS = ("rollout", "input_gradient")


def _motifs_from_map(score_map: np.ndarray, rank: int, seed: int, restarts: int) -> np.ndarray:
    res = cpmod.fit(np.abs(score_map), rank, restarts=restarts, seed=seed)
    return res.motifs() * res.weights[:, None, None, None]


def observation_maps(X: np.ndarray, y01: np.ndarray, rank: int, *, seed: int = 0, restarts: int = 10) -> dict:
    """X [n, C, T, M] (missing = 0). Returns name -> {"map": [C,T,M], "motifs": [K,C,T,M]}."""
    n = X.shape[0]
    shape = X.shape[1:]
    flat = X.reshape(n, -1)
    out = {}
    diff = discriminative(X, y01)
    out["class_difference"] = {"map": diff, "motifs": _motifs_from_map(diff, rank, seed, restarts)}
    out["cp_observations"] = out["class_difference"]
    centred = flat - flat.mean(0)
    u, s, vt = np.linalg.svd(centred, full_matrices=False)
    k = min(rank, len(s))
    pcs = np.abs(vt[:k]) * s[:k, None] / np.sqrt(max(n - 1, 1))
    out["pca"] = {"map": pcs.sum(0).reshape(shape), "motifs": pcs.reshape(k, *shape)}
    from sklearn.decomposition import NMF
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nmf = NMF(n_components=rank, init="nndsvda", max_iter=500, random_state=seed)
        nmf.fit(flat - flat.min())
    H = nmf.components_
    out["nmf"] = {"map": H.sum(0).reshape(shape), "motifs": H.reshape(rank, *shape)}
    from sklearn.linear_model import LogisticRegression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(flat, y01)
    coef = np.abs(clf.coef_[0]).reshape(shape)
    out["logistic"] = {"map": coef, "motifs": _motifs_from_map(coef, rank, seed, restarts)}
    return out


def model_maps(rollout_rtm: np.ndarray, grad_rtm: np.ndarray, y01: np.ndarray, rank: int, *, seed: int = 0,
               restarts: int = 10) -> dict:
    out = {}
    for name, per_trial in (("rollout", rollout_rtm), ("input_gradient", grad_rtm)):
        d = discriminative(per_trial, y01)
        out[name] = {"map": d, "motifs": _motifs_from_map(d, rank, seed, restarts)}
    return out
