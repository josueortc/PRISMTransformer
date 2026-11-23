"""Bootstrap intervals and label-permutation significance for PRISMt results.

* Bootstrap: held-out trials are resampled with replacement within each fold and class,
  and everything that depends on them (head weights, the decomposition) is recomputed.
  Intervals are percentile; a paired contrast between two methods uses the same draws,
  with a two-sided p of twice the smaller tail, floored at 1/B.
* Region significance: the statistic for channel c and modality m is
  S[c, m] = sum_t | mean(G_tilde | b) - mean(G_tilde | a) |; labels are permuted the way
  the label varies (whole sessions or subjects when it is constant within them), so the
  null keeps the data's structure. Channels with p < 0.05 are the PRISMt-selected regions;
  Benjamini-Hochberg q values are reported too.
"""

from __future__ import annotations

import numpy as np

from prismt.eval.baselines import permuted_labels


def bootstrap_indices(y01: np.ndarray, strata: np.ndarray | None, n_draws: int, seed: int = 0) -> list[np.ndarray]:
    """Index arrays resampled with replacement within each (stratum, class) cell."""
    rng = np.random.default_rng(seed)
    strata = np.zeros(len(y01), dtype=int) if strata is None else np.asarray(strata)
    cells = [np.flatnonzero((strata == s) & (y01 == c)) for s in np.unique(strata) for c in (0, 1)]
    cells = [c for c in cells if len(c)]
    return [np.concatenate([rng.choice(c, size=len(c), replace=True) for c in cells]) for _ in range(n_draws)]


def interval(values, level: float = 0.95) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if not len(v):
        return {"mean": None, "sd": None, "lo": None, "hi": None, "n": 0}
    lo, hi = np.quantile(v, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "lo": float(lo),
            "hi": float(hi), "n": int(len(v))}


def paired_p(a, b) -> float | None:
    """Two-sided bootstrap p for a - b on matched draws."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    if not len(d):
        return None
    tail = min((d <= 0).mean(), (d >= 0).mean())
    return float(max(2 * tail, 1.0 / len(d)))


def region_statistic(Gt: np.ndarray, y01: np.ndarray) -> np.ndarray:
    """[n, C, T, M] -> S [C, M]."""
    return np.abs(Gt[y01 == 1].mean(0) - Gt[y01 == 0].mean(0)).sum(1)


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q values (same shape as p)."""
    flat = np.ravel(p)
    order = np.argsort(flat)
    n = len(flat)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, flat[i] * n / k)
        q[i] = prev
    return q.reshape(np.shape(p))


def region_permutation(Gt: np.ndarray, y01: np.ndarray, groups: np.ndarray | None, *, n_perm: int = 1000,
                       seed: int = 0, exists: np.ndarray | None = None) -> dict:
    """p and q values per (channel, modality) for the region statistic."""
    rng = np.random.default_rng(seed)
    obs = region_statistic(Gt, y01)
    ge = np.zeros_like(obs)
    for _ in range(n_perm):
        yp = permuted_labels(y01, groups, rng)
        if yp.min() == yp.max():
            continue
        ge += region_statistic(Gt, yp) >= obs
    p = (1 + ge) / (1 + n_perm)
    if exists is not None:
        p = np.where(exists, p, np.nan)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    q[ok] = bh(p[ok])
    return {"statistic": obs, "p": p, "q": q, "n_perm": int(n_perm)}
