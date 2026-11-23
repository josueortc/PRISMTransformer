"""Non-negative CP decomposition with the modality factor on the probability simplex.

    T ~= sum_k lambda_k  a_k (x) b_k (x) ... (x) f_k,   all factors >= 0,  sum_m f_k(m) = 1

Fitted by multiplicative-update ALS. The last mode is the modality mode: after each of its
updates every column is divided by its sum and the scale moved into the first mode, so the
constraint holds inside every sweep (not only after the solver returns, which would only
relabel a fit made without it). Several random restarts are run and the lowest error kept.

Rectifying an attribution leaves a broad offset on every element that non-negative CP
cannot subtract; it is shared among the components and pulls modality allocations towards
the centre. ``fit`` therefore fits K + 1 components and drops the flattest one (lowest Gini
concentration of its motif) when it is clearly flatter than the others (< 0.75 x the median);
otherwise it returns a plain K-component fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from string import ascii_lowercase

import numpy as np

EPS = 1e-12


@dataclass
class CPResult:
    weights: np.ndarray           # [K] lambda
    factors: list[np.ndarray]     # per mode [dim, K]; L2-normalized except the last (sums to 1)
    error: float                  # ||T - T_hat|| / ||T||
    n_iter: int
    dropped_background: bool = False
    restart_errors: list[float] = field(default_factory=list)

    @property
    def rank(self) -> int:
        return len(self.weights)

    def motifs(self) -> np.ndarray:
        """[K, *dims] each component's rank-one tensor without lambda (for display/metrics)."""
        out = []
        for k in range(self.rank):
            m = self.factors[0][:, k]
            for f in self.factors[1:]:
                m = np.multiply.outer(m, f[:, k])
            out.append(m)
        return np.asarray(out)

    def reconstruct(self) -> np.ndarray:
        return np.tensordot(self.weights, self.motifs(), axes=1)

    def order(self) -> "CPResult":
        idx = np.argsort(-self.weights)
        return CPResult(self.weights[idx], [f[:, idx] for f in self.factors], self.error, self.n_iter,
                        self.dropped_background, self.restart_errors)

    def select(self, keep: np.ndarray) -> "CPResult":
        return CPResult(self.weights[keep], [f[:, keep] for f in self.factors], self.error, self.n_iter,
                        self.dropped_background, self.restart_errors)


def _mttkrp(T: np.ndarray, factors: list[np.ndarray], mode: int) -> np.ndarray:
    n = T.ndim
    letters = ascii_lowercase[:n]
    ins = [letters] + [letters[j] + "z" for j in range(n) if j != mode]
    ops = [T] + [factors[j] for j in range(n) if j != mode]
    return np.einsum(",".join(ins) + "->" + letters[mode] + "z", *ops, optimize=True)


def _gram(factors: list[np.ndarray], mode: int) -> np.ndarray:
    K = factors[0].shape[1]
    G = np.ones((K, K))
    for j, f in enumerate(factors):
        if j != mode:
            G *= f.T @ f
    return G


def _normalize(factors: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    K = factors[0].shape[1]
    lam = np.ones(K)
    out = []
    for j, f in enumerate(factors):
        s = f.sum(0) if j == len(factors) - 1 else np.linalg.norm(f, axis=0)
        s = np.where(s > EPS, s, 1.0)
        lam *= s
        out.append(f / s)
    return lam, out


def _error(T: np.ndarray, factors: list[np.ndarray], normT: float) -> float:
    lam = np.ones(factors[0].shape[1])
    full = np.zeros_like(T)
    for k in range(len(lam)):
        m = factors[0][:, k]
        for f in factors[1:]:
            m = np.multiply.outer(m, f[:, k])
        full += m
    return float(np.linalg.norm(T - full) / max(normT, EPS))


def _fit_once(T: np.ndarray, rank: int, rng: np.random.Generator, max_iter: int, tol: float) -> tuple[list[np.ndarray], float, int]:
    scale = (T.mean() / rank) ** (1.0 / T.ndim) if T.mean() > 0 else 1.0
    factors = [rng.uniform(0.1, 1.0, (d, rank)) * scale for d in T.shape]
    normT = float(np.linalg.norm(T))
    prev = np.inf
    err = np.inf
    it = 0
    last = T.ndim - 1
    for it in range(1, max_iter + 1):
        for mode in range(T.ndim):
            num = _mttkrp(T, factors, mode)
            den = factors[mode] @ _gram(factors, mode) + EPS
            factors[mode] = factors[mode] * num / den
            if mode == last:
                s = factors[last].sum(0)
                s = np.where(s > EPS, s, 1.0)
                factors[last] = factors[last] / s
                factors[0] = factors[0] * s
        if it % 10 == 0 or it == max_iter:
            err = _error(T, factors, normT)
            if np.isfinite(prev) and abs(prev - err) < tol * max(prev, 1e-12):
                break
            prev = err
    return factors, err, it


def nn_cp(T: np.ndarray, rank: int, *, restarts: int = 20, max_iter: int = 2000, tol: float = 1e-6,
          seed: int = 0) -> CPResult:
    """Plain non-negative CP (modality = last mode on the simplex), best of ``restarts``."""
    T = np.nan_to_num(np.asarray(T, dtype=float), nan=0.0)
    T = np.maximum(T, 0.0)
    rng = np.random.default_rng(seed)
    best, errors = None, []
    for _ in range(max(1, restarts)):
        factors, err, it = _fit_once(T, rank, rng, max_iter, tol)
        errors.append(err)
        if best is None or err < best[1]:
            best = (factors, err, it)
    lam, factors = _normalize(best[0])
    return CPResult(lam, factors, best[1], best[2], False, errors).order()


def gini(x: np.ndarray) -> float:
    """Gini concentration of a non-negative vector (0 = flat, -> 1 = concentrated)."""
    v = np.sort(np.abs(np.ravel(x)))
    n = v.size
    if n == 0 or v.sum() <= 0:
        return 0.0
    i = np.arange(1, n + 1)
    return float((2 * (i * v).sum()) / (n * v.sum()) - (n + 1) / n)


def fit(T: np.ndarray, rank: int, *, restarts: int = 20, max_iter: int = 2000, tol: float = 1e-6, seed: int = 0,
        background: bool = True) -> CPResult:
    """Rank-K fit, removing a flat background component when there is one (see module notes)."""
    if not background:
        return nn_cp(T, rank, restarts=restarts, max_iter=max_iter, tol=tol, seed=seed)
    wide = nn_cp(T, rank + 1, restarts=restarts, max_iter=max_iter, tol=tol, seed=seed)
    conc = np.array([gini(m) for m in wide.motifs()])
    flat = int(np.argmin(conc))
    if conc[flat] < 0.75 * np.median(conc):
        keep = np.array([k for k in range(wide.rank) if k != flat])
        out = wide.select(keep)
        out.dropped_background = True
        return out
    return nn_cp(T, rank, restarts=restarts, max_iter=max_iter, tol=tol, seed=seed)


def rank_curve(T: np.ndarray, ranks: list[int], **kw) -> list[dict]:
    """Normalized error E_K for candidate ranks (no background removal, for comparability)."""
    return [{"rank": k, "error": nn_cp(T, k, **kw).error} for k in ranks]


def choose_rank(curve: list[dict], min_gain: float = 0.02) -> int:
    """Smallest rank after which one more component lowers E_K by less than ``min_gain``."""
    for a, b in zip(curve, curve[1:]):
        if a["error"] - b["error"] < min_gain:
            return a["rank"]
    return curve[-1]["rank"]


def modality_call(weights: np.ndarray, threshold: float = 0.65) -> str:
    """Point rule: 'unique:<m>' when one modality holds more than ``threshold``, else 'shared'."""
    weights = np.asarray(weights, float)
    if weights.size < 2:
        return "unique:0"
    m = int(np.argmax(weights))
    return f"unique:{m}" if weights[m] > threshold else "shared"


def modality_call_interval(draws: np.ndarray, threshold: float = 0.65, level: float = 0.95) -> str:
    """From bootstrap draws [B, M] of one motif's allocation: unique when the interval of the
    leading modality clears the threshold, shared when the intervals include an even split,
    undetermined otherwise."""
    draws = np.asarray(draws, float)
    if draws.shape[1] < 2:
        return "unique:0"
    lo, hi = np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2], axis=0)
    m = int(np.argmax(draws.mean(0)))
    even = 1.0 / draws.shape[1]
    if lo[m] > threshold:
        return f"unique:{m}"
    if lo[m] <= even <= hi[m]:
        return "shared"
    return "undetermined"
