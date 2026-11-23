"""Does the model actually use what an attribution points to?

Generating structure (where signal was placed) is not the same as what a classifier reads.
These instruments measure use directly, on held-out trials:

* **occlusion**: set a set of elements to 0 (the training mean, since inputs are z-scored)
  and record the loss of classification margin, against occluding as many random elements
  (z score of the cost against that control);
* **interchange**: replace the same elements with those of a trial of the opposite class
  (several pairings), with a same-class swap as control. Unlike zeroing, this keeps the
  input on the data manifold;
* **block lesion map**: occlusion of every contiguous block (channels x time bins x 1
  modality), the in-silico analogue of silencing an area during a task epoch;
* **causal allocation agreement** and **false-attribution mass**: compare a method's mass
  on each pathway with the pathways' measured importance (no ground-truth motifs needed).

The margin of trial n is zeta_n * (+1 for class b, -1 for class a).
"""

from __future__ import annotations

import numpy as np
import torch

from prismt.data.tensors import TrialTensors, batches
from prismt.data.tokens import TokenGrid


def element_mask(rtm: np.ndarray, grid: TokenGrid) -> np.ndarray:
    """Boolean [channels, bins, modalities] -> [tokens, patch] on the model's input."""
    out = np.zeros((grid.n_tokens, grid.patch), dtype=bool)
    for q, (c, m) in enumerate(grid.pairs):
        for p in range(grid.n_patches):
            out[p * grid.n_pairs + q] = rtm[c, p * grid.patch:(p + 1) * grid.patch, m]
    return out


@torch.no_grad()
def margins(model, tensors: TrialTensors, rows: np.ndarray, y01: np.ndarray, classes: tuple[int, int], *,
            batch_size: int = 64, edit=None) -> np.ndarray:
    """Per-trial margins; ``edit(x, positions)`` may change a batch's inputs first."""
    model.eval()
    a, b = classes
    out = []
    for chunk in batches(np.arange(len(rows)), batch_size, shuffle=False):
        x, v, _ = tensors.batch(rows[chunk])
        if edit is not None:
            x = edit(x.clone(), chunk)
        logits = model(x, v).logits.float()
        zeta = (logits[:, b] - logits[:, a]).cpu().numpy()
        out.append(zeta * np.where(y01[chunk] == 1, 1.0, -1.0))
    return np.concatenate(out)


def _zero(mask: np.ndarray, device):
    m = torch.as_tensor(mask, device=device)

    def edit(x, _chunk):
        x[:, m] = 0.0
        return x
    return edit


def occlusion(model, tensors, rows, y01, classes, mask: np.ndarray, *, available: np.ndarray | None = None,
              n_random: int = 20, seed: int = 0, base: np.ndarray | None = None) -> dict:
    """Margin cost of zeroing ``mask`` ([tokens, patch]) against random masks of the same size."""
    device = tensors.device
    base = margins(model, tensors, rows, y01, classes) if base is None else base
    cost = float((base - margins(model, tensors, rows, y01, classes, edit=_zero(mask, device))).mean())
    pool = np.flatnonzero((np.ones_like(mask) if available is None else available).ravel())
    k = int(mask.sum())
    rng = np.random.default_rng(seed)
    ctrl = []
    for _ in range(n_random):
        rnd = np.zeros(mask.size, dtype=bool)
        rnd[rng.choice(pool, size=min(k, len(pool)), replace=False)] = True
        rnd = rnd.reshape(mask.shape)
        ctrl.append(float((base - margins(model, tensors, rows, y01, classes, edit=_zero(rnd, device))).mean()))
    ctrl = np.asarray(ctrl)
    sd = float(ctrl.std(ddof=1)) if len(ctrl) > 1 else 0.0
    return {"cost": cost, "control_mean": float(ctrl.mean()), "control_sd": sd,
            "z": float((cost - ctrl.mean()) / sd) if sd > 0 else None, "n_elements": k}


def occlusion_pooled(folds: list[tuple], classes: tuple[int, int], mask: np.ndarray, *,
                     available: np.ndarray | None = None, n_random: int = 20, seed: int = 0) -> dict:
    """Occlusion of one mask over several folds, each with its own model.

    ``folds``: (model, tensors, rows, y01) per fold. The cost is the mean loss of margin over
    all held-out trials; the control occludes as many randomly chosen elements (the same
    random masks in every fold), and z compares the cost with that control."""
    n = sum(len(f[2]) for f in folds)
    pool = np.flatnonzero((np.ones_like(mask) if available is None else available).ravel())
    k = int(mask.sum())
    rng = np.random.default_rng(seed)
    randoms = []
    for _ in range(n_random):
        rnd = np.zeros(mask.size, dtype=bool)
        rnd[rng.choice(pool, size=min(k, len(pool)), replace=False)] = True
        randoms.append(rnd.reshape(mask.shape))
    cost, base_sum, ctrl = 0.0, 0.0, np.zeros(n_random)
    for model, tensors, rows, y01 in folds:
        base = margins(model, tensors, rows, y01, classes)
        base_sum += float(base.sum())
        cost += float((base - margins(model, tensors, rows, y01, classes, edit=_zero(mask, tensors.device))).sum())
        for j, rnd in enumerate(randoms):
            ctrl[j] += float((base - margins(model, tensors, rows, y01, classes,
                                             edit=_zero(rnd, tensors.device))).sum())
    cost, ctrl, base_mean = cost / n, ctrl / n, base_sum / n
    sd = float(ctrl.std(ddof=1)) if n_random > 1 else 0.0
    z = float((cost - ctrl.mean()) / sd) if sd > 0 else None
    return {"cost": cost, "control_mean": float(ctrl.mean()), "control_sd": sd, "z": z, "n_elements": k,
            "margin": base_mean, "fraction_of_margin": float(cost / base_mean) if base_mean > 0 else None}


def interchange(model, tensors, rows, y01, classes, mask: np.ndarray, *, pairings: int = 5, seed: int = 0,
                base: np.ndarray | None = None) -> dict:
    """Margin cost of swapping ``mask`` in from opposite-class trials (and from same-class trials)."""
    rows = np.asarray(rows)
    base = margins(model, tensors, rows, y01, classes) if base is None else base
    rng = np.random.default_rng(seed)
    m = torch.as_tensor(mask, device=tensors.device)
    by_class = {c: np.flatnonzero(y01 == c) for c in (0, 1)}
    costs = {"opposite": [], "same": []}
    for _ in range(pairings):
        for kind in costs:
            partner = np.empty(len(rows), dtype=int)
            for c in (0, 1):
                own = by_class[c]
                src = by_class[1 - c] if kind == "opposite" else own
                partner[own] = rng.choice(src, size=len(own), replace=True)

            def edit(x, chunk, partner=partner):
                xp, _, _ = tensors.batch(rows[partner[chunk]])
                x[:, m] = xp[:, m]
                return x
            costs[kind].append(float((base - margins(model, tensors, rows, y01, classes, edit=edit)).mean()))
    return {"cost": float(np.mean(costs["opposite"])), "control": float(np.mean(costs["same"]))}


def block_lesion_map(model, tensors, rows, y01, classes, grid: TokenGrid, shape: tuple[int, int, int], *,
                     block: tuple[int, int] = (4, 2)) -> np.ndarray:
    """Margin cost of occluding each (channels x bins x 1 modality) block, as a [C, T, M] map
    (every element of a block gets the block's cost)."""
    C, T, M = shape
    base = margins(model, tensors, rows, y01, classes)
    out = np.full(shape, np.nan)
    for m in range(M):
        for c0 in range(0, C, block[0]):
            for t0 in range(0, T, block[1]):
                rtm = np.zeros(shape, dtype=bool)
                rtm[c0:c0 + block[0], t0:t0 + block[1], m] = True
                mask = element_mask(rtm, grid)
                if not mask.any():
                    continue
                cost = float((base - margins(model, tensors, rows, y01, classes,
                                             edit=_zero(mask, tensors.device))).mean())
                out[c0:c0 + block[0], t0:t0 + block[1], m] = cost
    return out


def pathway_importance(occlusions: list[dict]) -> np.ndarray:
    """Normalized importance c_k from occlusion z scores (negative z counts as unused)."""
    z = np.array([max(o["z"] or 0.0, 0.0) for o in occlusions])
    return z / z.sum() if z.sum() > 0 else np.full(len(z), 1.0 / len(z))


def mass_shares(score_map: np.ndarray, supports: np.ndarray) -> np.ndarray:
    """Size-corrected share of a method's mass on each pathway's support."""
    s = np.abs(score_map)
    dens = np.array([s[sup].sum() / max(sup.sum(), 1) for sup in supports])
    return dens / dens.sum() if dens.sum() > 0 else np.full(len(dens), 1.0 / len(dens))


def causal_scores(score_map: np.ndarray, supports: np.ndarray, importance: np.ndarray) -> dict:
    """Causal allocation agreement 1 - 0.5 sum|m - c| and false-attribution mass sum_{c<0.05} m."""
    m = mass_shares(score_map, supports)
    return {"allocation_agreement": float(1 - 0.5 * np.abs(m - importance).sum()),
            "false_attribution_mass": float(m[importance < 0.05].sum()),
            "mass_shares": m.tolist()}
