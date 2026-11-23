"""Recovery scores against known ground truth (the synthetic benchmark).

* motif correlation: mean |Pearson r| between recovered and generating motifs, flattened
  over (channel, time, modality), after one-to-one Hungarian matching;
* unit-support AUROC: the union of the generating pathways' support masks as labels, the
  summed absolute recovered motifs (or any importance map) as scores; per-pathway AUROC
  scores each mask alone;
* mass enrichment: share of mass inside the support divided by the support's share of
  elements (1 for a uniform map, so it cannot be earned by a flat solution);
* factor match: cosine agreement of the spatial, temporal and modality marginals separately;
* modality allocation error: L1 distance between recovered and generating allocations.

Ground truth means the pathways that generate the label as the classifier reads it. A
pathway that only correlates with the label in the scored trials, without being used by the
model, is excluded from the support and scored by the causal instruments in
:mod:`faithfulness` instead.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score, roc_auc_score


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.ravel(a) - np.mean(a), np.ravel(b) - np.mean(b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(abs(a @ b) / d) if d > 0 else 0.0


def match_motifs(recovered: np.ndarray, truth: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Hungarian matching on |r|. Returns (truth index, recovered index) pairs and the r matrix."""
    R = np.array([[_corr(t, r) for r in recovered] for t in truth])
    ti, ri = linear_sum_assignment(-R)
    return list(zip(ti.tolist(), ri.tolist())), R


def motif_correlation(recovered: np.ndarray, truth: np.ndarray) -> float:
    """Mean matched |r| over the generating motifs (an unmatched one counts as 0)."""
    pairs, R = match_motifs(recovered, truth)
    return float(sum(R[t, r] for t, r in pairs) / len(truth))


def auroc(score_map: np.ndarray, support: np.ndarray) -> float | None:
    y = np.ravel(support).astype(bool)
    if y.all() or not y.any():
        return None
    return float(roc_auc_score(y, np.ravel(np.abs(score_map))))


def auprc(score_map: np.ndarray, support: np.ndarray) -> float | None:
    y = np.ravel(support).astype(bool)
    if y.all() or not y.any():
        return None
    return float(average_precision_score(y, np.ravel(np.abs(score_map))))


def mass_enrichment(score_map: np.ndarray, support: np.ndarray) -> float | None:
    s = np.abs(np.ravel(score_map))
    y = np.ravel(support).astype(bool)
    if s.sum() <= 0 or not y.any():
        return None
    return float((s[y].sum() / s.sum()) / (y.sum() / y.size))


def factor_match(recovered: np.ndarray, truth: np.ndarray) -> dict:
    """Cosine of each mode's marginal profile for matched motifs ([K, R, T, M] arrays)."""
    pairs, _ = match_motifs(recovered, truth)
    out = {"spatial": [], "temporal": [], "modality": []}
    for t, r in pairs:
        for name, axes in (("spatial", (1, 2)), ("temporal", (0, 2)), ("modality", (0, 1))):
            a = np.abs(truth[t]).sum(axes)
            b = np.abs(recovered[r]).sum(axes)
            d = np.linalg.norm(a) * np.linalg.norm(b)
            out[name].append(float(a @ b / d) if d > 0 else 0.0)
    return {k: float(np.mean(v)) if v else None for k, v in out.items()}


def modality_allocation_error(recovered: np.ndarray, truth: np.ndarray) -> float:
    """Mean L1 distance between allocations of matched motifs."""
    pairs, _ = match_motifs(recovered, truth)
    errs = []
    for t, r in pairs:
        a = np.abs(truth[t]).sum((0, 1))
        b = np.abs(recovered[r]).sum((0, 1))
        errs.append(float(np.abs(a / max(a.sum(), 1e-12) - b / max(b.sum(), 1e-12)).sum()))
    return float(np.mean(errs)) if errs else float("nan")


def recovery(recovered_motifs: np.ndarray, truth_motifs: np.ndarray, supports: np.ndarray,
             score_map: np.ndarray | None = None) -> dict:
    """Every recovery score for one method. supports [K_truth, R, T, M] booleans (label-generating
    pathways only); score_map defaults to the summed absolute motifs."""
    if score_map is None:
        score_map = np.abs(recovered_motifs).sum(0)
    union = supports.any(0)
    return {
        "motif_correlation": motif_correlation(recovered_motifs, truth_motifs),
        "unit_auroc": auroc(score_map, union),
        "unit_auprc": auprc(score_map, union),
        "pathway_auroc": [auroc(score_map, s) for s in supports],
        "mass_enrichment": mass_enrichment(score_map, union),
        "factor_match": factor_match(recovered_motifs, truth_motifs),
        "modality_error": modality_allocation_error(recovered_motifs, truth_motifs),
    }


def flat_control(shape: tuple[int, ...], truth_motifs: np.ndarray, supports: np.ndarray) -> dict:
    """The scores a uniform map earns: what a method must clearly beat."""
    flat = np.ones((1, *shape))      # every element tied: AUROC 0.5, enrichment 1, correlation 0
    return recovery(flat, truth_motifs, supports)
