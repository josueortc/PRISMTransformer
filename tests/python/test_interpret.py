"""PRISMt interpretation: attribution, reduction, decomposition, alignment, statistics."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import torch

from prismt.data.tokens import TokenGrid
from prismt.interpret import align, attribution, cp, metrics, stats
from prismt.interpret.reduce import discriminative, head_weights, reduce_heads
from prismt.model.prismt_model import ModelSpec, PrismtModel


def _model(n_pairs=3, n_patches=2, layers=2, heads=2, d=8, seed=0):
    torch.manual_seed(seed)
    spec = ModelSpec("classify", n_pairs, n_patches, 1, d, layers, heads, 2, 0.0, "block_causal", "channel_time", 2)
    return PrismtModel(spec).eval()


# --- routing --------------------------------------------------------------------------------

def test_rollout_composes_later_layers_on_the_left():
    rng = np.random.default_rng(0)
    S, H = 4, 2
    A = [torch.as_tensor(rng.dirichlet(np.ones(S), size=(1, H, S)), dtype=torch.float64) for _ in range(2)]
    rho, full = attribution.rollout_routing(A, gamma=0.1)
    hat = lambda a: (a + 0.1 * np.eye(S)) / 1.1  # noqa: E731
    m = [hat(a[0].numpy().mean(0)) for a in A]
    expected_full = (m[1] @ m[0])[0]
    assert np.allclose(full[0].numpy(), expected_full)
    # layer 1 (the first), head 1: A_2(mean) A_1(head 1)
    assert np.allclose(rho[0, 0, 1].numpy(), (m[1] @ hat(A[0][0, 1].numpy()))[0])
    # layer 2, head 0: A_2(head 0) A_1(mean)
    assert np.allclose(rho[0, 1, 0].numpy(), (hat(A[1][0, 0].numpy()) @ m[0])[0])
    # the reverse order is a different quantity
    assert not np.allclose(expected_full, (m[0] @ m[1])[0])


def test_messages_are_the_heads_contributions_to_the_decision():
    """sum_j A[CLS, j] msg[j] equals u . (what that head writes into the CLS residual)."""
    model = _model()
    x = torch.randn(3, 6, 1)
    v = torch.ones(3, 6, dtype=torch.bool)
    out = model(x, v, need_weights=True)
    u = torch.randn(model.spec.d_model)
    blk = model.encoder.blocks[-1]
    W_O = blk.attn.per_head_out_proj()
    A, vals = out.internals["attention"][-1], out.internals["values"][-1]
    msg = torch.einsum("bhsk,hkd,d->bhs", vals, W_O, u)
    via_msg = (A[:, :, 0, :] * msg).sum(-1)                   # [B, H]
    ctx = A @ vals                                            # [B, H, S, d_head]
    head_out = torch.einsum("bhk,hkd->bhd", ctx[:, :, 0], W_O)
    assert torch.allclose(via_msg, head_out @ u, atol=1e-5)


def test_attribution_shapes_and_token_mapping():
    model = _model()
    grid = TokenGrid.make([(0, 0), (1, 0), (0, 1)], 2, 1)
    from prismt.data.tensors import TrialTensors
    X = np.random.default_rng(0).normal(size=(10, 6, 1)).astype(np.float32)
    y = np.array([0, 1] * 5)
    t = TrialTensors.make(X, np.ones((10, 6), bool), y, torch.device("cpu"))
    att = attribution.compute(model, t, np.arange(10), batch_size=4)
    assert att.G.shape == (10, 2, 2, 6) and np.isfinite(att.G).all()
    rtm = attribution.to_rtm(att.G.sum((1, 2)), grid, 2, 2)
    assert rtm.shape == (10, 2, 2, 2)
    assert np.all(rtm[:, 1, :, 1] == 0)                        # pair (channel 1, modality 1) does not exist
    assert attribution.exists_mask(grid, 2, 2).tolist() == [[True, True], [True, False]]
    assert np.isclose(np.linalg.norm(att.u), 1.0)


def test_head_weights_follow_class_differences():
    rng = np.random.default_rng(0)
    G = rng.normal(size=(40, 2, 2, 5)) * 0.01
    y = np.array([0, 1] * 20)
    G[y == 1, 1, 0, 2] += 1.0                                  # only layer 2, head 1 separates the classes
    w = head_weights(G, y)
    assert np.isclose(w.sum(), 1) and w[1, 0] > 0.9
    Gt = reduce_heads(G, w)
    assert Gt.shape == (40, 5) and np.argmax(discriminative(Gt, y)) == 2


# --- decomposition ----------------------------------------------------------------------------

def _planted(seed=1, offset=0.0):
    rng = np.random.default_rng(seed)
    A, B = rng.random((16, 3)), rng.random((8, 3))
    C = np.array([[0.9, 0.1, 0.5], [0.1, 0.9, 0.5]])
    return np.einsum("ik,jk,mk->ijm", A, B, C) + offset, C


def test_cp_recovers_a_planted_tensor_with_modality_on_the_simplex():
    T, C = _planted()
    res = cp.nn_cp(T, 3, restarts=5)
    assert res.error < 1e-4
    assert np.allclose(res.factors[-1].sum(0), 1)
    found = sorted(map(tuple, np.round(res.factors[-1].T, 2)))
    assert found == sorted(map(tuple, np.round(C.T, 2)))


def test_cp_removes_a_flat_background_component():
    T, C = _planted(offset=0.3)
    res = cp.fit(T, 3, restarts=5)
    assert res.dropped_background and res.rank == 3
    assert sorted(np.round(res.factors[-1][0], 1)) == [0.1, 0.5, 0.9]


def test_modality_calls():
    assert cp.modality_call([0.8, 0.2]) == "unique:0" and cp.modality_call([0.6, 0.4]) == "shared"
    rng = np.random.default_rng(0)
    near = np.clip(rng.normal(0.66, 0.05, (200, 1)), 0, 1)
    assert cp.modality_call_interval(np.hstack([near, 1 - near])) in ("undetermined", "shared")
    sure = np.clip(rng.normal(0.9, 0.02, (200, 1)), 0, 1)
    assert cp.modality_call_interval(np.hstack([sure, 1 - sure])) == "unique:0"


def test_alignment_is_invariant_to_component_order():
    T, _ = _planted()
    a = cp.nn_cp(T, 3, restarts=3)
    perm = np.array([2, 0, 1])
    b = a.select(perm)
    idx, sim = align.match(a, b)
    assert (perm[idx] == np.arange(3)).all() and np.allclose(sim, 1)
    assert np.allclose(align.consensus(a, [b, b])["stability"], 1)


# --- statistics and metrics ---------------------------------------------------------------------

def test_region_permutation_is_calibrated_and_finds_the_planted_region():
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 40)
    null = rng.normal(size=(80, 6, 4, 2))
    p = stats.region_permutation(null, y, None, n_perm=200, seed=1)["p"]
    assert 0.3 < p.mean() < 0.7
    sig = null.copy()
    sig[y == 1, 2, :, 0] += 1.0
    res = stats.region_permutation(sig, y, None, n_perm=200, seed=1)
    assert res["p"][2, 0] < 0.01 and (res["q"] >= res["p"] - 1e-12).all()


def test_bootstrap_keeps_strata_and_classes():
    y = np.array([0, 0, 1, 1, 0, 1])
    strata = np.array([1, 1, 1, 2, 2, 2])
    for idx in stats.bootstrap_indices(y, strata, 20, seed=0):
        assert sorted(zip(strata[idx], y[idx])) == sorted(zip(strata, y))
    assert stats.paired_p([1, 2, 3], [0, 0, 0]) == pytest.approx(1 / 3)


def test_recovery_metrics_bounds():
    truth = np.zeros((2, 6, 4, 2))
    truth[0, :2, :2, 0] = 1
    truth[1, 4:, 2:, 1] = 1
    sup = truth > 0
    perfect = metrics.recovery(truth[::-1], truth, sup)
    assert perfect["motif_correlation"] == pytest.approx(1) and perfect["unit_auroc"] == pytest.approx(1)
    assert perfect["modality_error"] == pytest.approx(0)
    flat = metrics.flat_control(truth.shape[1:], truth, sup)
    assert flat["unit_auroc"] == pytest.approx(0.5, abs=0.05)
    assert flat["mass_enrichment"] == pytest.approx(1, abs=0.01)


# --- a whole run --------------------------------------------------------------------------------

@pytest.mark.slow
def test_interpret_a_cross_validated_run(fast_dataset, tmp_path):
    from prismt.config import load_config
    from prismt.interpret.pipeline import InterpretOptions, interpret
    from prismt.run import run

    cfg = load_config({"task": "classify", "dataset": {"path": str(fast_dataset)}, "labels": {"column": "stim"},
                       "train": {"epochs": 4, "min_epochs": 0, "device": "cpu"},
                       "model": {"d_model": 16, "n_layers": 2, "n_heads": 2}, "output": {"root": str(tmp_path)}})
    out = run(cfg)
    res = interpret(out, InterpretOptions(rank=2, bootstrap=5, permutations=20, device="cpu"))
    info = json.loads((res / "interpret.json").read_text())
    assert info["rank"] == 2 and len(info["motifs"]) == 2
    from scipy.io import loadmat
    m = loadmat(res / "interpret.mat")
    assert m["motifs"].shape[0] == 2 and np.allclose(m["motif_modality"].sum(0), 1)
    # every attributed trial was held out by the fold whose model attributed it
    test_fold = {}
    with open(out / "splits.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["split"] == "test":
                test_fold[int(row["trial_index"])] = int(row["fold"])
    for t, f in zip(m["trial_index"].ravel().astype(int), m["trial_fold"].ravel().astype(int)):
        assert test_fold[t] == f
    assert len(m["trial_index"].ravel()) == len(test_fold)
