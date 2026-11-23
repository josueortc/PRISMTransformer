"""Evaluation fixes: cross-channel ridge baseline, paired comparison, session scaling, repeats."""

from __future__ import annotations

import json

import numpy as np
import pytest

from prismt.data.preprocess import session_zscore
from prismt.eval.baselines import mae_predictions, pair_covariance
from prismt.run import paired_comparison


def test_ridge_baseline_predicts_a_hidden_channel_from_a_correlated_one():
    rng = np.random.default_rng(0)
    n, T, Q, P = 200, 4, 3, 1
    base = rng.normal(size=(n, T, 1, P))
    values = np.concatenate([base, base + 0.05 * rng.normal(size=base.shape), rng.normal(size=base.shape)], axis=2)
    valid = np.ones((n, T * Q), bool)
    flat = values.reshape(n, T * Q, P)
    psth = np.zeros((T, Q, P))
    cov = pair_covariance(flat, valid, np.arange(150), T, Q, psth)
    test = values[150:]
    vis = np.ones(test.shape[:3], bool)
    vis[:, :, 1] = False                                        # hide channel 2 entirely
    pred = mae_predictions(test, vis, psth, cov)
    err = {k: float(((v[:, :, 1] - test[:, :, 1]) ** 2).mean()) for k, v in pred.items()}
    assert err["ridge"] < 0.2 * err["psth"]
    assert err["visible_mean"] == pytest.approx(err["psth"])  # nothing visible in that channel: falls back
    assert np.allclose(mae_predictions(test, vis, psth)["ridge"], psth[None])  # without a covariance


def test_paired_comparison_per_subject():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.array([f"m{i}" for i in range(8)], dtype=object), 40)
    y = np.tile([0, 1], 160)
    model = np.where(rng.random(320) < 0.9, y, 1 - y)
    ref = np.where(rng.random(320) < 0.6, y, 1 - y)
    out = paired_comparison(y, model, ref, groups, 2, seed=0)
    assert out["unit"] == "subject" and out["n"] == 8 and out["beats_reference"]
    assert out["better"] == 8 and out["sign_test_p"] < 0.01
    same = paired_comparison(y, model, model, groups, 2, seed=0)
    assert not same["beats_reference"] and same["mean_difference"] == 0
    trials = paired_comparison(y, model, ref, None, 2, seed=0)
    assert trials["unit"] == "trial" and trials["beats_reference"]


def test_session_zscore():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 3, 5, 2)).astype(np.float32)
    session = np.repeat(np.array(["a", "b"], dtype=object), 10)
    X[session == "b"] = X[session == "b"] * 4 + 10
    Z = session_zscore(X, session)
    for s in ("a", "b"):
        assert np.allclose(Z[session == s].mean(axis=(0, 2)), 0, atol=1e-5)
        assert np.allclose(Z[session == s].std(axis=(0, 2)), 1, atol=1e-4)


@pytest.mark.slow
def test_run_reports_session_baseline_paired_comparison_and_repeats(fast_dataset, tmp_path):
    from prismt.config import load_config
    from prismt.run import run

    cfg = load_config({"task": "classify", "dataset": {"path": str(fast_dataset)}, "labels": {"column": "stim"},
                       "preprocess": {"normalize": "zscore_session"},
                       "train": {"epochs": 2, "min_epochs": 0, "device": "cpu", "seeds": 2},
                       "model": {"d_model": 16, "n_layers": 1}, "output": {"root": str(tmp_path)}})
    out = run(cfg)
    m = json.loads((out / "metrics.json").read_text())
    assert "session_only" in m["baselines"] and "paired_vs_logistic" in m["baselines"]
    assert m["baselines"]["paired_vs_logistic"]["unit"] == "subject"
    assert len(m["repeats"]) == m["n_folds"] and len(m["repeats"][0]["test"]) == 2
    assert any("repeats ranged" in line for line in m["summary_lines"])
    assert (out / "fold_01" / "repeat_02" / "model.pt").exists() and (out / "fold_01" / "model.pt").exists()


@pytest.mark.slow
def test_autoencoder_run_includes_the_ridge_baseline(fast_dataset, tmp_path):
    from prismt.config import load_config
    from prismt.run import run

    cfg = load_config({"task": "mae", "dataset": {"path": str(fast_dataset)},
                       "mae": {"mask": {"strategy": "channel", "ratio": 0.3}},
                       "train": {"epochs": 2, "min_epochs": 0, "device": "cpu"},
                       "model": {"d_model": 16, "n_layers": 1}, "output": {"root": str(tmp_path)}})
    out = run(cfg)
    m = json.loads((out / "metrics.json").read_text())
    first = next(iter(m["test"].values()))
    assert "ridge" in first["baselines"] and first["baselines"]["ridge"]["r2"] is not None
