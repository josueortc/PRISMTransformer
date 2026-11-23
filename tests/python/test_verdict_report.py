"""The verdict, the trained-enough check, run comparison, motif tests and the PDF report."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from prismt.eval import verdict as vd


def _outcome(fold, hist, stopped, best, run):
    return SimpleNamespace(fold=SimpleNamespace(index=fold),
                           fit=SimpleNamespace(history=hist, stopped=stopped, best_epoch=best, epochs_run=run))


def _hist(train, val):
    return [{"train_loss": t, "val_loss": v} for t, v in zip(train, val)]


def test_trained_enough_flags_models_that_barely_learned_or_were_cut_short():
    barely = [_outcome(0, _hist([0.70, 0.68, 0.67], [0.69, 0.68, 0.68]), "max_epochs", 3, 3)]
    item = vd.trained_enough(vd.training_stats(barely, "classify", 2), reference=0.95, chance=0.5)
    assert item["status"] == "bad" and "information is there" in item["detail"]
    cut = [_outcome(0, _hist([0.6, 0.4, 0.2], [0.6, 0.45, 0.3]), "max_epochs", 3, 3)]
    assert vd.trained_enough(vd.training_stats(cut, "classify", 2))["status"] == "caution"
    done = [_outcome(0, _hist([0.6, 0.3, 0.1, 0.1] * 5, [0.6, 0.35, 0.3, 0.31] * 5), "patience", 3, 20)]
    assert vd.trained_enough(vd.training_stats(done, "classify", 2))["status"] == "good"
    none = [_outcome(0, [], "pbt", None, 10)]
    assert vd.trained_enough(vd.training_stats(none, "classify", 2))["status"] == "info"


def test_classification_verdict_is_conservative():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.array([f"m{i}" for i in range(8)], dtype=object), 40)
    y = np.tile([0, 1], 160)
    good = np.where(rng.random(320) < 0.9, y, 1 - y)
    by_group = {"subject": [{"group": g, "balanced_accuracy": 0.9, "accuracy": 0.9, "classes_present": 2}
                            for g in np.unique(groups)]}
    base = {"logistic": {"balanced_accuracy": 0.6},
            "paired_vs_logistic": {"mean_difference": 0.3, "ci": [0.2, 0.4], "unit": "subject"},
            "session_only": {"balanced_accuracy": 0.5}}
    items, ci = vd.classification(y=y, pred=good, groups=groups, group_name="subject", n_classes=2, ba=0.9,
                                  baselines=base, by_group=by_group, stats=[])
    status = {i["key"]: i["status"] for i in items}
    assert status["above_chance"] == "good" and status["vs_linear"] == "good" and status["subjects"] == "good"
    assert status["sessions"] == "good" and 0.8 < ci["lo"] < ci["hi"] <= 1
    chance = rng.integers(0, 2, 320)
    base2 = {"logistic": {"balanced_accuracy": 0.9},
             "paired_vs_logistic": {"mean_difference": -0.4, "ci": [-0.5, -0.3], "unit": "subject"},
             "session_only": {"balanced_accuracy": 0.9}}
    items2, _ = vd.classification(y=y, pred=chance, groups=groups, group_name="subject", n_classes=2,
                                  ba=vd.balanced_accuracy(y, chance, 2), baselines=base2, by_group=by_group, stats=[])
    status2 = {i["key"]: i["status"] for i in items2}
    assert status2["above_chance"] in ("bad", "caution") and status2["vs_linear"] == "caution"
    assert status2["sessions"] == "caution"
    assert all(line.startswith("[") for line in vd.lines(items))


@pytest.mark.slow
def test_run_verdict_motif_tests_comparison_and_report(fast_dataset, tmp_path):
    pytest.importorskip("matplotlib")
    from prismt.compare import compare
    from prismt.config import load_config
    from prismt.interpret.pipeline import InterpretOptions, interpret
    from prismt.report import report
    from prismt.run import run

    def cfg(name, **over):
        base = {"task": "classify", "name": name, "dataset": {"path": str(fast_dataset)}, "labels": {"column": "stim"},
                "train": {"epochs": 3, "min_epochs": 0, "device": "cpu"}, "model": {"d_model": 16, "n_layers": 2, "n_heads": 2},
                "output": {"root": str(tmp_path)}}
        base.update(over)
        return load_config(base)

    a = run(cfg("both"))
    b = run(cfg("calcium", selection={"modalities": ["calcium"]}))
    m = json.loads((a / "metrics.json").read_text())
    keys = [i["key"] for i in m["verdict"]]
    assert keys == ["above_chance", "vs_linear", "subjects", "sessions", "trained"]
    assert len(m["headline"]["ci"]) == 2 and len(m["training_check"]) == m["n_folds"]

    out = interpret(a, InterpretOptions(rank=2, bootstrap=3, permutations=10, device="cpu"))
    info = json.loads((out / "interpret.json").read_text())
    for motif in info["motifs"]:
        assert set(motif["test"]) >= {"cost", "control_mean", "z", "used", "answer", "n_elements"}
    from scipy.io import loadmat
    mat = loadmat(out / "interpret.mat")
    assert mat["motif_support"].shape == mat["motifs"].shape and mat["motif_test_cost"].size == 2

    res = compare([a, b])
    assert [r["name"] for r in res["runs"]] == ["both", "calcium"]
    assert len(res["pairs"]) == 1 and res["pairs"][0]["n_common"] == m["test"]["n"]

    pdf = report(a)
    assert pdf.exists() and pdf.stat().st_size > 20000
    text = (a / "report_methods.txt").read_text()
    assert "PRISMT (version" in text and "cross-validation" in text and "PRISMt" in text

    mae = run(load_config({"task": "mae", "dataset": {"path": str(fast_dataset)},
                           "train": {"epochs": 2, "min_epochs": 0, "device": "cpu"},
                           "model": {"d_model": 16, "n_layers": 1}, "output": {"root": str(tmp_path)}}))
    mm = json.loads((mae / "metrics.json").read_text())
    assert [i["key"] for i in mm["verdict"]][-1] == "trained"
    assert report(mae).exists()
