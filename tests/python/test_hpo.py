"""Finding the best model: nested selection keeps every fold's test trials out of its search."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("optuna")

from prismt import hpo  # noqa: E402
from prismt.config import load_config  # noqa: E402


def _cfg(path, tmp_path, **hp):
    return load_config({"task": "classify", "dataset": {"path": str(path)}, "labels": {"column": "stim"},
                        "train": {"epochs": 2, "min_epochs": 0, "device": "cpu"},
                        "model": {"d_model": 16, "n_layers": 1}, "output": {"root": str(tmp_path)},
                        "hpo": {"n_trials": 2, "space": "small", **hp}})


def test_consensus_takes_majorities_and_log_medians(tiny_dataset, tmp_path):
    cfg = _cfg(tiny_dataset, tmp_path, space="default")
    chosen = [{"lr": 1e-4, "dropout": 0.1, "d_model": 32, "n_layers": 2, "weight_decay": 1e-3, "pretrain": True},
              {"lr": 1e-2, "dropout": 0.1, "d_model": 64, "n_layers": 2, "weight_decay": 1e-3, "pretrain": False},
              {"lr": 1e-3, "dropout": 0.2, "d_model": 64, "n_layers": 1, "weight_decay": 1e-1, "pretrain": True}]
    c = hpo.consensus(chosen, cfg)
    assert c["train.lr"] == pytest.approx(1e-3)
    assert c["model.dropout"] == 0.1 and c["model.d_model"] == 64 and c["model.n_layers"] == 2
    assert c["pretrain"] is True


def test_spaces_drop_pretraining_for_autoencoders(tiny_dataset, tmp_path):
    cfg = _cfg(tiny_dataset, tmp_path, space="default")
    assert "pretrain" in hpo.space(cfg)
    cfg["task"] = "mae"
    assert "pretrain" not in hpo.space(cfg)


@pytest.mark.slow
def test_search_never_evaluates_its_folds_test_trials(fast_dataset, tmp_path, monkeypatch):
    from prismt.run import prepare
    from prismt.tasks.classify import ClassifyTask

    cfg = _cfg(fast_dataset, tmp_path, space="default")
    plan = prepare(cfg).plan
    seen: list[np.ndarray] = []
    original = ClassifyTask.evaluate

    def spy(self, model, tensors, rows, batch_size):
        seen.append(np.asarray(rows))
        return original(self, model, tensors, rows, batch_size)

    monkeypatch.setattr(ClassifyTask, "evaluate", spy)
    run_dir = tmp_path / "search"
    for k, fold in enumerate(plan.folds):
        seen.clear()
        hpo.search_fold(cfg, run_dir, k)
        assert seen, "the search evaluated nothing"
        for rows in seen:
            assert not set(rows.tolist()) & set(fold.test.tolist()), f"fold {k + 1}'s search saw its test trials"
    monkeypatch.setattr(ClassifyTask, "evaluate", original)
    summary = hpo.finalize(cfg, run_dir)
    assert summary["nested"] and len(summary["per_fold"]) == len(plan.folds)
    metrics = json.loads((run_dir / "final" / "metrics.json").read_text())
    assert metrics["model_selection"]["nested"] and metrics["test"]["n"] == sum(len(f.test) for f in plan.folds)
    final_cfg = json.loads((run_dir / "final" / "config.json").read_text())
    assert "filters" in final_cfg["selection"] and "consensus" in final_cfg["model_selection"]


@pytest.mark.slow
def test_population_based_training_is_nested_and_exploits(fast_dataset, tmp_path, monkeypatch):
    import csv

    from prismt.pbt import evolve_fold, finalize
    from prismt.run import prepare
    from prismt.tasks.classify import ClassifyTask

    cfg = load_config({"task": "classify", "dataset": {"path": str(fast_dataset)}, "labels": {"column": "stim"},
                       "train": {"epochs": 4, "min_epochs": 0, "device": "cpu"},
                       "model": {"d_model": 16, "n_layers": 1}, "output": {"root": str(tmp_path)},
                       "hpo": {"method": "pbt", "population": 4, "pbt_interval": 2, "space": "small"}})
    plan = prepare(cfg).plan
    seen: list[np.ndarray] = []
    original = ClassifyTask.evaluate

    def spy(self, model, tensors, rows, batch_size):
        seen.append(np.asarray(rows))
        return original(self, model, tensors, rows, batch_size)

    monkeypatch.setattr(ClassifyTask, "evaluate", spy)
    run_dir = tmp_path / "pbt"
    for k, fold in enumerate(plan.folds):
        seen.clear()
        info = evolve_fold(cfg, run_dir, k)
        assert info["exploits"] >= 1 and info["generations"] == 2
        for rows in seen:
            assert not set(rows.tolist()) & set(fold.test.tolist())
        with open(run_dir / "hpo" / f"fold_{k + 1:02d}" / "pbt_history.csv", newline="") as fh:
            hist = list(csv.DictReader(fh))
        assert any(h["copied_from"] for h in hist)
    monkeypatch.setattr(ClassifyTask, "evaluate", original)
    summary = finalize(cfg, run_dir)
    assert summary["method"] == "pbt" and len(summary["per_fold"]) == len(plan.folds)
    metrics = json.loads((run_dir / "final" / "metrics.json").read_text())
    assert metrics["model_selection"]["method"] == "pbt"
    assert metrics["test"]["n"] == sum(len(f.test) for f in plan.folds)
