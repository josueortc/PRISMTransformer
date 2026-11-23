"""Finding the best model ("Find the best model" in the app): nested selection with Optuna.

Selecting settings is part of fitting the model, so it must not see the trials the model is
scored on. Every cross-validation fold therefore runs its own search:

* fold k's search trains each candidate on fold k's training trials and scores it on fold
  k's validation trials (never on its test trials, and never on another fold's data);
* the best candidate of fold k is retrained (``hpo.final_seeds`` times; the seed with the
  best validation score is kept) and tested once on fold k's test trials;
* the per-fold test predictions are pooled into an ordinary run folder (``final/``), so the
  score is an honest cross-validated estimate of "search, then train", and every held-out
  trial is explained (PRISMt) by the model of its own fold;
* the settings each fold chose are reported, with "consensus" settings: the majority value of
  each discrete setting and the median (in log scale where the search is) of each continuous
  one. They describe the best model; they were not used to compute the score.

``hpo.method = "pbt"`` uses population-based training instead (an evolutionary method, see
:mod:`prismt.pbt`), nested in the same way.

Candidates differ in learning rate, dropout, weight decay, width, depth and, for
classification, whether the encoder is first pretrained as a masked autoencoder on the
fold's training trials. Each fold's study lives in ``<run>/hpo/fold_<k>/study.journal``
(Optuna's journal storage, safe on network file systems), so cluster jobs can search one
fold in parallel and a re-submitted job continues where it stopped.
"""

from __future__ import annotations

import copy
import dataclasses
import gc
import logging
import math
from pathlib import Path

import numpy as np

from prismt.errors import ConfigError, EnvironmentProblem
from prismt.io.runfiles import HistoryWriter, StatusWriter, write_csv
from prismt.jsonutil import read_json, write_json

log = logging.getLogger("prismt")

LOG_SCALE = {"train.lr", "train.weight_decay"}
SPACES = {
    "small": {"train.lr": ("float", 1e-4, 3e-3, True), "model.dropout": ("cat", [0.05, 0.1, 0.2])},
    "default": {"train.lr": ("float", 1e-4, 3e-3, True), "model.dropout": ("cat", [0.05, 0.1, 0.2]),
                "train.weight_decay": ("float", 1e-4, 1e-1, True), "model.d_model": ("cat", [32, 64, 128]),
                "model.n_layers": ("int", 1, 4), "pretrain": ("cat", [False, True])},
    # the grid of the PRISMt paper (needs a GPU)
    "paper": {"train.lr": ("float", 1e-4, 1e-2, True), "model.dropout": ("cat", [0.05, 0.1, 0.2]),
              "model.d_model": ("cat", [256, 512, 768, 1024]), "model.n_layers": ("int", 2, 6),
              "pretrain": ("cat", [False, True])},
}


def _optuna():
    try:
        import optuna
    except ImportError as exc:
        raise EnvironmentProblem("E_ENV_NO_OPTUNA", "Finding the best model needs the optuna package.",
                                 hint="Install it with: pip install optuna (or recreate the PRISMT environment).") from exc
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna


def _storage(study_dir: Path):
    optuna = _optuna()
    from optuna.storages import JournalStorage

    study_dir.mkdir(parents=True, exist_ok=True)
    path = str(study_dir / "study.journal")
    try:
        from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

        backend = JournalFileBackend(path, lock_obj=JournalFileOpenLock(path))
    except ImportError:  # optuna 3.x
        from optuna.storages import JournalFileOpenLock, JournalFileStorage

        backend = JournalFileStorage(path, lock_obj=JournalFileOpenLock(path))
    return optuna, JournalStorage(backend)


def space(cfg: dict) -> dict:
    sp = dict(SPACES[cfg["hpo"]["space"]])
    if cfg["task"] != "classify":
        sp.pop("pretrain", None)
    return sp


def suggest(trial, cfg: dict) -> dict:
    """Sampled settings as {dotted path: value} ('pretrain' is not a config path)."""
    out = {}
    for path, spec in space(cfg).items():
        name = path.split(".")[-1]
        if spec[0] == "float":
            out[path] = trial.suggest_float(name, spec[1], spec[2], log=spec[3])
        elif spec[0] == "int":
            out[path] = trial.suggest_int(name, spec[1], spec[2])
        else:
            out[path] = trial.suggest_categorical(name, spec[1])
    return out


def params_to_overrides(params: dict, cfg: dict) -> dict:
    return {path: params[path.split(".")[-1]] for path in space(cfg) if path.split(".")[-1] in params}


def apply(cfg: dict, overrides: dict) -> dict:
    from prismt.config import set_value

    new = copy.deepcopy(cfg)
    for path, value in overrides.items():
        if path != "pretrain":
            set_value(new, path, value)
    if new["model"]["d_model"] % new["model"]["n_heads"]:
        new["model"]["n_heads"] = next(h for h in (8, 4, 2, 1) if new["model"]["d_model"] % h == 0)
    return new


def fit_candidate(prep, fold, cfg: dict, device, *, pretrain: bool = False, epochs: int | None = None,
                  on_epoch_end=None, history: HistoryWriter | None = None, fold_label: int = 1, n_folds: int = 1,
                  should_stop=None):
    """Train one candidate on ``fold.train`` (validated on ``fold.val``); test trials are not touched.

    Returns (model, task, tensors, (normalizer, values, valid), fit result)."""
    from prismt.model.checkpoint import load_pretrained_encoder
    from prismt.run import _build
    from prismt.train.loop import fit

    tcfg = copy.deepcopy(cfg)
    if epochs:
        tcfg["train"]["epochs"] = epochs
        tcfg["train"]["min_epochs"] = min(tcfg["train"].get("min_epochs") or 0, epochs)
    pre = None
    if pretrain and tcfg["task"] == "classify":
        mcfg = copy.deepcopy(tcfg)
        mcfg["task"] = "mae"
        mprep = dataclasses.replace(prep, cfg=mcfg)
        mmodel, mtask, mtensors, _ = _build(mprep, fold, device)
        fit(mmodel, mtask, mtensors, fold, mcfg["train"], monitor="val_loss")
        pre = {"model_spec": mmodel.spec.to_json(), "state_dict": mmodel.state_dict()}
        del mmodel
    cprep = dataclasses.replace(prep, cfg=tcfg)
    model, task, tensors, extras = _build(cprep, fold, device)
    if pre is not None:
        load_pretrained_encoder(model, pre, source="pretraining within the fold")
    result = fit(model, task, tensors, fold, tcfg["train"], monitor="val_loss", on_epoch_end=on_epoch_end,
                 history=history, fold_label=fold_label, n_folds=n_folds, should_stop=should_stop)
    return model, task, tensors, extras, result


def search_fold(cfg: dict, run_dir: Path, fold_index: int, *, worker: int = 0, status: StatusWriter | None = None) -> dict:
    """Run (or continue) fold ``fold_index``'s search until its budget is used."""
    import torch

    from prismt.env import select_device
    from prismt.run import prepare

    optuna, storage = _storage(run_dir / "hpo" / f"fold_{fold_index + 1:02d}")
    from optuna.study import MaxTrialsCallback
    from optuna.trial import TrialState

    sampler = optuna.samplers.TPESampler(seed=cfg["train"]["seed"] + 1000 * worker + 17 * fold_index,
                                         constant_liar=True, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study = optuna.create_study(study_name="prismt", storage=storage, sampler=sampler, pruner=pruner,
                                direction="minimize", load_if_exists=True)
    prep = prepare(cfg)
    fold = prep.plan.folds[fold_index]
    n_folds = len(prep.plan.folds)
    device = select_device(cfg["train"]["device"])
    n_trials = cfg["hpo"]["n_trials"]
    epochs = cfg["hpo"]["epochs_per_trial"] or cfg["train"]["epochs"]

    def objective(trial):
        overrides = suggest(trial, cfg)
        tcfg = apply(cfg, overrides)

        def report(epoch, value):
            trial.report(value, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        model = None
        try:
            model, *_rest, result = fit_candidate(prep, fold, tcfg, device, pretrain=bool(overrides.get("pretrain")),
                                                  epochs=epochs, on_epoch_end=report)
            done = len([t for t in study.get_trials(deepcopy=False)
                        if t.state in (TrialState.COMPLETE, TrialState.PRUNED)])
            if status:
                status.update("training", force=True, message=f"Fold {fold_index + 1}/{n_folds}: tried {done + 1} "
                              f"of {n_trials} settings", fold=fold_index + 1, n_folds=n_folds,
                              trials_done=done + 1, n_trials=n_trials)
            if not np.isfinite(result.best_value):
                raise optuna.TrialPruned()
            return result.best_value
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                log.warning("trial %d ran out of memory; pruned", trial.number)
                raise optuna.TrialPruned() from exc
            raise
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    timeout = 60 * cfg["hpo"]["timeout_minutes"] if cfg["hpo"]["timeout_minutes"] else None
    finished = [t for t in study.get_trials(deepcopy=False) if t.state in (TrialState.COMPLETE, TrialState.PRUNED)]
    if len(finished) < n_trials:
        study.optimize(objective, n_trials=n_trials, timeout=timeout, gc_after_trial=True,
                       callbacks=[MaxTrialsCallback(n_trials, states=(TrialState.COMPLETE, TrialState.PRUNED))],
                       catch=())
    return {"fold": fold_index + 1, "worker": worker, "trials": len(study.trials)}


def consensus(chosen: list[dict], cfg: dict) -> dict:
    """Majority value of each discrete setting, median (log-median for log-scale ones) of continuous ones."""
    out = {}
    for path, spec in space(cfg).items():
        name = path.split(".")[-1]
        vals = [c[name] for c in chosen if name in c]
        if not vals:
            continue
        if spec[0] == "float":
            arr = np.asarray(vals, float)
            out[path] = float(math.exp(np.median(np.log(arr)))) if spec[3] else float(np.median(arr))
        else:
            counts = {}
            for v in vals:
                counts[v] = counts.get(v, 0) + 1
            top = max(counts.values())
            out[path] = next(v for v in vals if counts[v] == top)  # ties: the first fold's choice
    return out


def finalize(cfg: dict, run_dir: Path, status: StatusWriter | None = None) -> dict:
    """Retrain each fold's best settings, test each once, and pool them into ``final/``."""
    from optuna.trial import TrialState

    from prismt.env import select_device
    from prismt.model.checkpoint import save_checkpoint
    from prismt.run import _evaluate_fold, _write_splits, _write_summary, data_signature, prepare

    prep = prepare(cfg)
    device = select_device(cfg["train"]["device"])
    n_folds = len(prep.plan.folds)
    final = run_dir / "final"
    final.mkdir(exist_ok=True)
    history = HistoryWriter(final / "history.csv")
    outcomes, per_fold, rows = [], [], []
    for fold in prep.plan.folds:
        optuna, storage = _storage(run_dir / "hpo" / f"fold_{fold.index + 1:02d}")
        study = optuna.load_study(study_name="prismt", storage=storage)
        complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if not complete:
            raise ConfigError("E_HPO_NO_TRIALS", f"No tried setting finished in fold {fold.index + 1}, so there is "
                              "nothing to retrain.", hint="Check log.txt; give the search more time or use smaller models.")
        rows += [{"fold": fold.index + 1, "number": t.number, "state": t.state.name, "value": t.value, **t.params}
                 for t in study.trials]
        best = min(complete, key=lambda t: t.value)
        overrides = params_to_overrides(best.params, cfg)
        fcfg = apply(cfg, overrides)
        fcfg["preset"] = "custom"
        tries = []
        for k in range(cfg["hpo"]["final_seeds"]):
            if status:
                status.update("training", force=True, message=f"Fold {fold.index + 1}/{n_folds}: retraining the best "
                              f"settings (repeat {k + 1})", fold=fold.index + 1, n_folds=n_folds)
            scfg = copy.deepcopy(fcfg)
            scfg["train"]["seed"] = cfg["train"]["seed"] + k
            fitted = fit_candidate(prep, fold, scfg, device, pretrain=bool(overrides.get("pretrain")),
                                   history=history if k == 0 else None, fold_label=fold.index + 1, n_folds=n_folds)
            tries.append((fitted[-1].best_value, k, scfg, fitted))
        best_val, k_best, scfg, (model, task, tensors, (norm, values, valid), result) = min(tries, key=lambda t: t[0])
        sprep = dataclasses.replace(prep, cfg=scfg)
        outcome = _evaluate_fold(sprep, fold, model, task, tensors, values, valid, norm, result)
        outcomes.append(outcome)
        fold_dir = final if n_folds == 1 else final / f"fold_{fold.index + 1:02d}"
        fold_dir.mkdir(exist_ok=True)
        save_checkpoint(fold_dir / "model.pt", model, extra={
            "normalizer": norm.to_state(), "data_signature": data_signature(prep),
            "class_names": prep.sel.class_names, "best_epoch": result.best_epoch,
            "best_value": result.best_value if np.isfinite(result.best_value) else None,
            "epochs_run": result.epochs_run, "stopped": result.stopped, "seconds": result.seconds,
            "dataset_fingerprint": prep.ds.fingerprint, "split_fingerprint": prep.plan.fingerprint(),
            "run_config": scfg, "fold": fold.index + 1, "selected": overrides})
        headline = outcome.test["metrics"].get("balanced_accuracy" if cfg["task"] == "classify" else "r2")
        per_fold.append({"fold": fold.index + 1, "best_trial": best.number, "validation": best.value,
                         "settings": {p: overrides[p] for p in overrides}, "kept_repeat": k_best + 1,
                         "repeat_validation": [float(t[0]) for t in tries], "test": headline,
                         "n_trials": len(study.trials)})
        del tries
        gc.collect()
    write_csv(run_dir / "trials.csv", rows)
    agreed = consensus([{p.split(".")[-1]: v for p, v in f["settings"].items()} for f in per_fold], cfg)
    resolved = dict(cfg)
    resolved.update({"dataset_fingerprint": prep.ds.fingerprint,
                     "model_selection": {"per_fold": per_fold, "consensus": agreed}})
    write_json(final / "config.json", resolved)
    _write_splits(final, prep)
    _write_summary(final, prep, outcomes, device)
    metrics = read_json(final / "metrics.json")
    lines = [f"Settings were chosen within each of the {n_folds} folds, using only its training and validation trials."]
    for f in per_fold:
        lines.append(f"Fold {f['fold']}: " + ", ".join(_fmt(p, v) for p, v in f["settings"].items()))
    lines.append("Consensus settings: " + ", ".join(_fmt(p, v) for p, v in agreed.items()))
    metrics["summary_lines"] = metrics["summary_lines"] + lines
    metrics["model_selection"] = {"per_fold": per_fold, "consensus": agreed, "nested": True}
    write_json(final / "metrics.json", metrics)
    StatusWriter(final / "status.json", task=cfg["task"], run_dir=str(final)).update(
        "finished", force=True, message=metrics["summary_lines"][0])
    summary = {"nested": True, "per_fold": per_fold, "consensus": agreed, "final_run": str(final), "kept": str(final),
               "n_trials": int(sum(f["n_trials"] for f in per_fold)),
               "summary_lines": [metrics["summary_lines"][0], *lines]}
    write_json(run_dir / "hpo_summary.json", summary)
    write_json(run_dir / "best_config.json", {k: v for k, v in apply(cfg, agreed).items() if k != "config_version"})
    return summary


def _fmt(path: str, v) -> str:
    name = path.split(".")[-1]
    if isinstance(v, bool):
        return f"{name} = {'yes' if v else 'no'}"
    return f"{name} = {v:.3g}" if isinstance(v, float) else f"{name} = {v}"


def run_hpo(cfg: dict, run_dir: str | Path | None, *, worker: int | None = None, fold: int | None = None,
            task_index: int | None = None, do_finalize: bool | None = None, source: dict | None = None) -> Path:
    """Locally: search every fold, then finalize. On a cluster: ``task_index`` (the array task)
    picks the fold and worker; a separate job finalizes."""
    from prismt.io.runfiles import create_run_dir
    from prismt.run import prepare

    out = Path(run_dir) if run_dir else create_run_dir(cfg["output"]["root"], "best-" + cfg["task"], cfg["name"])
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", cfg)
    if source is not None and not (out / "run.json").exists():
        write_json(out / "run.json", source)
    status = StatusWriter(out / "status.json", task=cfg["task"], mode="hpo", run_dir=str(out))
    try:
        n_folds = len(prepare(cfg).plan.folds)
        if task_index is not None:
            workers = max(1, cfg["hpo"]["workers"])
            fold, worker = task_index // workers, task_index % workers
            if fold >= n_folds:
                status.update("finished", force=True, message="Nothing to search for this array task")
                return out
        pbt = cfg["hpo"].get("method", "tpe") == "pbt"
        if not do_finalize:
            folds = [fold] if fold is not None else range(n_folds)
            for f in folds:
                if pbt:
                    from prismt.pbt import evolve_fold

                    if worker:   # a population evolves within one job; extra workers have nothing to do
                        continue
                    evolve_fold(cfg, out, f, status=status)
                else:
                    search_fold(cfg, out, f, worker=worker or 0, status=status)
        if do_finalize or (task_index is None and worker is None and do_finalize is None):
            status.update("evaluating", force=True, message="Testing each fold's best model")
            if pbt:
                from prismt.pbt import finalize as pbt_finalize

                summary = pbt_finalize(cfg, out, status)
            else:
                summary = finalize(cfg, out, status)
            status.update("finished", force=True, message=summary["summary_lines"][0], final_run=summary["final_run"])
        else:
            status.update("finished", force=True, message=f"Search worker finished (fold {(fold or 0) + 1})")
    except Exception as exc:  # noqa: BLE001
        err = exc.to_dict() if hasattr(exc, "to_dict") else {"code": "E_INTERNAL", "title": "Finding the best model failed",
                                                              "message": str(exc), "hint": "", "field": ""}
        status.fail(err)
        raise
    return out
