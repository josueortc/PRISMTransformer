"""Population-based training (PBT; Jaderberg et al., 2017), nested within each fold.

An evolutionary alternative to the TPE search of :mod:`prismt.hpo`. Within fold k:

1. A population of models starts from settings sampled from the search space (width,
   depth, learning rate, weight decay, dropout, and for classification whether the encoder
   is first pretrained as a masked autoencoder on the fold's training trials).
2. Every member trains for ``hpo.pbt_interval`` epochs and is scored on fold k's
   validation trials.
3. **Exploit**: each member in the bottom quarter copies the weights and settings of a
   member of the top quarter. **Explore**: the copy then perturbs its learning rate and
   weight decay (x0.8 or x1.25) and its dropout (+-0.05).
4. Steps 2-3 repeat until every member has trained ``train.epochs`` epochs.

The member with the best validation score is fold k's model: it is tested once on fold k's
test trials, which no step above has seen. The folds are then pooled into an ordinary run
folder (``final/``), as with TPE, so the score is a cross-validated estimate of "evolve,
then test". Each fold's history of exploits and settings is written to
``hpo/fold_<k>/pbt_history.csv``.

The learning rate is held constant within each interval (PBT itself schedules it).
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import math
from pathlib import Path

import numpy as np

from prismt.io.runfiles import StatusWriter, write_csv
from prismt.jsonutil import write_json

log = logging.getLogger("prismt")

CONTINUOUS = {"train.lr": (1e-5, 1e-1), "train.weight_decay": (1e-6, 1.0)}


def _sample(space: dict, rng: np.random.Generator) -> dict:
    out = {}
    for path, spec in space.items():
        if spec[0] == "float":
            lo, hi, log_scale = spec[1], spec[2], spec[3]
            out[path] = float(math.exp(rng.uniform(math.log(lo), math.log(hi)))) if log_scale else float(rng.uniform(lo, hi))
        elif spec[0] == "int":
            out[path] = int(rng.integers(spec[1], spec[2] + 1))
        else:
            out[path] = spec[1][int(rng.integers(len(spec[1])))]
    return out


def _perturb(settings: dict, rng: np.random.Generator) -> dict:
    new = dict(settings)
    for path, (lo, hi) in CONTINUOUS.items():
        if path in new:
            new[path] = float(np.clip(new[path] * (0.8 if rng.random() < 0.5 else 1.25), lo, hi))
    if "model.dropout" in new:
        new["model.dropout"] = float(np.clip(new["model.dropout"] + rng.choice([-0.05, 0.05]), 0.0, 0.5))
    return new


def _set_dropout(model, p: float) -> None:
    from torch import nn

    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = p
    for blk in getattr(model.encoder, "blocks", []):
        blk.attn.dropout = p


def evolve_fold(cfg: dict, run_dir: Path, fold_index: int, *, status: StatusWriter | None = None) -> dict:
    """Run PBT for one fold; saves the best member to ``hpo/fold_<k>/pbt_best.pt``."""
    import torch

    from prismt.env import select_device
    from prismt.hpo import apply, space
    from prismt.model.checkpoint import load_pretrained_encoder, save_checkpoint
    from prismt.run import _build, prepare
    from prismt.train.loop import fit

    prep = prepare(cfg)
    fold = prep.plan.folds[fold_index]
    n_folds = len(prep.plan.folds)
    device = select_device(cfg["train"]["device"])
    hp = cfg["hpo"]
    P = int(hp["population"])
    interval = int(hp["pbt_interval"])
    total = int(cfg["train"]["epochs"])
    rounds = max(1, math.ceil(total / interval))
    rng = np.random.default_rng(cfg["train"]["seed"] + 7907 * (fold_index + 1))
    sp = space(cfg)
    out_dir = run_dir / "hpo" / f"fold_{fold_index + 1:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors = extras = None
    members = []
    for i in range(P):
        settings = _sample(sp, rng)
        mcfg = apply(cfg, settings)
        mcfg["train"]["seed"] = cfg["train"]["seed"] + 101 * i
        mprep = dataclasses.replace(prep, cfg=mcfg)
        model, task, t, ex = _build(mprep, fold, device)
        if tensors is None:
            tensors, extras = t, ex
        if settings.get("pretrain") and mcfg["task"] == "classify":
            if status:
                status.update("training", force=True, message=f"Fold {fold_index + 1}/{n_folds}: pretraining member "
                              f"{i + 1}/{P}", fold=fold_index + 1, n_folds=n_folds)
            acfg = copy.deepcopy(mcfg)
            acfg["task"] = "mae"
            amodel, atask, at, _ = _build(dataclasses.replace(prep, cfg=acfg), fold, device)
            fit(amodel, atask, at, fold, {**acfg["train"], "epochs": interval}, monitor="val_loss")
            load_pretrained_encoder(model, {"model_spec": amodel.spec.to_json(), "state_dict": amodel.state_dict()},
                                    source="pretraining within the fold")
            del amodel
        members.append({"id": i, "settings": settings, "cfg": mcfg, "model": model, "task": task,
                        "value": math.inf, "epochs": 0, "lineage": [i]})

    history = []
    for r in range(rounds):
        epochs = min(interval, total - r * interval)
        for m in members:
            if status:
                status.update("training", force=True, message=f"Fold {fold_index + 1}/{n_folds}: generation "
                              f"{r + 1}/{rounds}, member {m['id'] + 1}/{P}", fold=fold_index + 1, n_folds=n_folds,
                              trials_done=r * P + m["id"] + 1, n_trials=rounds * P)
            tcfg = {**m["cfg"]["train"], "epochs": epochs, "min_epochs": epochs, "patience": epochs,
                    "warmup_fraction": 0.0, "min_lr_ratio": 1.0}
            _set_dropout(m["model"], m["cfg"]["model"]["dropout"])
            res = fit(m["model"], m["task"], tensors, fold, tcfg, monitor="val_loss")
            m["value"] = res.best_value if np.isfinite(res.best_value) else math.inf
            m["epochs"] += epochs
            history.append({"generation": r + 1, "member": m["id"] + 1, "validation": m["value"],
                            "lineage": "-".join(str(x + 1) for x in m["lineage"]), "copied_from": "",
                            **{k.split(".")[-1]: v for k, v in m["settings"].items()}})
        if r == rounds - 1:
            break
        order = sorted(members, key=lambda m: m["value"])
        q = max(1, P // 4)
        top, bottom = order[:q], order[-q:]
        for loser in bottom:
            if loser in top:
                continue
            winner = top[int(rng.integers(len(top)))]
            loser["model"] = copy.deepcopy(winner["model"])
            loser["task"] = winner["task"]
            loser["settings"] = _perturb(winner["settings"], rng)
            loser["cfg"] = apply(cfg, loser["settings"])
            loser["cfg"]["train"]["seed"] = winner["cfg"]["train"]["seed"] + 1
            loser["lineage"] = winner["lineage"] + [loser["id"]]
            history.append({"generation": r + 1, "member": loser["id"] + 1, "validation": winner["value"],
                            "lineage": "-".join(str(x + 1) for x in loser["lineage"]),
                            "copied_from": winner["id"] + 1,
                            **{k.split(".")[-1]: v for k, v in loser["settings"].items()}})
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    best = min(members, key=lambda m: m["value"])
    write_csv(out_dir / "pbt_history.csv", history)
    norm = extras[0]
    save_checkpoint(out_dir / "pbt_best.pt", best["model"], extra={"normalizer": norm.to_state(),
                                                                   "run_config": best["cfg"],
                                                                   "selected": best["settings"],
                                                                   "best_value": float(best["value"])})
    info = {"fold": fold_index + 1, "settings": best["settings"], "validation": float(best["value"]),
            "lineage": [x + 1 for x in best["lineage"]], "population": P, "generations": rounds,
            "interval": interval, "exploits": sum(1 for h in history if h["copied_from"] != "")}
    write_json(out_dir / "pbt_best.json", info)
    return info


def finalize(cfg: dict, run_dir: Path, status: StatusWriter | None = None) -> dict:
    """Test each fold's best member once and pool the folds into ``final/``."""
    from types import SimpleNamespace

    from prismt.env import select_device
    from prismt.hpo import _fmt, apply, consensus
    from prismt.jsonutil import read_json
    from prismt.model.checkpoint import extra, load_checkpoint, save_checkpoint
    from prismt.run import _evaluate_fold, _write_splits, _write_summary, data_signature, model_from_checkpoint, prepare
    from prismt.errors import ConfigError

    prep = prepare(cfg)
    device = select_device(cfg["train"]["device"])
    n_folds = len(prep.plan.folds)
    final = run_dir / "final"
    final.mkdir(exist_ok=True)
    outcomes, per_fold = [], []
    for fold in prep.plan.folds:
        fdir = run_dir / "hpo" / f"fold_{fold.index + 1:02d}"
        if not (fdir / "pbt_best.pt").exists():
            raise ConfigError("E_HPO_NO_TRIALS", f"Fold {fold.index + 1} has not finished evolving.",
                              hint="Wait for every fold's job to finish, then finalize again.")
        info = read_json(fdir / "pbt_best.json")
        ckpt = load_checkpoint(fdir / "pbt_best.pt")
        scfg = extra(ckpt, "run_config")
        sprep = dataclasses.replace(prep, cfg=scfg)
        model, task, tensors, (norm, values, valid) = model_from_checkpoint(sprep, fold, device, ckpt)
        result = SimpleNamespace(best_epoch=None, epochs_run=scfg["train"]["epochs"], stopped="pbt", seconds=0.0,
                                 best_value=info["validation"])
        outcome = _evaluate_fold(sprep, fold, model, task, tensors, values, valid, norm, result)
        outcomes.append(outcome)
        fold_dir = final if n_folds == 1 else final / f"fold_{fold.index + 1:02d}"
        fold_dir.mkdir(exist_ok=True)
        save_checkpoint(fold_dir / "model.pt", model, extra={
            "normalizer": norm.to_state(), "data_signature": data_signature(prep), "class_names": prep.sel.class_names,
            "dataset_fingerprint": prep.ds.fingerprint, "split_fingerprint": prep.plan.fingerprint(),
            "run_config": scfg, "fold": fold.index + 1, "selected": info["settings"]})
        headline = outcome.test["metrics"].get("balanced_accuracy" if cfg["task"] == "classify" else "r2")
        per_fold.append({**info, "test": headline})
    agreed = consensus([{p.split(".")[-1]: v for p, v in f["settings"].items()} for f in per_fold], cfg)
    resolved = dict(cfg)
    resolved.update({"dataset_fingerprint": prep.ds.fingerprint,
                     "model_selection": {"method": "pbt", "per_fold": per_fold, "consensus": agreed}})
    write_json(final / "config.json", resolved)
    _write_splits(final, prep)
    _write_summary(final, prep, outcomes, device)
    metrics = read_json(final / "metrics.json")
    lines = [f"Each of the {n_folds} folds evolved a population of {cfg['hpo']['population']} models "
             "(population-based training) on its own training and validation trials; its best model was tested once."]
    for f in per_fold:
        lines.append(f"Fold {f['fold']}: " + ", ".join(_fmt(p, v) for p, v in f["settings"].items()) +
                     f" ({f['exploits']} exploits)")
    metrics["summary_lines"] = metrics["summary_lines"] + lines
    metrics["model_selection"] = {"method": "pbt", "nested": True, "per_fold": per_fold, "consensus": agreed}
    write_json(final / "metrics.json", metrics)
    StatusWriter(final / "status.json", task=cfg["task"], run_dir=str(final)).update(
        "finished", force=True, message=metrics["summary_lines"][0])
    summary = {"nested": True, "method": "pbt", "per_fold": per_fold, "consensus": agreed, "final_run": str(final),
               "kept": str(final), "summary_lines": [metrics["summary_lines"][0], *lines]}
    write_json(run_dir / "hpo_summary.json", summary)
    write_json(run_dir / "best_config.json", {k: v for k, v in apply(cfg, agreed).items() if k != "config_version"})
    return summary
