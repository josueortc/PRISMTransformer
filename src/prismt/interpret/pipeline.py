"""PRISMt for a finished classification run: attribute, reduce, decompose, test, write.

Every held-out trial is attributed by the model of the fold in which it was held out, so no
trial is explained by a model that was trained on it. Head weights are computed within each
fold (heads of different models are not the same heads); the reduced tensors, which live on
the recording's own axes, are then pooled over folds.

Two decompositions are available:

* ``trials`` (default for recordings): the magnitude of the per-trial tensor
  |G_tilde[n, c, t, m]| is factorized, so each motif has an expression per trial a_k(n)
  that can be compared between conditions, subjects or learning stages;
* ``discriminative`` (used for the synthetic benchmark, whose ground truth is defined on
  channel x time x modality): | mean(G_tilde | b) - mean(G_tilde | a) | is factorized.

Outputs go to ``<run>/interpret/``: ``interpret.json`` (summary), ``interpret.mat`` (arrays
for MATLAB) and ``attribution.npz`` (the pooled per-trial tensor).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from prismt.errors import ConfigError
from prismt.interpret import align, attribution, baselines, cp, faithfulness, metrics, stats
from prismt.interpret.reduce import discriminative, head_weights, reduce_heads
from prismt.io.runfiles import StatusWriter, utc_now, write_results_mat
from prismt.jsonutil import read_json, to_jsonable, write_json

log = logging.getLogger("prismt")
_TITLE = "PRISMt interpretation cannot run on this run"
G_MEMORY_LIMIT = 1.5 * 2**30  # keep per-head attribution for bootstrap refits below this size


@dataclass
class InterpretOptions:
    classes: tuple[str, str] | None = None   # class names (a, b); default: the two classes of the run
    rank: int | str = "auto"
    max_rank: int = 6
    mode: str = "auto"                       # trials | discriminative | auto
    restarts: int | None = None              # default 20 (discriminative) / 5 (trials)
    bootstrap: int = 200
    permutations: int = 1000
    compare: bool = True                     # baselines (rollout, observation-level methods)
    faithfulness: bool = True                # occlusion of ground-truth pathways (benchmark only)
    test_motifs: bool = True                 # occlude each motif's footprint: does the model use it?
    per_trial_u: bool = False
    max_trials: int | None = None            # attribute at most this many held-out trials (evenly spaced)
    batch_size: int = 32
    seed: int = 0
    device: str = "auto"
    extra: dict = field(default_factory=dict)


@dataclass
class FoldAttribution:
    fold: int
    rows: np.ndarray        # positions in the selection
    y01: np.ndarray
    G: np.ndarray | None    # [n, L, H, tokens] (kept when small enough)
    w: np.ndarray           # [L, H]
    Gt: np.ndarray          # [n, C, T, M] head-reduced
    rollout: np.ndarray     # [n, C, T, M]
    grad: np.ndarray        # [n, C, T, M]
    X: np.ndarray           # [n, C, T, M] normalized inputs (missing = 0)
    zeta: np.ndarray


def _core(res: cp.CPResult) -> cp.CPResult:
    """The channel, time and modality factors of a fit (dropping a trial mode)."""
    return cp.CPResult(res.weights, res.factors[-3:], res.error, res.n_iter, res.dropped_background)


def _class_pair(sel, opts: InterpretOptions) -> tuple[int, int]:
    names = list(sel.class_names)
    if opts.classes:
        missing = [c for c in opts.classes if c not in names]
        if missing:
            raise ConfigError("E_INTERP_CLASSES", f"There is no class {missing[0]!r} in this run.",
                              hint=f"Classes: {', '.join(names)}.", title=_TITLE)
        return names.index(opts.classes[0]), names.index(opts.classes[1])
    if len(names) != 2:
        raise ConfigError("E_INTERP_CLASSES", f"This run has {len(names)} classes; PRISMt contrasts two.",
                          hint="Choose the two classes to compare.", title=_TITLE)
    return 0, 1


def attribute_run(run_dir: Path, opts: InterpretOptions, status: StatusWriter | None = None):
    """Attribution for every fold's test trials. Returns (prep, class pair, folds, device)."""
    from prismt.env import select_device
    from prismt.model.checkpoint import extra, load_checkpoint
    from prismt.run import model_from_checkpoint, prepare

    cfg = read_json(run_dir / "config.json")
    if cfg.get("task") != "classify":
        raise ConfigError("E_INTERP_TASK", "PRISMt explains classifications; this run is not one.",
                          hint="Interpret a classification run (it may start from an autoencoder).", title=_TITLE)
    prep = prepare(cfg)
    pair = _class_pair(prep.sel, opts)
    device = select_device(opts.device if opts.device != "auto" else cfg["train"]["device"])
    n_folds = len(prep.plan.folds)
    C, M = len(prep.sel.channels), len(prep.sel.modalities)
    out = []
    for fold in prep.plan.folds:
        fold_dir = run_dir if n_folds == 1 else run_dir / f"fold_{fold.index + 1:02d}"
        if not (fold_dir / "model.pt").exists():
            raise ConfigError("E_INTERP_MISSING", f"Fold {fold.index + 1} has no trained model.",
                              hint="Wait for the run to finish (or combine its cluster folds).", title=_TITLE)
        if status:
            status.update("evaluating", force=True, message=f"Attributing fold {fold.index + 1}/{n_folds}",
                          fold=fold.index + 1, n_folds=n_folds)
        ckpt = load_checkpoint(fold_dir / "model.pt")
        saved = extra(ckpt, "split_fingerprint")
        if saved and saved != prep.plan.fingerprint():
            raise ConfigError("E_INTERP_SPLIT", "The trials are now split differently from when this run was "
                              "trained (the dataset or the settings changed).", title=_TITLE)
        model, _task, tensors, (_norm, values, valid) = model_from_checkpoint(prep, fold, device, ckpt)
        y = prep.sel.y[fold.test]
        keep = np.isin(y, pair)
        rows = fold.test[keep]
        if opts.max_trials and len(rows) > opts.max_trials // n_folds:
            take = max(opts.max_trials // n_folds, 2)
            rows = rows[np.linspace(0, len(rows) - 1, take).round().astype(int)]
        y01 = (prep.sel.y[rows] == pair[1]).astype(int)
        if y01.min() == y01.max():
            raise ConfigError("E_INTERP_ONECLASS", f"Fold {fold.index + 1}'s test trials contain only one of the "
                              "two classes.", title=_TITLE)
        att = attribution.compute(model, tensors, rows, classes=pair, batch_size=opts.batch_size,
                                  per_trial_u=opts.per_trial_u)
        w = head_weights(att.G, y01)
        grid = prep.grid
        Gt = attribution.to_rtm(reduce_heads(att.G, w), grid, C, M)
        roll = attribution.to_rtm(att.rollout, grid, C, M)
        grad = np.nan_to_num(grid.from_tokens(att.grad_input, C, M))
        X = np.nan_to_num(grid.from_tokens(np.where(valid[rows][..., None], values[rows], np.nan), C, M))
        G = att.G if att.G.nbytes * n_folds < G_MEMORY_LIMIT else None
        out.append(FoldAttribution(fold.index + 1, rows, y01, G, w, Gt, roll, grad, X, att.zeta))
        out[-1].model, out[-1].tensors = model, tensors  # kept for occlusion; dropped before writing
    return prep, pair, out, device


def _tensor(Gt: np.ndarray, y01: np.ndarray, mode: str) -> np.ndarray:
    return discriminative(Gt, y01) if mode == "discriminative" else np.abs(Gt)


def _ground_truth(prep) -> dict | None:
    prov = prep.ds.meta.get("provenance") or {}
    bench = prov.get("benchmark") if isinstance(prov, dict) else None
    if not bench or prep.ds.path is None:
        return None
    path = Path(prep.ds.path).with_name(bench["ground_truth"])
    if not path.exists():
        return None
    from scipy.io import loadmat

    g = loadmat(path, squeeze_me=False)
    names = [str(np.ravel(x)[0]) for x in np.ravel(g["pathway_names"])]
    used = np.ravel(g["generates_label"]).astype(bool)
    return {"motifs": g["motifs"].astype(float), "supports": g["supports"].astype(bool), "names": names,
            "generates_label": used, "system": bench.get("system")}


def interpret(run_dir: str | Path, opts: InterpretOptions | None = None) -> Path:
    opts = opts or InterpretOptions()
    run_dir = Path(run_dir)
    out_dir = run_dir / "interpret"
    out_dir.mkdir(exist_ok=True)
    status = StatusWriter(out_dir / "status.json", task="interpret", run_dir=str(run_dir))
    status.update("preparing", force=True, message="Loading the run")
    t0 = time.time()
    try:
        result = _interpret(run_dir, out_dir, opts, status)
        status.update("finished", force=True, message="Finished", elapsed_s=round(time.time() - t0, 1))
        return result
    except Exception as exc:  # the app reads the reason from status.json
        err = exc.to_dict() if hasattr(exc, "to_dict") else {
            "code": "E_INTERNAL", "title": "PRISMt interpretation hit an unexpected error", "message": str(exc),
            "hint": "Please report this, with the run folder.", "field": ""}
        status.fail(err, "failed")
        raise


def _interpret(run_dir: Path, out_dir: Path, opts: InterpretOptions, status: StatusWriter) -> Path:
    prep, pair, folds, _device = attribute_run(run_dir, opts, status)
    gt = _ground_truth(prep)
    mode = opts.mode if opts.mode != "auto" else ("discriminative" if gt is not None else "trials")
    restarts = opts.restarts or (20 if mode == "discriminative" else 5)
    Gt = np.concatenate([f.Gt for f in folds])
    y01 = np.concatenate([f.y01 for f in folds])
    fold_of = np.concatenate([np.full(len(f.rows), f.fold) for f in folds])
    rows = np.concatenate([f.rows for f in folds])
    C, T, M = Gt.shape[1:]
    exists = attribution.exists_mask(prep.grid, C, M)
    T_main = _tensor(Gt, y01, mode)

    status.update("evaluating", force=True, message="Choosing the number of motifs")
    curve = None
    if opts.rank == "auto":
        disc = discriminative(Gt, y01)
        curve = cp.rank_curve(disc, list(range(1, opts.max_rank + 1)), restarts=max(3, restarts // 4),
                              seed=opts.seed)
        rank = cp.choose_rank(curve)
    else:
        rank = int(opts.rank)
    status.update(message=f"Decomposing into {rank} motifs")
    main = cp.fit(T_main, rank, restarts=restarts, seed=opts.seed)
    core = _core(main)
    motifs = core.motifs()                                       # [K, C, T, M] unit scale
    expression = main.factors[0] * main.weights if mode == "trials" else None

    # stability across folds
    fold_stab = None
    if len(folds) > 1:
        fits = [_core(cp.fit(_tensor(f.Gt, f.y01, mode), rank, restarts=max(3, restarts // 2), seed=opts.seed))
                for f in folds if len(f.rows) >= 10]
        fold_stab = align.consensus(core, fits)["stability"]

    # bootstrap (on the discriminative tensor: cheap, and what the modality intervals describe)
    status.update(message=f"Bootstrap ({opts.bootstrap} draws)")
    disc_main = _core(cp.fit(discriminative(Gt, y01), rank, restarts=restarts, seed=opts.seed)) \
        if mode != "discriminative" else core
    boot_mod = np.zeros((opts.bootstrap, rank, M))
    boot_sim = np.zeros((opts.bootstrap, rank))
    boot_draw_fits = []
    starts = np.cumsum([0] + [len(f.rows) for f in folds])
    for b, idx in enumerate(stats.bootstrap_indices(y01, fold_of, opts.bootstrap, seed=opts.seed + 1)):
        parts = []
        for fi, f in enumerate(folds):
            sel = idx[(idx >= starts[fi]) & (idx < starts[fi + 1])] - starts[fi]
            if f.G is not None:
                w = head_weights(f.G[sel], f.y01[sel])
                parts.append(attribution.to_rtm(reduce_heads(f.G[sel], w), prep.grid, C, M))
            else:
                parts.append(f.Gt[sel])
        Gb = np.concatenate(parts)
        fit_b = _core(cp.fit(discriminative(Gb, y01[idx]), rank, restarts=max(3, restarts // 4),
                             seed=opts.seed + 7 + b))
        m_idx, sim = align.match(disc_main, fit_b)
        boot_sim[b] = sim
        for k, j in enumerate(m_idx):
            if j >= 0:
                boot_mod[b, k] = fit_b.factors[2][:, j]
        boot_draw_fits.append(fit_b)
        if b % 10 == 0:
            status.update(message=f"Bootstrap {b + 1}/{opts.bootstrap}")
    # map the discriminative fit's components onto the main fit's for reporting
    to_main, _ = align.match(core, disc_main)
    modality_calls = []
    for k in range(rank):
        j = to_main[k]
        draws = boot_mod[:, j] if j >= 0 and opts.bootstrap else core.factors[2][None, :, k]
        modality_calls.append({"point": cp.modality_call(core.factors[2][:, k]),
                               "interval": cp.modality_call_interval(draws) if opts.bootstrap else None,
                               "weights": core.factors[2][:, k].tolist(),
                               "weights_lo": np.quantile(draws, 0.025, 0).tolist(),
                               "weights_hi": np.quantile(draws, 0.975, 0).tolist()})
    stability_boot = [float(boot_sim[:, to_main[k]].mean()) if to_main[k] >= 0 and opts.bootstrap else None
                      for k in range(rank)]

    # which channels carry significant class-dependent attribution
    status.update(message=f"Permutation test ({opts.permutations} relabellings)")
    level = prep.plan.label_level
    groups_all = {"subject": prep.subject, "session": prep.session}.get(level)
    groups = None if groups_all is None else groups_all[rows]
    sig = stats.region_permutation(Gt, y01, groups, n_perm=opts.permutations, seed=opts.seed, exists=exists)

    summary = {
        "schema": "prismt.interpret/1", "run_dir": str(run_dir), "created": utc_now(),
        "classes": [prep.sel.class_names[pair[0]], prep.sel.class_names[pair[1]]],
        "mode": mode, "rank": rank, "rank_curve": curve, "restarts": restarts,
        "background_removed": main.dropped_background, "cp_error": main.error,
        "n_trials": int(len(rows)), "n_folds": len(folds),
        "head_weights": [f.w.tolist() for f in folds],
        "motifs": [{"index": k + 1, "weight": float(core.weights[k]), "modality": modality_calls[k],
                    "stability_bootstrap": stability_boot[k],
                    "stability_folds": None if fold_stab is None else float(fold_stab[k])} for k in range(rank)],
        "significant_regions": {
            prep.ds.modality_names[prep.sel.modalities[m]]: [
                prep.ds.channel_names[prep.sel.channels[c]] for c in range(C) if sig["p"][c, m] < 0.05]
            for m in range(M)},
        "permutations": sig["n_perm"], "permuted_by": level, "bootstrap": opts.bootstrap,
    }

    mat = {
        "format": "prismt.interpret/1", "classes": summary["classes"], "mode": mode,
        "channel_names": [prep.ds.channel_names[i] for i in prep.sel.channels],
        "modality_names": [prep.ds.modality_names[i] for i in prep.sel.modalities],
        "bin_centers_s": prep.bins.centers_s, "exists": exists,
        "motif_spatial": core.factors[0], "motif_temporal": core.factors[1], "motif_modality": core.factors[2],
        "motif_weight": core.weights, "motifs": motifs,
        "modality_lo": np.array([m["weights_lo"] for m in modality_calls]),
        "modality_hi": np.array([m["weights_hi"] for m in modality_calls]),
        "attribution_mean_signed": Gt[y01 == 1].mean(0) - Gt[y01 == 0].mean(0),
        "attribution_discriminative": discriminative(Gt, y01),
        "region_statistic": sig["statistic"], "region_p": sig["p"], "region_q": sig["q"],
        "trial_index": (prep.sel.trial_index[rows] + 1).astype(float), "trial_fold": fold_of.astype(float),
        "trial_class": (y01 + 1).astype(float), "zeta": np.concatenate([f.zeta for f in folds]),
        # folds of a best-model search can differ in depth and heads: one array per fold
        "head_weights": np.array([f.w for f in folds] + [None], dtype=object)[:-1],
    }
    if expression is not None:
        mat["motif_expression"] = expression
    if prep.subject is not None:
        mat["trial_subject"] = prep.subject[rows].astype(object)
    if prep.session is not None:
        mat["trial_session"] = prep.session[rows].astype(object)

    # does the model use each motif? occlude its footprint and compare with random occlusion
    if opts.test_motifs:
        status.update(message="Testing each motif by occlusion")
        tests, supports = _test_motifs(motifs, folds, prep, pair, exists, opts.seed)
        for k, t in enumerate(tests):
            summary["motifs"][k]["test"] = t
        mat["motif_support"] = supports
        mat["motif_test_cost"] = np.array([t["cost"] for t in tests])
        mat["motif_test_control"] = np.array([t["control_mean"] for t in tests])
        mat["motif_test_control_sd"] = np.array([t["control_sd"] for t in tests])
        mat["motif_test_z"] = np.array([np.nan if t["z"] is None else t["z"] for t in tests])
        mat["motif_test_fraction"] = np.array([np.nan if t["fraction_of_margin"] is None else t["fraction_of_margin"]
                                               for t in tests])

    # comparison methods and, for the benchmark, recovery and faithfulness
    if opts.compare or gt is not None:
        status.update(message="Comparison methods")
        X = np.concatenate([f.X for f in folds])
        roll = np.concatenate([f.rollout for f in folds])
        grad = np.concatenate([f.grad for f in folds])
        maps = {"prismt": {"map": discriminative(Gt, y01), "motifs": motifs * core.weights[:, None, None, None]}}
        maps.update(baselines.model_maps(roll, grad, y01, rank, seed=opts.seed))
        maps.update(baselines.observation_maps(X, y01, rank, seed=opts.seed))
        mat["method_names"] = list(maps)
        mat["method_maps"] = np.stack([maps[k]["map"] for k in maps])
        summary["methods"] = list(maps)
        if gt is not None:
            summary["ground_truth"] = _score_against_truth(maps, gt, boot_draw_fits, folds, prep, y01, opts, status)
            mat["truth_motifs"] = gt["motifs"]
            mat["truth_supports"] = gt["supports"]
            mat["truth_names"] = gt["names"]

    write_json(out_dir / "interpret.json", to_jsonable(summary))
    write_results_mat(out_dir / "interpret.mat", mat)
    arrays = {"G_tilde": Gt, "y01": y01, "trial_index": prep.sel.trial_index[rows], "fold": fold_of}
    if len(folds) == 1 and folds[0].G is not None and folds[0].G.nbytes < 2**28:
        # per-head tensors and the comparison inputs, for bootstrap reports (benchmark-sized runs)
        f = folds[0]
        arrays.update(G_heads=attribution.to_rtm(f.G, prep.grid, C, M), rollout=f.rollout, grad=f.grad, X=f.X)
    np.savez_compressed(out_dir / "attribution.npz", **arrays)
    return out_dir


MOTIF_SUPPORT = 0.3   # a motif's footprint: where it exceeds this fraction of its peak
MOTIF_MIN_EFFECT = 0.02  # "used" needs at least this share of the model's mean margin


def _test_motifs(motifs: np.ndarray, folds: list, prep, pair: tuple[int, int], exists: np.ndarray, seed: int):
    """Occlude each motif's footprint in the model's input and measure the loss of margin."""
    C, T, M = motifs.shape[1:]
    avail_rtm = np.broadcast_to(exists[:, None, :], (C, T, M))
    available = faithfulness.element_mask(avail_rtm, prep.grid)
    fold_args = [(f.model, f.tensors, f.rows, f.y01) for f in folds]
    tests, supports = [], np.zeros(motifs.shape, dtype=bool)
    for k, mt in enumerate(motifs):
        sup = (mt > MOTIF_SUPPORT * mt.max()) & avail_rtm if mt.max() > 0 else np.zeros_like(avail_rtm)
        supports[k] = sup
        mask = faithfulness.element_mask(sup, prep.grid)
        if not mask.any():
            tests.append({"cost": 0.0, "control_mean": 0.0, "control_sd": 0.0, "z": None, "n_elements": 0,
                          "margin": None, "fraction_of_margin": None, "used": None,
                          "answer": "The motif has no footprint to test."})
            continue
        t = faithfulness.occlusion_pooled(fold_args, pair, mask, available=available, seed=seed + 31 * k)
        z = t["z"]
        frac = t["fraction_of_margin"]
        reliable = z is not None and z > 2 and t["cost"] > 0
        # a reliable but tiny effect is not "use": require at least 2% of the model's evidence
        t["used"] = bool(reliable and (frac is None or frac >= MOTIF_MIN_EFFECT))
        share = "" if frac is None else f" ({100 * frac:.0f}% of its evidence)"
        control = f"{t['control_mean']:.2f} for as many random values"
        if t["used"]:
            t["answer"] = (f"The model uses it: removing these {t['n_elements']} values lowers the model's "
                           f"confidence by {t['cost']:.2f}{share}, against {control} (z = {z:.1f}).")
        elif reliable:
            t["answer"] = (f"The model hardly uses it: removing these {t['n_elements']} values lowers its confidence "
                           f"by only {t['cost']:.3f}{share}.")
        else:
            t["answer"] = (f"No evidence that the model uses it: removing these {t['n_elements']} values changes the "
                           f"model's confidence by {t['cost']:.2f}, against {control}"
                           + (f" (z = {z:.1f})." if z is not None else "."))
        tests.append(t)
    return tests, supports


def _score_against_truth(maps, gt, boot_fits, folds, prep, y01, opts, status) -> dict:
    used = gt["generates_label"]
    truth = gt["motifs"][used]
    supports = gt["supports"][used]
    shape = truth.shape[1:]
    out = {"system": gt["system"], "pathways": gt["names"], "generates_label": used.tolist(), "methods": {}}
    for name, entry in maps.items():
        out["methods"][name] = metrics.recovery(entry["motifs"], truth, supports, entry["map"])
    out["flat_control"] = metrics.flat_control(shape, truth, supports)
    # bootstrap interval for PRISMt's recovery
    draws = [metrics.recovery(f.motifs() * f.weights[:, None, None, None], truth, supports) for f in boot_fits]
    out["prismt_bootstrap"] = {k: stats.interval([d[k] for d in draws])
                               for k in ("motif_correlation", "unit_auroc", "mass_enrichment")}
    if opts.faithfulness:
        status.update(message="Occlusion of each pathway")
        occl = []
        for k, sup in enumerate(gt["supports"]):
            per_fold = []
            for f in folds:
                mask = faithfulness.element_mask(sup, prep.grid)
                per_fold.append(faithfulness.occlusion(f.model, f.tensors, f.rows, f.y01, _class_pair(prep.sel, opts),
                                                       mask, seed=opts.seed + k))
            occl.append({"pathway": gt["names"][k], "cost": float(np.mean([p["cost"] for p in per_fold])),
                         "z": float(np.mean([p["z"] for p in per_fold if p["z"] is not None] or [0.0]))})
        importance = faithfulness.pathway_importance(occl)
        out["occlusion"] = occl
        out["importance"] = importance.tolist()
        out["causal"] = {name: faithfulness.causal_scores(entry["map"], gt["supports"], importance)
                         for name, entry in maps.items()}
    return out

