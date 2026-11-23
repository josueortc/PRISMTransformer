"""A run's result as answers to the questions an experimenter asks, each with a status.

Every finished run gets a ``verdict`` in ``metrics.json``: a short list of questions
("Above chance?", "Better than a linear model?", "Holds in most subjects?", "Could a
difference between sessions explain it?", "Was the model trained enough?"), each with a
one-word answer, a status (``good``, ``caution``, ``bad`` or ``info``) and one sentence of
detail with the numbers behind it. The app shows it as the first thing in the Results tab.

The statuses are deliberately conservative: an interval that includes "no effect" is a
caution, never a yes.
"""

from __future__ import annotations

import math

import numpy as np


def _item(key: str, question: str, answer: str, status: str, detail: str) -> dict:
    return {"key": key, "question": question, "answer": answer, "status": status, "detail": detail}


def balanced_accuracy(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    rec = [float(np.mean(pred[y == k] == k)) for k in range(n_classes) if (y == k).any()]
    return float(np.mean(rec)) if rec else float("nan")


def score_interval(y: np.ndarray, pred: np.ndarray, groups: np.ndarray | None, n_classes: int, *,
                   n_boot: int = 2000, seed: int = 0) -> dict:
    """95% bootstrap interval of the pooled balanced accuracy, resampling whole groups
    (subjects, else sessions) when there are at least three, else trials."""
    rng = np.random.default_rng(seed)
    vals = []
    if groups is not None and len(set(groups.tolist())) >= 3:
        keys = np.unique(groups)
        rows = {k: np.flatnonzero(groups == k) for k in keys}
        for _ in range(n_boot):
            idx = np.concatenate([rows[k] for k in rng.choice(keys, size=len(keys), replace=True)])
            vals.append(balanced_accuracy(y[idx], pred[idx], n_classes))
        unit = "group"
    else:
        n = len(y)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            vals.append(balanced_accuracy(y[idx], pred[idx], n_classes))
        unit = "trial"
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return {"lo": float(lo), "hi": float(hi), "resampled": unit, "n_boot": n_boot}


def training_stats(outcomes: list, task: str, n_classes: int) -> list[dict]:
    """Per fold: how training ended and how far the loss moved (None where no history was kept)."""
    out = []
    for o in outcomes:
        fit = o.fit
        hist = getattr(fit, "history", None) or []
        entry = {"fold": o.fold.index + 1, "stopped": getattr(fit, "stopped", None),
                 "best_epoch": getattr(fit, "best_epoch", None), "epochs_run": getattr(fit, "epochs_run", None),
                 "train_loss_first": None, "train_loss_min": None, "val_loss_first": None, "val_loss_min": None}
        tr = [r["train_loss"] for r in hist if r.get("train_loss") is not None and np.isfinite(r["train_loss"])]
        va = [r["val_loss"] for r in hist if r.get("val_loss") is not None and np.isfinite(r["val_loss"])]
        if tr:
            entry["train_loss_first"], entry["train_loss_min"] = float(tr[0]), float(min(tr))
        if va:
            entry["val_loss_first"], entry["val_loss_min"] = float(va[0]), float(min(va))
        run, best = entry["epochs_run"], entry["best_epoch"]
        entry["still_improving"] = bool(entry["stopped"] == "max_epochs" and run and best is not None
                                        and best >= run - max(1, round(0.1 * run)))
        if task == "classify" and tr:
            start = math.log(max(n_classes, 2))     # cross-entropy of guessing
            entry["barely_learned"] = bool((start - min(tr)) / start < 0.10)
        elif va:
            entry["barely_learned"] = bool((va[0] - min(va)) / max(abs(va[0]), 1e-12) < 0.02)
        else:
            entry["barely_learned"] = None
        out.append(entry)
    return out


def trained_enough(stats: list[dict], *, reference: float | None = None, chance: float | None = None) -> dict:
    """The "Was the model trained enough?" item. ``reference``: a simple model's score on the
    same trials (when it is well above ``chance``, a model that barely learned was undertrained)."""
    q = "Was the model trained enough?"
    known = [s for s in stats if s["barely_learned"] is not None]
    if not known:
        return _item("trained", q, "Not checked", "info", "No training history was kept for this run.")
    n = len(known)
    barely = sum(bool(s["barely_learned"]) for s in known)
    improving = sum(bool(s["still_improving"]) for s in known)
    where = "" if n == 1 else f" in {{k}} of {n} folds"
    if barely * 2 >= n:
        detail = ("The model barely fitted even its training trials" + where.format(k=barely) + ". ")
        if reference is not None and chance is not None and reference > chance + 0.1:
            detail += (f"A simple reference model reaches {reference:.2f} on the same trials, so the information "
                       "is there: train longer (more epochs), use the Standard preset, or use Find the best model.")
        else:
            detail += ("Either training was too short or too small (try more epochs or the Standard preset), "
                       "or the recordings do not carry this information.")
        return _item("trained", q, "No", "bad", detail)
    if improving * 2 >= n:
        return _item("trained", q, "Probably not", "caution",
                     "Training was still improving when it reached the epoch limit" + where.format(k=improving) +
                     ". Raise 'Epochs (at most)' and train again; the score is likely to go up.")
    return _item("trained", q, "Yes", "good", "Training converged (it stopped improving before the epoch limit).")


def classification(*, y: np.ndarray, pred: np.ndarray, groups: np.ndarray | None, group_name: str, n_classes: int,
                   ba: float, baselines: dict, by_group: dict, stats: list[dict], seed: int = 0) -> tuple[list[dict], dict]:
    chance = 1.0 / n_classes
    ci = score_interval(y, pred, groups, n_classes, seed=seed)
    items = []
    # 1. above chance
    q = "Can the recordings tell the conditions apart (above chance)?"
    rng_txt = f"{ba:.2f} (95% interval {ci['lo']:.2f} to {ci['hi']:.2f}; chance {chance:.2f})"
    sh = baselines.get("shuffled_labels")
    extra = f" Shuffled-label test: p = {sh['p_value_model']:.3g}." if sh else ""
    if ci["lo"] > chance and (not sh or sh["p_value_model"] < 0.05):
        items.append(_item("above_chance", q, "Yes", "good", f"Balanced accuracy {rng_txt}." + extra))
    elif ba > chance + 0.02:
        items.append(_item("above_chance", q, "Not clearly", "caution",
                           f"Balanced accuracy {rng_txt}: above chance, but the interval includes chance." + extra))
    else:
        items.append(_item("above_chance", q, "No", "bad", f"Balanced accuracy {rng_txt}." + extra))
    # 2. against the linear model
    q = "Does the transformer read more than a linear model?"
    pv = baselines.get("paired_vs_logistic")
    if pv and pv.get("mean_difference") is not None:
        lo, hi = pv["ci"]
        log_ba = baselines["logistic"]["balanced_accuracy"]
        nums = (f"Transformer {ba:.2f}, logistic regression {log_ba:.2f}; difference {pv['mean_difference']:+.2f} "
                f"(95% interval {lo:+.2f} to {hi:+.2f}, per {pv['unit']}).")
        if lo > 0:
            items.append(_item("vs_linear", q, "Yes", "good", nums))
        elif hi < 0:
            items.append(_item("vs_linear", q, "No, the linear model is better", "caution",
                               nums + " The information is there, but a linear read-out finds it better than this "
                               "transformer did."))
        else:
            items.append(_item("vs_linear", q, "About the same", "info",
                               nums + " That is a finding too: no complex model is needed to read this out."))
    else:
        items.append(_item("vs_linear", q, "Not compared", "info", "The linear reference model was switched off."))
    # 3. across subjects
    rows = by_group.get("subject") or []
    if len(rows) >= 2:
        q = "Does it hold in most subjects?"
        vals = [(r["balanced_accuracy"] if r.get("classes_present", 0) > 1 else r["accuracy"]) for r in rows]
        vals = [v for v in vals if v is not None]
        above = sum(v > chance for v in vals)
        frac = above / max(len(vals), 1)
        detail = (f"Above chance in {above} of {len(vals)} subjects (range {min(vals):.2f} to {max(vals):.2f}). "
                  "See 'Accuracy per subject'.")
        if frac >= 0.75:
            items.append(_item("subjects", q, "Yes", "good", detail))
        elif frac >= 0.5:
            items.append(_item("subjects", q, "In some", "caution", detail))
        else:
            items.append(_item("subjects", q, "No", "bad", detail))
    # 4. session-level explanation
    so = baselines.get("session_only")
    if so is not None:
        q = "Could a difference between sessions alone explain it?"
        s = so["balanced_accuracy"]
        detail = (f"A model that sees only each session's average reaches {s:.2f} (transformer {ba:.2f}, chance "
                  f"{chance:.2f}).")
        if s > chance + 0.1 and s >= ba - 0.05:
            items.append(_item("sessions", q, "Possibly", "caution",
                               detail + " Recording-day differences (expression, bleaching, arousal) may carry the "
                               "result: try Scaling = zscore_session, or compare within sessions."))
        else:
            items.append(_item("sessions", q, "No", "good", detail))
    # 5. training
    ref = baselines.get("logistic", {}).get("balanced_accuracy")
    items.append(trained_enough(stats, reference=ref, chance=chance))
    return items, ci


def autoencoder(*, by_mask: dict, train_name: str, stats: list[dict]) -> list[dict]:
    entry = by_mask[train_name]
    r2 = entry.get("r2")
    base = {k: v.get("r2") for k, v in entry.get("baselines", {}).items() if v.get("r2") is not None}
    items = []
    q = "Are hidden values predictable from the rest?"
    if r2 is None:
        items.append(_item("predictable", q, "Not measured", "info", "No hidden values could be scored."))
    else:
        psth = base.get("psth")
        detail = f"R² = {r2:.2f} for hidden values" + (f" (the trial average alone: {psth:.2f})." if psth is not None else ".")
        if psth is not None and r2 > psth + 0.02 and r2 > 0:
            items.append(_item("predictable", q, "Yes", "good", detail))
        elif r2 > 0:
            items.append(_item("predictable", q, "Barely", "caution", detail + " Little beyond the trial average."))
        else:
            items.append(_item("predictable", q, "No", "bad", detail))
    if base and r2 is not None:
        q = "Does the transformer beat the simple reference models?"
        best = max(base, key=base.get)
        detail = f"Transformer R² {r2:.2f}; best reference ({best.replace('_', ' ')}) {base[best]:.2f}."
        if r2 > base[best] + 0.02:
            items.append(_item("vs_reference", q, "Yes", "good", detail))
        elif r2 >= base[best] - 0.02:
            items.append(_item("vs_reference", q, "About the same", "info", detail))
        else:
            hint = (" A linear combination of the other channels predicts the hidden ones better."
                    if best == "ridge" else "")
            items.append(_item("vs_reference", q, "No", "caution", detail + hint))
    items.append(trained_enough(stats))
    return items


def lines(items: list[dict]) -> list[str]:
    mark = {"good": "[ok]", "caution": "[!]", "bad": "[x]", "info": "[i]"}
    return [f"{mark.get(i['status'], '[i]')} {i['question']} {i['answer']}. {i['detail']}" for i in items]
