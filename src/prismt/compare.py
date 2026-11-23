"""Putting finished runs side by side: their scores, intervals and paired differences.

The recommended way to ask "does behaviour explain my neural result?" or "which signal
carries it?" is to train twice and compare. This module does the comparison: each run's
headline score with its interval, and, for classification runs that tested the same
trials, the difference to the first run per subject (bootstrap interval over subjects,
sign test), computed from the runs' saved predictions.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from prismt.jsonutil import read_json


def resolve(run_dir: str | Path) -> Path:
    """A run folder, the ``final/`` of a best-model search, or the ``results/`` of a job folder."""
    p = Path(run_dir)
    if (p / "results").is_dir() and (p / "job.env").exists():
        p = p / "results"
    if not (p / "metrics.json").exists() and (p / "final" / "metrics.json").exists():
        p = p / "final"
    return p


def describe(run_dir: str | Path) -> dict:
    p = resolve(run_dir)
    metrics = read_json(p / "metrics.json")
    cfg = read_json(p / "config.json")
    head = metrics["headline"]
    sel = cfg.get("selection") or {}
    return {"run_dir": str(p), "name": cfg.get("name") or p.name, "folder": p.name, "task": metrics["task"],
            "score": head["name"], "value": head["value"], "ci": head.get("ci"), "chance": head.get("chance"),
            "label": (cfg.get("labels") or {}).get("column"), "classes": metrics["selection"].get("class_names"),
            "n_trials": metrics["selection"]["n_trials"], "signals": sel.get("modalities") or "all",
            "channel_groups": sel.get("channel_groups") or "all", "test_on": metrics.get("test_on"),
            "n_folds": metrics.get("n_folds"), "summary": metrics.get("run_summary")}


def _predictions(run_dir: Path) -> dict | None:
    f = run_dir / "predictions.csv"
    if not f.exists():
        return None
    rows = {}
    with open(f, newline="") as fh:
        for r in csv.DictReader(fh):
            rows[int(r["trial_index"])] = (r["true_class"], r["predicted_class"], r.get("subject") or r.get("session") or "")
    return rows


def compare(run_dirs: list[str | Path]) -> dict:
    from prismt.run import paired_comparison

    runs = [describe(d) for d in run_dirs]
    notes, pairs = [], []
    if len({r["task"] for r in runs}) > 1:
        notes.append("The runs are of different kinds (classification and autoencoder); their scores are not comparable.")
    if len({str(r["label"]) for r in runs}) > 1 or len({str(r["classes"]) for r in runs}) > 1:
        notes.append("The runs do not predict the same classes; compare them with care.")
    ref = runs[0]
    ref_pred = _predictions(Path(ref["run_dir"])) if ref["task"] == "classify" else None
    for other in runs[1:]:
        if ref_pred is None or other["task"] != "classify" or other["classes"] != ref["classes"]:
            continue
        pred = _predictions(Path(other["run_dir"]))
        if pred is None:
            continue
        common = sorted(set(ref_pred) & set(pred))
        if len(common) < 0.8 * max(len(ref_pred), len(pred)) or any(ref_pred[t][0] != pred[t][0] for t in common):
            notes.append(f"'{other['name']}' was not tested on the same trials as '{ref['name']}': no paired comparison.")
            continue
        names = ref["classes"]
        y = np.array([names.index(ref_pred[t][0]) for t in common])
        a = np.array([names.index(pred[t][1]) for t in common])
        b = np.array([names.index(ref_pred[t][1]) for t in common])
        groups = np.array([ref_pred[t][2] for t in common], dtype=object)
        if len(set(groups.tolist())) < 2 or groups[0] == "":
            groups = None
        pc = paired_comparison(y, a, b, groups, len(names), ref_name=f"'{ref['name']}'", unit="subject")
        pc["line"] = pc["line"].replace("Transformer minus", f"'{other['name']}' minus")
        pairs.append({"run": other["name"], "reference": ref["name"], "n_common": len(common),
                      "difference": pc["mean_difference"], "ci": pc["ci"], "differs": bool(pc["ci"][0] > 0 or pc["ci"][1] < 0),
                      "line": pc["line"]})
    lines = [f"{r['name']}: {r['score'].replace('_', ' ')} {r['value']:.2f}" +
             (f" (95% interval {r['ci'][0]:.2f} to {r['ci'][1]:.2f})" if r.get("ci") else "") for r in runs
             if r["value"] is not None]
    lines += [p["line"] + ("" if p["differs"] else " The interval includes 0: no clear difference.") for p in pairs]
    return {"runs": runs, "pairs": pairs, "notes": notes, "summary_lines": lines + notes}
