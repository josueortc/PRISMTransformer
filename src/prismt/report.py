"""A PDF report of one run: the verdict, the key figures, the settings and a methods paragraph.

    python -m prismt report RUN_DIR            # writes RUN_DIR/report.pdf and report_methods.txt

Meant to be read by the person who ran the experiment and by their collaborators: page 1
answers the questions in words, the following pages show the figures behind the answers,
and the last page gives the settings and a methods paragraph that can be pasted into a
manuscript and edited. Text in the PDF stays editable (fonts are embedded as text).
"""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import numpy as np

from prismt import __version__
from prismt.compare import resolve
from prismt.errors import EnvironmentProblem
from prismt.jsonutil import read_json

STATUS_COLOR = {"good": "#0ca30c", "caution": "#e39a00", "bad": "#d03b3b", "info": "#52514e"}
BLUE, GREY, INK = "#2a78d6", "#b9b8b4", "#0b0b0b"
PAPER = ("Ortega Caro J, Batchelor HM, Lohani S, van Dijk D, Cardin JA. Selective changes in cortical cholinergic "
         "signaling during learning. bioRxiv (2025). doi:10.1101/2025.08.29.673096")


def _plt():
    try:
        import matplotlib
    except ImportError as exc:
        raise EnvironmentProblem("E_ENV_NO_MATPLOTLIB", "Reports need the matplotlib package.",
                                 hint="Install it with: pip install matplotlib (or recreate the PRISMT environment).") from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "pdf.fonttype": 42, "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.6, "axes.titlesize": 9})
    return plt


def methods_text(cfg: dict, metrics: dict, interp: dict | None = None) -> str:
    """A methods paragraph for this run, to paste into a manuscript and edit."""
    m, tr, pre, sp = cfg["model"], cfg["train"], cfg["preprocess"], cfg["split"]
    task = metrics["task"]
    scale = {"zscore_train": "z-scored per channel and signal with the mean and standard deviation of the training "
                             "trials", "zscore_session": "z-scored per channel and signal within each session",
             "none": "left unscaled"}[pre["normalize"]]
    patch = m.get("time_patch", 1)
    token = "one token per channel, signal and time bin" if patch == 1 else (
        "one token per channel and signal" if patch == "all" else f"one token per channel, signal and {patch} time bins")
    attn = "block-causal attention (each time bin attends to itself and earlier bins)" if m["attention"] == "block_causal" \
        else "full attention"
    what = {"subject": "subjects", "session": "sessions", "trial": "trials", "column": "a predefined set of trials"}[
        metrics.get("test_on", "trial")]
    n_folds = metrics.get("n_folds", 1)
    cv = (f"{n_folds}-fold cross-validation in which whole {what} were held out, every one tested exactly once"
          if n_folds > 1 else f"a single split holding out {what}")
    parts = [f"Trials were analysed with PRISMT (version {__version__}; https://github.com/josueortc/PRISMTransformer), which "
             f"implements the PRISMt transformer ({PAPER})."]
    win = pre.get("time_window_s")
    window = f" ({win[0]:g} to {win[1]:g} s)" if win else ""
    width = pre.get("bin_width_s")
    binned = f", averaged in {width:g} s bins" if width else ""
    parts.append(f"Each trial{window} was {scale}{binned} and tokenized as {token}; missing values were excluded "
                 "from attention.")
    parts.append(f"The model was a transformer encoder with {m['n_layers']} layers, width {m['d_model']}, "
                 f"{m['n_heads']} attention heads, {attn} and a learned summary (CLS) token.")
    opt = (f"AdamW (learning rate {tr['lr']:g}, weight decay {tr['weight_decay']:g}, batch size {tr['batch_size']}, "
           f"cosine schedule), for at most {tr['epochs']} epochs with early stopping on the validation loss "
           f"(patience {tr['patience']} epochs)")
    sel = metrics.get("model_selection")
    if task == "classify":
        names = metrics["selection"]["class_names"]
        label = (cfg.get("labels") or {}).get("column")
        pretrain = " after masked-autoencoder pretraining of the encoder" if m.get("init_from") else ""
        parts.append(f"It was trained to classify {label} ({' vs '.join(map(str, names))}){pretrain} with {opt}.")
        head = metrics["headline"]
        ci = head.get("ci")
        unit = "subjects" if head.get("ci_resampled") == "group" else "trials"
        interval = f" (95% bootstrap interval {ci[0]:.2f} to {ci[1]:.2f}, resampling {unit})" if ci else ""
        parts.append(f"Performance was estimated by {cv}, and is reported as balanced accuracy on held-out trials "
                     f"pooled over folds: {head['value']:.2f}{interval}; chance is {head['chance']:.2f}.")
        b = metrics.get("baselines", {})
        if "logistic" in b:
            s = (f"As a reference, L2-regularized logistic regression was fitted to the same preprocessed trials "
                 f"on the same splits (balanced accuracy {b['logistic']['balanced_accuracy']:.2f})")
            pv = b.get("paired_vs_logistic")
            if pv and pv.get("mean_difference") is not None:
                s += (f"; the transformer differed from it by {pv['mean_difference']:+.2f} per {pv['unit']} "
                      f"(95% bootstrap interval {pv['ci'][0]:+.2f} to {pv['ci'][1]:+.2f})")
            parts.append(s + ".")
        if "session_only" in b:
            parts.append(f"Logistic regression on session averages alone reached "
                         f"{b['session_only']['balanced_accuracy']:.2f}.")
        if "shuffled_labels" in b:
            sh = b["shuffled_labels"]
            parts.append(f"A null distribution was obtained by refitting logistic regression {sh['n']} times with "
                         f"training labels shuffled across {sh['shuffled_by']}s (p = {sh['p_value_model']:.3g}).")
    else:
        mask = cfg["mae"]["mask"]
        strategy = mask["strategy"]
        hidden = f", {100 * mask['ratio']:.0f}% of tokens hidden" if strategy == "random" else ""
        parts.append(f"It was trained as a masked autoencoder ({strategy} masking{hidden}) with {opt}; the loss was "
                     "the mean squared error on hidden values.")
        head = metrics["headline"]
        best_name = str(head.get("baseline")).replace("_", " ")
        parts.append(f"Performance was estimated by {cv} and is reported as R² of hidden values on held-out trials "
                     f"({head['value']:.2f}; best simple reference, {best_name}: "
                     f"{head.get('baseline_value'):.2f}). Reference predictions were the trial average, the mean of "
                     "visible values, the last visible value, linear interpolation, and a linear (ridge) prediction "
                     "from the other channels.")
    if sel:
        how = ("population-based training" if sel.get("method") == "pbt" else "a tree-structured Parzen estimator search")
        parts.append(f"Model settings were selected within each cross-validation fold by {how}, using only that "
                     "fold's training and validation trials (nested selection).")
    if interp:
        parts.append(f"Predictions on held-out trials were interpreted with PRISMt: attention rollout from the summary "
                     f"token was combined with each token's signed value message to give an attribution over "
                     f"channel, time and signal for every layer and head; layers and heads were weighted by their "
                     f"class-discriminative magnitude; and the result was factorized into {interp['rank']} motifs by "
                     f"non-negative CP decomposition with the signal factor constrained to sum to 1. Channels with "
                     f"class-dependent attribution were identified by {interp['permutations']} label permutations "
                     f"(p < 0.05), intervals by {interp['bootstrap']} bootstrap resamples of held-out trials, and "
                     "each motif was tested by occluding its footprint against size-matched random occlusion.")
    return " ".join(parts)


def _wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width))


def _page_title(fig, title: str, subtitle: str = "") -> None:
    fig.text(0.07, 0.955, title, fontsize=14, fontweight="bold", color=INK, va="top")
    if subtitle:
        fig.text(0.07, 0.925, subtitle, fontsize=8, color="#52514e", va="top")
    fig.text(0.93, 0.02, f"PRISMT {__version__}", fontsize=6, color="#8a8985", ha="right")


def _first_page(plt, pdf, name: str, cfg: dict, metrics: dict) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    _page_title(fig, f"PRISMT report: {name}",
                f"{metrics['task']} · dataset {Path(cfg['dataset']['path']).name} · {metrics['selection']['n_trials']} trials")
    y = 0.88
    fig.text(0.07, y, "What the result says", fontsize=11, fontweight="bold")
    y -= 0.03
    for it in metrics.get("verdict", []):
        col = STATUS_COLOR.get(it["status"], "#52514e")
        fig.text(0.075, y + 0.002, "\u25cf", fontsize=11, color=col, va="top")
        q = fig.text(0.105, y, it["question"], fontsize=9, fontweight="bold", color=INK, va="top")
        fig.canvas.draw()
        x1 = q.get_window_extent().x1 / fig.bbox.width
        fig.text(x1 + 0.012, y, it["answer"], fontsize=9, fontweight="bold", color=col, va="top")
        detail = _wrap(it["detail"], 112)
        fig.text(0.105, y - 0.018, detail, fontsize=8, color="#3a3936", va="top")
        y -= 0.03 + 0.0135 * (detail.count("\n") + 1)
    y -= 0.015
    fig.text(0.07, y, "Summary", fontsize=11, fontweight="bold")
    y -= 0.025
    for line in metrics.get("summary_lines", []):
        t = _wrap(line, 118)
        fig.text(0.075, y, t, fontsize=8, color=INK, va="top")
        y -= 0.0135 * (t.count("\n") + 1) + 0.006
    warns = [w["message"] for w in metrics.get("warnings", []) if w.get("severity", "warning") == "warning"]
    if warns and y > 0.12:
        y -= 0.01
        fig.text(0.07, y, "Warnings", fontsize=11, fontweight="bold")
        y -= 0.025
        for w in warns[:8]:
            t = _wrap("- " + w, 118)
            fig.text(0.075, y, t, fontsize=8, color="#8a5a00", va="top")
            y -= 0.0135 * (t.count("\n") + 1) + 0.004
    pdf.savefig(fig)
    plt.close(fig)


def _history(run_dir: Path) -> dict:
    f = run_dir / "history.csv"
    out: dict[int, list[dict]] = {}
    if not f.exists():
        return out
    with open(f, newline="") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(int(float(r["fold"])), []).append(r)
    return out


def _curves(ax, hist: dict) -> None:
    for k, rows in sorted(hist.items()):
        ep = [int(float(r["epoch"])) for r in rows]
        ax.plot(ep, [float(r["train_loss"]) for r in rows], color=GREY, lw=0.8)
        ax.plot(ep, [float(r["val_loss"]) for r in rows], color=BLUE, lw=0.9)
    ax.plot([], [], color=GREY, label="training")
    ax.plot([], [], color=BLUE, label="validation")
    ax.legend(frameon=False, fontsize=7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Learning curves (one line per fold)")


def _figures_classify(plt, pdf, run_dir: Path, metrics: dict) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(8.27, 8.2))
    fig.subplots_adjust(top=0.88, hspace=0.45, wspace=0.4, left=0.2, right=0.95)
    _page_title(fig, "Score and where it comes from")
    b = metrics["baselines"]
    head = metrics["headline"]
    rows = [("PRISMT", head["value"], BLUE)]
    for key, lab in (("logistic", "logistic regression"), ("session_only", "session averages only")):
        if key in b:
            rows.append((lab, b[key]["balanced_accuracy"], GREY))
    if "shuffled_labels" in b:
        rows.append(("shuffled labels (95th pct)", b["shuffled_labels"]["p95"], GREY))
    rows.append(("chance", head["chance"], GREY))
    ax = axs[0, 0]
    ax.barh(range(len(rows)), [r[1] for r in rows], color=[r[2] for r in rows], height=0.6)
    if head.get("ci"):
        ax.errorbar(head["value"], 0, xerr=[[head["value"] - head["ci"][0]], [head["ci"][1] - head["value"]]],
                    color=INK, capsize=2, lw=0.8)
    for i, r in enumerate(rows):
        ax.text(min(r[1] + 0.02, 0.9), i, f"{r[1]:.2f}", va="center", fontsize=7)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.axvline(head["chance"], ls=":", color="#52514e", lw=0.6)
    ax.set_xlabel("Balanced accuracy on held-out trials")
    ax.set_title("Score vs reference models")
    ax = axs[0, 1]
    groups = (metrics.get("by_group") or {}).get("subject") or (metrics.get("by_group") or {}).get("session") or []
    if groups:
        vals = [g["balanced_accuracy"] if g.get("classes_present", 0) > 1 else g["accuracy"] for g in groups]
        order = np.argsort(vals)
        ax.scatter([vals[i] for i in order], range(len(vals)), color=BLUE, s=14)
        ax.set_yticks(range(len(vals)), [groups[i]["group"] for i in order], fontsize=6 if len(vals) > 12 else 7)
        ax.axvline(head["chance"], ls=":", color="#52514e", lw=0.6)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("Balanced accuracy")
        ax.set_title("Per subject" if (metrics.get("by_group") or {}).get("subject") else "Per session")
    else:
        ax.axis("off")
    ax = axs[1, 0]
    conf = np.asarray(metrics["test"]["confusion"], float)
    frac = conf / np.maximum(conf.sum(1, keepdims=True), 1)
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    names = metrics["selection"]["class_names"]
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{frac[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if frac[i, j] > 0.5 else INK)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (fractions of each true class)")
    for sp in ax.spines.values():
        sp.set_visible(False)
    _curves(axs[1, 1], _history(run_dir))
    pdf.savefig(fig)
    plt.close(fig)


def _figures_mae(plt, pdf, run_dir: Path, metrics: dict) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(8.27, 4.6))
    fig.subplots_adjust(top=0.8, wspace=0.5, left=0.2, right=0.95, bottom=0.15)
    _page_title(fig, "Reconstruction of hidden values")
    masks = list(metrics["test"])
    r2 = [metrics["test"][k]["r2"] or 0 for k in masks]
    best = []
    for k in masks:
        bl = {n: v["r2"] for n, v in metrics["test"][k]["baselines"].items() if v["r2"] is not None}
        n = max(bl, key=bl.get) if bl else None
        best.append((n, bl.get(n, 0)))
    ax = axs[0]
    y = np.arange(len(masks))
    ax.barh(y - 0.2, r2, height=0.38, color=BLUE, label="PRISMT")
    ax.barh(y + 0.2, [v for _, v in best], height=0.38, color=GREY, label="best simple reference")
    for i, (n, v) in enumerate(best):
        ax.text(max(v, 0) + 0.02, i + 0.2, str(n).replace("_", " "), va="center", fontsize=6, color="#52514e")
    ax.set_yticks(y, [k.replace("_", " ") for k in masks])
    ax.invert_yaxis()
    ax.axvline(0, color="#52514e", lw=0.6)
    ax.set_xlabel("R² of hidden values (held-out trials)")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.set_title("Each hiding pattern vs its best reference")
    _curves(axs[1], _history(run_dir))
    pdf.savefig(fig)
    plt.close(fig)


def _figures_interpret(plt, pdf, run_dir: Path, interp: dict) -> None:
    from scipy.io import loadmat

    m = loadmat(run_dir / "interpret" / "interpret.mat")
    motifs = np.asarray(m["motifs"], float)
    if motifs.ndim == 3:
        motifs = motifs[..., None]
    K, C, T, M = motifs.shape
    mods = [str(np.ravel(x)[0]) for x in np.ravel(m["modality_names"])]
    n_show = min(K, 4)
    fig = plt.figure(figsize=(8.27, 2.6 + 2.3 * n_show))
    _page_title(fig, "PRISMt: what the classifier uses",
                f"{interp['classes'][0]} vs {interp['classes'][1]} · {K} motifs · held-out trials only")
    gs = fig.add_gridspec(n_show, 2, width_ratios=[1.25, 1], left=0.09, right=0.96, top=1 - 1.1 / fig.get_figheight(),
                          bottom=0.6 / fig.get_figheight(), hspace=0.75, wspace=0.12)
    for k in range(n_show):
        ax = fig.add_subplot(gs[k, 0])
        img = np.concatenate([np.concatenate([motifs[k, :, :, j], np.full((C, 1), np.nan)], 1) for j in range(M)], 1)[:, :-1]
        ax.imshow(img, aspect="auto", cmap="Blues", origin="lower")
        ax.set_xticks([j * (T + 1) + (T - 1) / 2 for j in range(M)], mods)
        ax.set_ylabel("Channel")
        info = interp["motifs"][k]
        w = info["modality"]["weights"]
        ax.set_title(f"Motif {k + 1}: " + ", ".join(f"{n} {100 * v:.0f}%" for n, v in zip(mods, w)), loc="left")
        tx = fig.add_subplot(gs[k, 1])
        tx.axis("off")
        lines = []
        call = info["modality"].get("interval") or info["modality"]["point"]
        if str(call).startswith("unique:"):
            call = f"specific to {mods[int(str(call).split(':')[1])]}"
        lines.append(f"Signal: {call}")
        if info.get("stability_bootstrap") is not None:
            lines.append(f"Stability across resamples: {info['stability_bootstrap']:.2f}")
        if info.get("test"):
            lines.append(info["test"]["answer"])
        tx.text(0, 0.95, "\n\n".join(_wrap(s, 52) for s in lines), va="top", fontsize=7.5, color=INK)
    sig = interp.get("significant_regions") or {}
    txt = "; ".join(f"{k}: {', '.join(v) if v else 'none'}" for k, v in sig.items())
    fig.text(0.09, 0.25 / fig.get_figheight(), _wrap("Channels with significant class-dependent attribution (p < 0.05): " + txt, 130),
             fontsize=7, va="bottom")
    pdf.savefig(fig)
    plt.close(fig)


def _last_page(plt, pdf, cfg: dict, methods: str) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    _page_title(fig, "Methods and settings")
    fig.text(0.07, 0.9, "Methods paragraph (paste and edit; also saved as report_methods.txt)", fontsize=10, fontweight="bold")
    wrapped = _wrap(methods, 118)
    fig.text(0.07, 0.88, wrapped, fontsize=8, va="top", color=INK, linespacing=1.35)
    y = 0.88 - 0.0158 * (wrapped.count("\n") + 2) - 0.02
    fig.text(0.07, y, "Settings", fontsize=10, fontweight="bold")
    y -= 0.022
    rows = []
    for section in ("preprocess", "split", "model", "mae", "train", "baselines", "hpo"):
        sec = cfg.get(section)
        if not isinstance(sec, dict) or (section == "mae" and cfg["task"] != "mae"):
            continue
        for k, v in sec.items():
            if isinstance(v, dict):
                rows += [(f"{section}.{k}.{kk}", vv) for kk, vv in v.items()]
            else:
                rows.append((f"{section}.{k}", v))
    half = (len(rows) + 1) // 2
    for col, chunk in enumerate((rows[:half], rows[half:])):
        yy = y
        for k, v in chunk:
            if yy < 0.05:
                break
            fig.text(0.07 + 0.45 * col, yy, k, fontsize=6.5, color="#52514e", va="top")
            fig.text(0.27 + 0.45 * col, yy, str(v)[:38], fontsize=6.5, color=INK, va="top")
            yy -= 0.0125
    pdf.savefig(fig)
    plt.close(fig)


def report(run_dir: str | Path, out: str | Path | None = None) -> Path:
    plt = _plt()
    from matplotlib.backends.backend_pdf import PdfPages

    run_dir = resolve(run_dir)
    metrics = read_json(run_dir / "metrics.json")
    cfg = read_json(run_dir / "config.json")
    interp = read_json(run_dir / "interpret" / "interpret.json") if (run_dir / "interpret" / "interpret.json").exists() else None
    name = cfg.get("name") or run_dir.name
    out = Path(out) if out else run_dir / "report.pdf"
    methods = methods_text(cfg, metrics, interp)
    out.with_name("report_methods.txt").write_text(methods + "\n", encoding="utf-8")
    with PdfPages(out) as pdf:
        _first_page(plt, pdf, name, cfg, metrics)
        if metrics["task"] == "classify":
            _figures_classify(plt, pdf, run_dir, metrics)
        else:
            _figures_mae(plt, pdf, run_dir, metrics)
        if interp:
            _figures_interpret(plt, pdf, run_dir, interp)
        _last_page(plt, pdf, cfg, methods)
        pdf.infodict().update({"Title": f"PRISMT report: {name}", "Creator": f"PRISMT {__version__}"})
    return out
