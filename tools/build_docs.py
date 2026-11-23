"""Build the supplementary PDF and the GitHub wiki from README.md and docs/.

The repository's Markdown files are the only source; the PDF and the wiki are generated
from them, so the three never disagree.

    python tools/build_docs.py pdf                 # docs/PRISMT_supplementary.pdf
    python tools/build_docs.py wiki OUT_DIR        # wiki pages, ready to commit to prismt.wiki.git

The PDF needs pandoc (>= 2.19) and XeLaTeX. Links between documents become links inside
the PDF (or between wiki pages); links to code become links to the file on GitHub, and
pictures are embedded (PDF) or loaded from GitHub (wiki).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/josueortc/PRISMTransformer"
RAW = "https://raw.githubusercontent.com/josueortc/PRISMTransformer"
PDF_PATH = "docs/PRISMT_supplementary.pdf"

# (source, wiki page, chapter title), in reading order
PAGES = [
    ("README.md", "Home", "Overview"),
    ("docs/tutorial.md", "Tutorial", "Tutorial: the PRISMT app"),
    ("docs/results-and-pitfalls.md", "Results-and-Pitfalls", "Reading the results, and the pitfalls PRISMT guards against"),
    ("docs/interpretation.md", "Interpretation", "Interpreting a classifier with PRISMt"),
    ("docs/hyperparameters.md", "Settings-Reference", "Settings reference"),
    ("docs/data-format.md", "Dataset-File-Format", "The PRISMT dataset file"),
    ("docs/cluster.md", "Cluster", "Training on a SLURM cluster"),
    ("docs/containers.md", "Containers", "Containers"),
    ("docs/dev/contracts.md", "Developer-Contracts", "Developer notes: contracts between MATLAB and Python"),
    ("docs/dev/testing.md", "Developer-Testing", "Developer notes: running the tests"),
]
PAGES = [p for p in PAGES if (REPO / p[0]).exists()]   # a distribution may carry fewer pages
PAGE_OF = {src: (page, title) for src, page, title in PAGES}
LINK = re.compile(r"(!?)\[([^\]]*)\]\(([^)\s]+)\)")
# characters missing from common text fonts
GLYPHS = {"\u2716": "\u00d7"}


def version() -> str:
    return (REPO / "src" / "prismt" / "resources" / "VERSION").read_text().strip()


def citation() -> dict:
    return json.loads((REPO / ".zenodo.json").read_text())


def chapter_id(src: str) -> str:
    return "ch-" + PAGE_OF[src][0].lower()


def split_title(text: str) -> tuple[str, str]:
    """(first H1 line, rest)."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("# "):
            return line, "\n".join(lines[:i] + lines[i + 1:])
    return "", text


def rewrite_links(src: str, text: str, mode: str, ref: str) -> str:
    """Point every relative link of ``src`` at the right place for ``mode`` (pdf or wiki)."""
    base = (REPO / src).parent

    def fix(m: re.Match) -> str:
        bang, label, target = m.groups()
        if target.startswith(("http://", "https://", "mailto:")):
            return m.group(0)
        file, _, anchor = target.partition("#")
        if not file:  # a section of the same document
            new = f"#{anchor}" if mode == "pdf" else f"{PAGE_OF[src][0]}#{anchor}"
            return f"{bang}[{label}]({new})"
        rel = (base / file).resolve().relative_to(REPO).as_posix()
        if rel in PAGE_OF:
            if mode == "pdf":
                new = f"#{anchor}" if anchor else f"#{chapter_id(rel)}"
            else:
                new = PAGE_OF[rel][0] + (f"#{anchor}" if anchor else "")
        elif bang:  # a picture
            new = (REPO / rel).as_posix() if mode == "pdf" else f"{RAW}/{ref}/{rel}"
        else:  # code, folders, other files: show them on GitHub
            kind = "tree" if (REPO / rel).is_dir() else "blob"
            new = f"{GITHUB}/{kind}/{ref}/{rel}" + (f"#{anchor}" if anchor else "")
        return f"{bang}[{label}]({new})"

    return LINK.sub(fix, text)


# ---------------------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------------------


def _first_font(candidates: list[str]) -> str | None:
    try:
        listed = subprocess.run(["fc-list", ":", "family"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return candidates[0]
    families = {f.strip() for line in listed.splitlines() for f in line.split(",")}
    return next((c for c in candidates if c in families), None)


_SEP = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _cell_width(cell: str) -> float:
    """Rough printed width: code is wider than text, links print only their label."""
    cell = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", cell)
    code = sum(len(c) for c in re.findall(r"`([^`]*)`", cell))
    return len(re.sub(r"[`*]", "", cell)) - code + 1.4 * code


def size_tables(text: str) -> str:
    """Give every table column a width that suits its contents (the separator row's dashes
    set the relative widths in pandoc), so long settings names and descriptions both fit."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if i == 0 or not _SEP.match(line):
            continue
        rows = [lines[i - 1]]
        j = i + 1
        while j < len(lines) and lines[j].lstrip().startswith("|"):
            rows.append(lines[j])
            j += 1
        n = len(_cells(line))
        widths = [4.0] * n
        for row in rows:
            for k, cell in enumerate(_cells(row)[:n]):
                widths[k] = max(widths[k], min(_cell_width(cell), 45.0))
        lines[i] = "|" + "|".join("-" * max(3, round(w)) for w in widths) + "|"
    return "\n".join(lines)


def pdf_markdown(ref: str) -> str:
    parts = []
    for src, _, title in PAGES:
        _, body = split_title((REPO / src).read_text(encoding="utf-8"))
        body = re.sub(r"^\[!\[.*$\n?", "", body, flags=re.M)  # web badges (SVG, need a network)
        body = size_tables(rewrite_links(src, body, "pdf", ref))
        parts.append(f"# {title} {{#{chapter_id(src)}}}\n\n"
                     f"*Source: [`{src}`]({GITHUB}/blob/{ref}/{src})*\n\n{body.strip()}\n")
    text = "\n\n".join(parts)
    for bad, good in GLYPHS.items():
        text = text.replace(bad, good)
    return text


def build_pdf(out: Path, ref: str, doi: str | None) -> Path:
    if not shutil.which("pandoc"):
        sys.exit("pandoc is not installed (https://pandoc.org/installing.html)")
    meta = citation()
    authors = [c["name"].split(", ", 1)[::-1] for c in meta["creators"]]
    v = version()
    today = dt.date.today().isoformat()
    doi_line = f"DOI: \\href{{https://doi.org/{doi}}}{{{doi}}}" if doi else \
        "DOI: assigned by Zenodo at release (see the repository's README)"
    title_page = rf"""
\begin{{titlepage}}
\centering
\vspace*{{3cm}}
{{\Huge\bfseries PRISMT\par}}
\vspace{{0.6cm}}
{{\Large Supplementary software manual\par}}
\vspace{{0.4cm}}
{{\large Transformers for trial-structured neural and behavioral recordings, driven from MATLAB\par}}
\vspace{{1.5cm}}
{{{', '.join(' '.join(a) for a in authors)}\par}}
\vspace{{0.4cm}}
{{Yale University School of Medicine, New Haven, CT, USA\par}}
\vfill
{{Software version {v} \quad·\quad {today}\par}}
\vspace{{0.2cm}}
{{\url{{{GITHUB}}}\par}}
\vspace{{0.2cm}}
{{{doi_line}\par}}
\vspace{{0.2cm}}
{{Licensed under the Apache License, Version 2.0\par}}
\vspace{{0.8cm}}
{{\small This manual is generated from the documentation in the repository (README.md and docs/).
Method reference: Ortega Caro, J. et al., bioRxiv (2025), doi:10.1101/2025.08.29.673096.\par}}
\end{{titlepage}}
"""
    header = r"""
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small PRISMT supplementary manual}
\fancyhead[R]{\small\nouppercase{\leftmark}}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\fancypagestyle{plain}{\fancyhf{}\fancyfoot[C]{\thepage}\renewcommand{\headrulewidth}{0pt}}
\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\small}
\setlength{\emergencystretch}{3em}
"""
    main = _first_font(["Arial", "Liberation Sans", "DejaVu Sans"])
    mono = _first_font(["Menlo", "DejaVu Sans Mono", "Liberation Mono"])
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "manual.md").write_text(pdf_markdown(ref), encoding="utf-8")
        (tmp / "header.tex").write_text(header, encoding="utf-8")
        (tmp / "title.tex").write_text(title_page, encoding="utf-8")
        cmd = ["pandoc", str(tmp / "manual.md"), "-o", str(out), "--pdf-engine=xelatex",
               "--from=markdown-yaml_metadata_block-tex_math_dollars", "--columns=60", "--toc", "--toc-depth=2",
               "--top-level-division=chapter", "--include-in-header", str(tmp / "header.tex"),
               "--include-before-body", str(tmp / "title.tex"), "--resource-path", str(REPO / "docs"),
               "-V", "documentclass=report", "-V", "geometry:margin=2.2cm", "-V", "fontsize=10pt",
               "-V", "colorlinks=true", "-V", "linkcolor=blue", "-V", "urlcolor=blue", "-V", "toccolor=black",
               "-V", f"title-meta=PRISMT supplementary manual (version {v})",
               "-V", "author-meta=" + "; ".join(" ".join(a) for a in authors), "-M", "lang=en-US"]
        if main:
            cmd += ["-V", f"mainfont={main}"]
        if mono:
            cmd += ["-V", f"monofont={mono}", "-V", "monofontoptions=Scale=0.85"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode:
            sys.exit(proc.stderr)
        for line in proc.stderr.splitlines():
            print(line, file=sys.stderr)
    return out


# ---------------------------------------------------------------------------------------
# Wiki
# ---------------------------------------------------------------------------------------


def build_wiki(out: Path, ref: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    banner = ("> This page is generated from [`{src}`]({url}) in the repository; edit it there. "
              "The whole manual is also available as a [PDF]({pdf}).\n\n")
    for src, page, _ in PAGES:
        _, body = split_title((REPO / src).read_text(encoding="utf-8"))
        body = rewrite_links(src, body, "wiki", ref)
        head = banner.format(src=src, url=f"{GITHUB}/blob/{ref}/{src}", pdf=f"{GITHUB}/blob/{ref}/{PDF_PATH}")
        path = out / f"{page}.md"
        path.write_text(head + body.strip() + "\n", encoding="utf-8")
        written.append(path)
    side = ["**PRISMT**", ""] + [f"- [{title}]({page})" for _, page, title in PAGES]
    side += ["", f"[Supplementary manual (PDF)]({GITHUB}/blob/{ref}/{PDF_PATH})",
             "", f"[Repository]({GITHUB})"]
    (out / "_Sidebar.md").write_text("\n".join(side) + "\n", encoding="utf-8")
    (out / "_Footer.md").write_text(f"PRISMT {version()} · Apache-2.0 · generated by "
                                    f"[tools/build_docs.py]({GITHUB}/blob/{ref}/tools/build_docs.py)\n",
                                    encoding="utf-8")
    return written + [out / "_Sidebar.md", out / "_Footer.md"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    p = sub.add_parser("pdf", help="build the supplementary PDF")
    p.add_argument("--out", default=str(REPO / PDF_PATH))
    p.add_argument("--ref", default="main", help="git ref that links to code point at (a tag for a release)")
    p.add_argument("--doi", default=None, help="the release DOI, printed on the title page")
    w = sub.add_parser("wiki", help="write the wiki pages to a folder")
    w.add_argument("out")
    w.add_argument("--ref", default="main")
    a = ap.parse_args(argv)
    if a.what == "pdf":
        print(build_pdf(Path(a.out), a.ref, a.doi))
    else:
        for path in build_wiki(Path(a.out), a.ref):
            print(path)


if __name__ == "__main__":
    main()
