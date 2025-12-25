#!/usr/bin/env python3
"""
Draft rename suggestions for PDFs:
- Scans the repo for *.pdf.
- Detects arXiv-like IDs in filenames.
- Builds a slugged title from the filename (minus the ID) plus optional parent folder context.
- Writes suggestions to docs/rename_plan.csv (old_path,new_name,reason).
- Does not rename anything; review the plan and apply manually (e.g., with mv) once satisfied.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "docs" / "rename_plan.csv"

ARXIV_RE = re.compile(r"\b(\d{4}\.\d{5}(?:v\d+)?)\b")


def find_arxiv(stem: str) -> Tuple[str, str]:
    """Return (stem_without_id, arxiv_id or '')."""
    match = ARXIV_RE.search(stem)
    if not match:
        return stem, ""
    arxiv_id = match.group(1)
    cleaned = (stem[: match.start()] + stem[match.end() :]).strip("_- .")
    return cleaned, arxiv_id


def slugify(text: str) -> str:
    text = text.replace(":", " ").replace("/", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_").lower()


def suggest_for_pdf(pdf_path: Path) -> Tuple[str, str, str]:
    rel = pdf_path.relative_to(ROOT)
    stem = pdf_path.stem
    title_part, arxiv_id = find_arxiv(stem)
    context = pdf_path.parent.name
    parts = [p for p in [title_part, context] if p]
    base_title = "_".join(parts) if parts else stem
    slug = slugify(base_title)
    if arxiv_id:
        new_name = f"{slug}_arxiv_{arxiv_id}.pdf"
        reason = "add arxiv id"
    else:
        new_name = f"{slug}.pdf"
        reason = "slugified"
    return rel.as_posix(), new_name, reason


def main() -> None:
    pdfs = sorted(ROOT.rglob("*.pdf"))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows = [("old_path", "new_name", "reason")]
    seen_targets = {}

    for pdf in pdfs:
        old_path, new_name, reason = suggest_for_pdf(pdf)
        if new_name in seen_targets:
            reason += "; collision with " + seen_targets[new_name]
        else:
            seen_targets[new_name] = old_path
        rows.append((old_path, new_name, reason))

    with OUTPUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)

    print(f"Wrote suggestions to {OUTPUT} (total PDFs: {len(pdfs)})")
    print("Review carefully before renaming; collisions are noted in the 'reason' column.")


if __name__ == "__main__":
    main()
