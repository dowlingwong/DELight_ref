#!/usr/bin/env python3
"""
Propose or apply PDF renames, optionally using arXiv titles.
- Scans the repo for *.pdf.
- Detects arXiv-like IDs in filenames.
- If --use-arxiv is set, fetches titles from arXiv and uses `title_arxiv_<id>.pdf`.
- Otherwise, slugifies filenames with optional parent-folder context.
- Default: write suggestions to docs/rename_plan.csv (old_path,new_name,reason).
- With --apply: read the CSV and rename files safely (skip missing/target-exists/collisions).
"""
from __future__ import annotations

import argparse
import csv
import functools
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "docs" / "rename_plan.csv"

ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?)")
ARXIV_UNDERSCORE_RE = re.compile(r"(\d{4})_(\d{4,5}(?:v\d+)?)")
ARXIV_API = "http://export.arxiv.org/api/query?id_list="


def find_arxiv(stem: str) -> Tuple[str, str]:
    """Return (stem_without_id, arxiv_id or ''). Handles dot or underscore-separated IDs."""
    match = ARXIV_RE.search(stem)
    if not match:
        match = ARXIV_UNDERSCORE_RE.search(stem)
        if not match:
            return stem, ""
        arxiv_id = f"{match.group(1)}.{match.group(2)}"
    else:
        arxiv_id = match.group(1)
    cleaned = (stem[: match.start()] + stem[match.end() :]).strip("_- .")
    return cleaned, arxiv_id


def slugify(text: str) -> str:
    text = text.replace(":", " ").replace("/", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_").lower()


def dedup_slug(text: str) -> str:
    """Remove consecutive duplicate tokens in an underscore-separated slug."""
    tokens = [t for t in text.split("_") if t]
    dedup = []
    for t in tokens:
        if not dedup or dedup[-1] != t:
            dedup.append(t)
    return "_".join(dedup)


@functools.lru_cache(maxsize=256)
def fetch_arxiv_title(arxiv_id: str) -> Optional[str]:
    """Fetch title from arXiv API. Returns None on error."""
    url = ARXIV_API + urllib.parse.quote(arxiv_id)
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
    except (urllib.error.URLError, TimeoutError):
        return None

    # First <title> is feed title; subsequent is the paper title.
    titles = re.findall(r"<title>(.*?)</title>", data, flags=re.DOTALL | re.IGNORECASE)
    if len(titles) < 2:
        return None
    title = titles[1].strip()
    title = re.sub(r"\s+", " ", title)
    return title or None


def suggest_for_pdf(pdf_path: Path, use_arxiv: bool = False) -> Tuple[str, str, str]:
    rel = pdf_path.relative_to(ROOT)
    stem = pdf_path.stem
    title_part, arxiv_id = find_arxiv(stem)
    context = pdf_path.parent.name

    title_slug = ""
    reason = "slugified"
    if use_arxiv and arxiv_id:
        fetched = fetch_arxiv_title(arxiv_id)
        if fetched:
            title_slug = slugify(fetched)
            reason = "arxiv title"

    if not title_slug:
        parts = []
        if title_part:
            parts.append(title_part)
        if context and (not title_part or context.lower() not in title_part.lower()):
            parts.append(context)
        base_title = "_".join(parts) if parts else stem
        title_slug = dedup_slug(slugify(base_title))
        if arxiv_id and reason != "arxiv title":
            reason = "add arxiv id"

    if arxiv_id:
        new_name = f"{title_slug}_arxiv_{arxiv_id}.pdf"
    else:
        new_name = f"{title_slug}.pdf"
    return rel.as_posix(), new_name, reason


def write_plan(output: Path, use_arxiv: bool) -> None:
    pdfs = sorted(ROOT.rglob("*.pdf"))
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Tuple[str, str, str]] = [("old_path", "new_name", "reason")]
    seen_targets: Dict[str, str] = {}

    for pdf in pdfs:
        old_path, new_name, reason = suggest_for_pdf(pdf, use_arxiv=use_arxiv)
        if new_name in seen_targets:
            reason += "; collision with " + seen_targets[new_name]
        else:
            seen_targets[new_name] = old_path
        rows.append((old_path, new_name, reason))

    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)

    print(f"Wrote suggestions to {output} (total PDFs: {len(pdfs)})")
    print("Review carefully before renaming; collisions are noted in the 'reason' column.")


def apply_plan(plan_path: Path) -> None:
    if not plan_path.exists():
        raise SystemExit(f"Plan not found: {plan_path}")

    with plan_path.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        rows = list(reader)

    if not rows:
        raise SystemExit("Plan is empty.")

    header = rows[0]
    if header[:2] != ["old_path", "new_name"]:
        raise SystemExit("Plan header must start with old_path,new_name")

    ops = []
    targets = set()
    for old_path, new_name, *_rest in rows[1:]:
        old = ROOT / old_path
        if not old.exists():
            print(f"Skip (missing): {old_path}")
            continue
        new = old.parent / new_name
        if new.exists():
            print(f"Skip (target exists): {new}")
            continue
        if new.as_posix() in targets:
            print(f"Skip (collision in plan): {new_name}")
            continue
        targets.add(new.as_posix())
        ops.append((old, new))

    if not ops:
        print("No applicable renames.")
        return

    for old, new in ops:
        new.parent.mkdir(parents=True, exist_ok=True)
        old.rename(new)
        print(f"Renamed: {old.relative_to(ROOT)} -> {new.name}")

    print(f"Applied {len(ops)} renames from {plan_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Propose or apply PDF renames.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply renames from the plan CSV instead of generating it.",
    )
    parser.add_argument(
        "--plan",
        default=OUTPUT,
        type=Path,
        help=f"Path to the plan CSV (default: {OUTPUT})",
    )
    parser.add_argument(
        "--use-arxiv",
        action="store_true",
        help="Fetch titles from arXiv and include them in suggested names.",
    )
    args = parser.parse_args()

    plan_path = args.plan if args.plan.is_absolute() else ROOT / args.plan

    if args.apply:
        apply_plan(plan_path)
    else:
        write_plan(plan_path, use_arxiv=args.use_arxiv)


if __name__ == "__main__":
    main()
