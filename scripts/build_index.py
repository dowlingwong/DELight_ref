#!/usr/bin/env python3
"""
Generate docs/index.md from catalog.yaml.
Keeps the repo structure unchanged; only reads metadata and writes the index.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - handled by fallback parser
    yaml = None

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "catalog.yaml"
OUTPUT = ROOT / "docs" / "index.md"


def _parse_value(raw: str) -> Any:
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [v.strip() for v in inner.split(",") if v.strip()]
    if raw.startswith(("'", '"')) and raw.endswith(("'", '"')):
        return raw[1:-1]
    return raw


def _fallback_parse_catalog(text: str) -> Dict[str, Any]:
    """Parse a very small subset of YAML (mappings + list of mappings)."""
    lines = [
        line.rstrip("\n")
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    meta: Dict[str, Any] = {}
    entries: List[Dict[str, Any]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("metadata:"):
            i += 1
            while i < len(lines) and lines[i].startswith("  "):
                key, val = lines[i].strip().split(":", 1)
                meta[key.strip()] = _parse_value(val)
                i += 1
            continue
        if line.startswith("entries:"):
            i += 1
            while i < len(lines):
                if not lines[i].lstrip().startswith("- "):
                    i += 1
                    continue
                entry: Dict[str, Any] = {}
                prefix = lines[i].split("- ", 1)[1].strip()
                if prefix:
                    if ":" in prefix:
                        key, val = prefix.split(":", 1)
                        entry[key.strip()] = _parse_value(val)
                i += 1
                while i < len(lines) and lines[i].startswith("    "):
                    key, val = lines[i].strip().split(":", 1)
                    entry[key.strip()] = _parse_value(val)
                    i += 1
                if entry:
                    entries.append(entry)
            continue
        i += 1

    return {"metadata": meta, "entries": entries}


def load_catalog() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not CATALOG.exists():
        raise SystemExit(f"catalog not found: {CATALOG}")
    text = CATALOG.read_text(encoding="utf-8")
    if yaml:
        data = yaml.safe_load(text) or {}
    else:
        data = _fallback_parse_catalog(text)

    meta = data.get("metadata", {})
    entries = data.get("entries", [])
    if not isinstance(entries, list):
        raise SystemExit("catalog 'entries' must be a list")
    return meta, entries


def normalize_entry(raw: Dict[str, Any]) -> Dict[str, Any]:
    if "path" not in raw:
        raise ValueError("entry missing required field 'path'")
    rel_path = Path(raw["path"])
    abs_path = ROOT / rel_path

    entry_type = raw.get("type")
    if not entry_type:
        if abs_path.is_dir():
            entry_type = "collection"
        elif rel_path.suffix.lower() == ".pdf":
            entry_type = "paper"
        else:
            entry_type = rel_path.suffix.lstrip(".") or "other"

    tags = raw.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]

    notes = raw.get("notes") or ""
    if isinstance(notes, list):
        notes = "; ".join(str(n) for n in notes)

    title = raw.get("title") or rel_path.name

    return {
        "path": rel_path.as_posix(),
        "title": title,
        "type": entry_type,
        "tags": tags,
        "notes": notes,
        "exists": abs_path.exists(),
    }


def build_markdown(meta: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    repo_name = meta.get("repo", ROOT.name)
    owner = meta.get("owner", "")
    total = len(rows)
    existing = sum(1 for r in rows if r["exists"])

    lines = [
        f"# {repo_name} Index",
        "",
        f"Generated {now} from `catalog.yaml`.",
        "",
        f"- Entries: {total}",
        f"- Existing paths: {existing}",
    ]
    if owner:
        lines.append(f"- Owner: {owner}")
    lines.extend(
        [
            "",
            "| Path | Title | Type | Tags | Notes | Exists |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )

    for row in rows:
        tags = ", ".join(row["tags"])
        exists = "yes" if row["exists"] else "no"
        notes = row["notes"].replace("\n", " ").strip()
        lines.append(
            f"| `{row['path']}` | {row['title']} | {row['type']} | {tags} | {notes} | {exists} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    meta, raw_entries = load_catalog()
    normalized = [normalize_entry(entry) for entry in raw_entries]
    normalized.sort(key=lambda r: r["path"].lower())

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    content = build_markdown(meta, normalized)
    OUTPUT.write_text(content, encoding="utf-8")
    print(f"Wrote index to {OUTPUT}")


if __name__ == "__main__":
    main()
