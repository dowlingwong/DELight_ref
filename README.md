# DELight Reference Library

Reference repo for papers, slides, and project notes. Files stay in their current folders; we add lightweight metadata so you can find things quickly.

## Layout (current)
- `CNN:DeepClean/`, `GWAK/`, `PCA/`, `SWE_MLE/`, `TIDMID/`, `normalize_flow/`, `trigger/`: topic-specific collections of PDFs.
- `collaboration/`: collaboration materials (intro, theory, trigger, calibration, L1/L2, meeting slides, theses, etc.).
- `computation_infrastructure/`: computation and infrastructure references.
- `external/`: external references (GGI, xenon, LEE, SNOLAB trigger, less relevant).
- `DPG_abstract.pdf`: abstract at repo root.

## Conventions
- Keep existing structure; add new items inside the closest matching folder.
- Filenames: prefer lowercase with underscores/hyphens; include year or arXiv ID when easy (e.g., `2024_trigger_slides.pdf`, `2503.02112.pdf`).
- Metadata lives in `catalog.yaml`; add an entry when you add a new file.
- Auto-index: run `python3 scripts/build_index.py` to regenerate `docs/index.md`.

## Catalog
- Schema lives in `catalog.yaml` under `entries:`; each entry has at minimum `path` and optional `title`, `type`, `tags`, `notes`.
- Use `type` values like `paper`, `slides`, `thesis`, `notes`, or `collection` (for folders).
- Tags are free-form (`[trigger, calibration]`).

## Index generation
1. Update `catalog.yaml` with any new files.
2. Run `python3 scripts/build_index.py` (creates/updates `docs/index.md`).
3. Open `docs/index.md` for a browsable table of everything in the catalog.

## Renaming workflow (title + arXiv)
- Propose names (fetch arXiv titles when IDs exist): `python3 scripts/propose_renames.py --use-arxiv`
- Review `docs/rename_plan.csv`.
- Apply safely: `python3 scripts/propose_renames.py --apply` (skips missing/target-exists/collisions).
- Rebuild index: `python3 scripts/build_index.py`.
- If folders change, rerun the propose/apply + build_index steps so names and index stay in sync.
