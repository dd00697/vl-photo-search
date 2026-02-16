# Data Contract: Vision-Language Photo Search

## Purpose
This project uses a dataset manifest as the single source of truth for indexing.
All pipelines should read from the manifest.

---

## Directory Layout
- `data/raw/` raw downloaded dataset
- `data/processed/` generated artifacts used by pipelines
- `artifacts/` model/index artifacts

---

## Manifest File
Path: `data/processed/manifest.csv`

### Schema (required columns)

| column | type | description |
|---|---|---|
| image_id | string | Unique identifier for the image |
| filepath | string | Relative path to the image file from `dataset_root` |
| caption | string | Optional caption (may be empty) |
| split | string | One of: `train`, `val`, `test` |
| source | string | Dataset name (for example, `flickr8k`) |

### Invariants (must always be true)
1) `image_id` is unique.
2) `filepath` points to an existing image file under `dataset_root`.
3) No missing values in `image_id` or `filepath`.
4) `split` (if used) must be one of `train`, `val`, `test`.
5) The manifest is deterministic: same raw dataset -> same manifest.

---

## Dataset Root
`dataset_root` is configured in `configs/default.yaml` and is not hardcoded in code.
