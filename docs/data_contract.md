# Data Contract: Vision-Language Photo Search

## Purpose
This project uses a dataset manifest as the single source of truth for indexing and evaluation.
All pipelines must read from the manifest.

---

## Directory layout
- data/raw/               Raw downloaded dataset
- data/processed/         Generated artifacts used by pipelines
- artifacts/              Model/index artifacts

---

## Manifest file
**Path:** `data/processed/manifest.csv`

### Schema (required columns)
| column     | type   | description |
|-----------|--------|-------------|
| image_id  | string | Unique identifier for the image |
| filepath  | string | Relative path to the image file from `dataset_root` |
| caption   | string | Optional caption (may be empty) |
| split     | string | One of: `train`, `val`, `test`
| source    | string | Dataset name (e.g., `flickr8k`) |

### Invariants (must always be true)
1) `image_id` is unique.
2) `filepath` points to an existing image file under `dataset_root`.
3) No missing values in `image_id` or `filepath`.
4) `split` (if used) must be one of `train/val/test`.
5) The manifest is deterministic: same raw dataset → same manifest.

---

## Dataset root
`dataset_root` is configured in `configs/default.yaml` and is not hardcoded in code.
