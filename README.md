# Vision-Language Photo Search

Text-to-image retrieval over Flickr8k using CLIP embeddings and a FAISS vector index.

This project is not sentiment analysis. It is a multimodal retrieval system:
- Input: natural-language query (`"a dog running on grass"`)
- Output: ranked list of similar photos

## What The Code Does

The pipeline has 3 stages:

1. Dataset manifest generation
- Script: `scripts/build_manifest.py`
- Code: `src/vl_photo_search/data/manifest.py`
- Builds `data/processed/manifest.csv` from raw images + captions.

2. Embedding and index build
- Script: `scripts/build_index.py`
- Code: `src/vl_photo_search/indexing/embed_clip.py`, `src/vl_photo_search/indexing/faiss_backend.py`
- Computes image embeddings with CLIP and stores:
  - `artifacts/embeddings.npy`
  - `artifacts/metadata.csv`
  - `artifacts/index.faiss`

3. Online search service
- App: `src/vl_photo_search/service/app.py`
- Retriever: `src/vl_photo_search/service/retriever.py`
- FastAPI endpoint embeds query text with CLIP, searches FAISS, returns top-k matches.

## Theory

This project uses the standard dense-retrieval recipe:

1. CLIP maps images and text into the same vector space.
2. Similar concepts become nearby vectors.
3. Search is nearest-neighbor lookup:
   - Cosine similarity (implemented via L2-normalized vectors + inner product)
   - FAISS (`IndexFlatIP`) for fast top-k retrieval

Model in use:
- `openai/clip-vit-base-patch32` (configured in `configs/default.yaml`)

## Project Layout

```text
src/vl_photo_search/
  data/            # manifest + config loading
  indexing/        # CLIP embedding + FAISS backend
  service/         # FastAPI app + retriever
scripts/
  build_manifest.py
  build_index.py
configs/
  default.yaml
web/
  index.html       # simple search UI
tests/
```

## Clone + Setup

```bash
git clone <your-repo-url>
cd vl-photo-search
```

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt -e .
```

### macOS/Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt -e .
```

## Run The Project

### Option A: Use existing artifacts in repo

If `artifacts/index.faiss`, `artifacts/embeddings.npy`, and `artifacts/metadata.csv` already exist, run the API directly:

```bash
uvicorn vl_photo_search.service.app:create_app --factory --reload --host 127.0.0.1 --port 8000
```

Open:
- UI: `http://127.0.0.1:8000`
- Health: `http://127.0.0.1:8000/health`

### Option B: Rebuild manifest + index

```bash
python scripts/build_manifest.py --config configs/default.yaml
python scripts/build_index.py --config configs/default.yaml
```

Then start the API:

```bash
uvicorn vl_photo_search.service.app:create_app --factory --reload --host 127.0.0.1 --port 8000
```

## API Example

```bash
curl "http://127.0.0.1:8000/search?q=a%20dog%20running&k=5"
```

Response shape:
- `query`: original query
- `k`: requested top-k
- `results`: list of `{image_id, filepath, caption, score}`

## Development Checks

```bash
python -m pytest -q
ruff check .
black --check .
```

## Notes / Troubleshooting

- First startup can be slow because CLIP weights are loaded from Hugging Face.
- You can set `HF_TOKEN` for higher API rate limits when downloading model files.
- Config lives in `configs/default.yaml`.
- By default, model runs on CPU (`model.device: cpu`).
