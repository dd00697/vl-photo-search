from __future__ import annotations

from pathlib import Path

import numpy as np

from vl_photo_search.indexing.faiss_backend import (
    FaissConfig,
    build_index,
    load_index,
    search,
    save_index,
)


def test_build_and_search_cosine_self_match() -> None:
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(10, 32)).astype(np.float32)

    index = build_index(emb, FaissConfig(metric="cosine"))

    # Query using the first vector; should retrieve itself at rank 0
    scores, idx = search(index, emb[:1], top_k=5, metric="cosine")
    assert scores.shape == (1, 5)
    assert idx.shape == (1, 5)
    assert idx[0, 0] == 0


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(20, 16)).astype(np.float32)

    index = build_index(emb, FaissConfig(metric="cosine"))

    path = tmp_path / "index.faiss"
    save_index(index, path)
    assert path.exists()

    loaded = load_index(path)
    scores, idx = search(loaded, emb[:1], top_k=3, metric="cosine")
    assert idx[0, 0] == 0
