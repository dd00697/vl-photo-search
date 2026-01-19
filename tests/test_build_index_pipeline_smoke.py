from __future__ import annotations

from pathlib import Path

import numpy as np

from vl_photo_search.indexing.faiss_backend import FaissConfig, build_index


def test_index_build_smoke(tmp_path: Path) -> None:
    # Fake "embeddings" like CLIP would output
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(5, 32)).astype(np.float32)

    index = build_index(embeddings, FaissConfig(metric="cosine"))
    scores, idx = index.search(embeddings[:1], 3)

    assert scores.shape == (1, 3)
    assert idx.shape == (1, 3)
