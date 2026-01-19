from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as e:
    raise ImportError(
        "FAISS is not installed. Add `faiss-cpu` to requirements.txt and reinstall."
    ) from e


@dataclass(frozen=True)
class FaissConfig:
    metric: str = "cosine"


def _as_float32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    return np.ascontiguousarray(x)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = _as_float32(x)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def build_index(embeddings: np.ndarray, cfg: FaissConfig) -> "faiss.Index":
    """
    Build a FAISS index from embeddings.

    """
    x = _as_float32(embeddings)
    if x.ndim != 2:
        raise ValueError(f"embeddings must be 2D [N, D], got shape={x.shape}")

    if cfg.metric.lower() == "cosine":
        x = l2_normalize(x)
        index = faiss.IndexFlatIP(x.shape[1])
    elif cfg.metric.lower() == "l2":
        index = faiss.IndexFlatL2(x.shape[1])
    else:
        raise ValueError(f"Unknown metric: {cfg.metric} (use 'cosine' or 'l2')")

    index.add(x)
    return index


def search(
    index: "faiss.Index",
    queries: np.ndarray,
    top_k: int,
    metric: str = "cosine",
) -> Tuple[np.ndarray, np.ndarray]:

    q = _as_float32(queries)
    if q.ndim != 2:
        raise ValueError(f"queries must be 2D [Q, D], got shape={q.shape}")

    if metric.lower() == "cosine":
        q = l2_normalize(q)

    scores, idx = index.search(q, top_k)
    return scores, idx


def save_index(index: "faiss.Index", path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Path) -> "faiss.Index":
    if not path.exists():
        raise FileNotFoundError(f"Index not found: {path}")
    return faiss.read_index(str(path))
