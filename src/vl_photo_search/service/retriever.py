from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


from vl_photo_search.data.manifest import load_yaml_config
from vl_photo_search.indexing.embed_clip import ClipEmbedConfig, ClipEmbedder
from vl_photo_search.indexing.faiss_backend import load_index, search


@dataclass(frozen=True)
class SearchResult:
    image_id: str
    filepath: str
    caption: str
    score: float


class Retriever:
    """
    Loads artifacts once (FAISS index + metadata) and serves text queries.

    ML engineering principle: keep online path fast by loading offline-built artifacts.
    """

    def __init__(self, config_path: Path):
        self.config_path = config_path.resolve()
        self.cfg = load_yaml_config(self.config_path)
        config_dir = self.config_path.parent
        project_root = config_dir.parent

        def _resolve_cfg_path(path_str: str) -> Path:
            p = Path(path_str)
            if p.is_absolute():
                return p

            for base in (project_root, config_dir, Path.cwd()):
                candidate = base / p
                if candidate.exists():
                    return candidate

            # For output paths that may not exist yet, default to project-root relative.
            return project_root / p

        data_cfg = self.cfg["data"]
        model_cfg = self.cfg["model"]
        index_cfg = self.cfg["index"]

        self.dataset_root = _resolve_cfg_path(data_cfg["dataset_root"])
        self.metric = index_cfg.get("metric", "cosine")

        self.index_path = _resolve_cfg_path(index_cfg["index_path"])
        self.metadata_path = _resolve_cfg_path(index_cfg["metadata_path"])

        # Load index + metadata (fast path after startup)
        self.index = load_index(self.index_path)
        self.metadata = self._load_metadata(self.metadata_path)

        # Load CLIP for text embeddings (online)
        self.embedder = ClipEmbedder(
            ClipEmbedConfig(
                model_name=model_cfg["name"],
                device=model_cfg.get("device", "cpu"),
                batch_size=int(model_cfg.get("batch_size", 32)),
            )
        )

    def _load_metadata(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Metadata not found: {path}")
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        return rows

    def search_text(self, query: str, top_k: int = 10) -> list[SearchResult]:
        q_emb = self.embedder.embed_texts([query])  # shape (1, D)

        scores, idx = search(self.index, q_emb, top_k=top_k, metric=self.metric)

        results: list[SearchResult] = []
        for rank in range(idx.shape[1]):
            j = int(idx[0, rank])
            if j < 0:
                continue
            m = self.metadata[j]
            results.append(
                SearchResult(
                    image_id=m["image_id"],
                    filepath=m["filepath"],
                    caption=m.get("caption", ""),
                    score=float(scores[0, rank]),
                )
            )
        return results
