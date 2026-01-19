from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from vl_photo_search.data.manifest import load_yaml_config
from vl_photo_search.indexing.embed_clip import ClipEmbedConfig, ClipEmbedder
from vl_photo_search.indexing.faiss_backend import FaissConfig, build_index, save_index


def read_manifest(manifest_path: Path):
    rows = []
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build embeddings + FAISS index from manifest.csv"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_yaml_config(Path(args.config))

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    index_cfg = cfg["index"]

    dataset_root = Path(data_cfg["dataset_root"])
    manifest_path = Path(data_cfg["manifest_path"])

    embeddings_path = Path(index_cfg["embeddings_path"])
    metadata_path = Path(index_cfg["metadata_path"])
    index_path = Path(index_cfg["index_path"])

    metric = index_cfg.get("metric", "cosine")

    rows = read_manifest(manifest_path)

    # Build absolute image paths from dataset_root + filepath in manifest
    image_paths = [dataset_root / r["filepath"] for r in rows]

    embedder = ClipEmbedder(
        ClipEmbedConfig(
            model_name=model_cfg["name"],
            device=model_cfg.get("device", "cpu"),
            batch_size=int(model_cfg.get("batch_size", 32)),
        )
    )

    batch_size = embedder.cfg.batch_size
    all_embs = []

    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i : i + batch_size]
        embs = embedder.embed_images(batch)
        all_embs.append(embs)

    embeddings = np.vstack(all_embs).astype(np.float32)

    # Save artifacts
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["row_idx", "image_id", "filepath", "caption", "split", "source"]
        )
        for idx, r in enumerate(rows):
            writer.writerow(
                [
                    idx,
                    r["image_id"],
                    r["filepath"],
                    r["caption"],
                    r["split"],
                    r["source"],
                ]
            )

    # Build + save index
    faiss_index = build_index(embeddings, FaissConfig(metric=metric))
    save_index(faiss_index, index_path)

    print("Build complete")
    print(f"Embeddings: {embeddings_path}  shape={embeddings.shape}")
    print(f"Metadata:   {metadata_path}")
    print(f"Index:      {index_path}")
    print(f"Metric:     {metric}")


if __name__ == "__main__":
    main()
