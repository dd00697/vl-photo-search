from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import numpy as np


@dataclass(frozen=True)
class ClipEmbedConfig:
    model_name: str
    device: str = "cpu"
    batch_size: int = 32


def _to_numpy_embeddings(feats: Any) -> np.ndarray:
    """
    Normalize CLIP outputs to float32 numpy arrays across transformers versions.

    In some versions, `get_*_features` returns a tensor directly; in others it may
    return a model output object with `pooler_output`.
    """

    tensor = None

    if hasattr(feats, "detach"):
        tensor = feats
    elif hasattr(feats, "pooler_output") and hasattr(feats.pooler_output, "detach"):
        tensor = feats.pooler_output
    elif hasattr(feats, "last_hidden_state") and hasattr(
        feats.last_hidden_state, "detach"
    ):
        # Fallback to CLS token embedding when only hidden states are exposed.
        tensor = feats.last_hidden_state[:, 0, :]
    elif isinstance(feats, dict):
        if "pooler_output" in feats and hasattr(feats["pooler_output"], "detach"):
            tensor = feats["pooler_output"]
        elif "last_hidden_state" in feats and hasattr(
            feats["last_hidden_state"], "detach"
        ):
            tensor = feats["last_hidden_state"][:, 0, :]

    if tensor is None:
        raise TypeError(
            "Unsupported CLIP feature output type. Expected a tensor-like output "
            "or an object/dict containing `pooler_output`."
        )

    emb = tensor.detach().cpu().numpy()
    return np.asarray(emb, dtype=np.float32)


class ClipEmbedder:
    """
    Loads a CLIP model and produces image embeddings.
    """

    def __init__(self, cfg: ClipEmbedConfig):
        self.cfg = cfg

        import torch  # local import (optional dependency)
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.processor = CLIPProcessor.from_pretrained(cfg.model_name)
        self.model = CLIPModel.from_pretrained(cfg.model_name)
        self.model.eval()
        self.model.to(cfg.device)

    def embed_images(self, image_paths: List[Path]) -> np.ndarray:
        from PIL import Image

        if not image_paths:
            raise ValueError("embed_images received an empty image path list.")

        imgs = []
        for p in image_paths:
            with Image.open(p) as img:
                imgs.append(img.convert("RGB"))
        inputs = self.processor(images=imgs, return_tensors="pt")

        inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}

        with self.torch.no_grad():
            feats = self.model.get_image_features(**inputs)

        return _to_numpy_embeddings(feats)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("embed_texts received an empty text list.")

        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}

        with self.torch.no_grad():
            feats = self.model.get_text_features(**inputs)

        return _to_numpy_embeddings(feats)
