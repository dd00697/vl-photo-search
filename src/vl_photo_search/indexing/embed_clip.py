from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass(frozen=True)
class ClipEmbedConfig:
    model_name: str
    device: str = "cpu"
    batch_size: int = 32


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

        imgs = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(images=imgs, return_tensors="pt")

        inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}

        with self.torch.no_grad():
            feats = self.model.get_image_features(**inputs)

        emb = feats.detach().cpu().numpy().astype(np.float32)
        return emb
