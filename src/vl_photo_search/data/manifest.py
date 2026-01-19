from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import yaml


@dataclass(frozen=True)
class ManifestRow:
    image_id: str
    filepath: str
    caption: str
    split: str
    source: str


def load_yaml_config(config_path: Path) -> dict:
    """Load YAML config from disk."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping/object at the top level.")
    return cfg


def parse_flickr8k_captions(captions_path: Path) -> Dict[str, List[str]]:
    """
    Parse Flickr8k captions into: filename -> list of captions.

    """
    if not captions_path.exists():
        raise FileNotFoundError(f"Captions file not found: {captions_path}")

    text = captions_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not text:
        return {}

    looks_like_tokens = any(
        ("\t" in line and "#" in line.split("\t", 1)[0]) for line in text[:20]
    )

    mapping: Dict[str, List[str]] = {}

    if looks_like_tokens:
        for line in text:
            if not line.strip():
                continue
            if "\t" not in line:
                continue
            left, caption = line.split("\t", 1)

            filename = left.split("#", 1)[0].strip()
            cap = caption.strip()
            if not filename:
                continue
            mapping.setdefault(filename, []).append(cap)
    else:

        reader = csv.reader(text)
        for row in reader:
            if not row:
                continue
            if len(row) < 2:
                continue
            filename = row[0].strip()
            cap = ",".join(row[1:]).strip()
            if not filename:
                continue
            mapping.setdefault(filename, []).append(cap)

    return mapping


def build_manifest_rows(
    dataset_root: Path,
    images_dir: str,
    captions_file: str,
    dataset_name: str,
    split_default: str = "train",
) -> List[ManifestRow]:
    """
    Build manifest rows for all images under dataset_root/images_dir.

    """
    images_path = dataset_root / images_dir
    if not images_path.exists():
        raise FileNotFoundError(f"Images directory not found: {images_path}")

    captions_path = dataset_root / captions_file
    captions_map = parse_flickr8k_captions(captions_path)

    exts = {".jpg", ".jpeg", ".png"}
    files = sorted(
        [p for p in images_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
    )

    rows: List[ManifestRow] = []
    for img_path in files:
        filename = img_path.name
        image_id = img_path.stem

        caps = captions_map.get(filename, [])
        caption = " || ".join(caps) if caps else ""

        rel = Path(images_dir) / filename

        filepath = rel.as_posix()

        rows.append(
            ManifestRow(
                image_id=image_id,
                filepath=filepath,
                caption=caption,
                split=split_default,
                source=dataset_name,
            )
        )

    return rows


def validate_manifest(rows: Sequence[ManifestRow], dataset_root: Path) -> None:
    """Validate invariants from the data contract."""
    if not rows:
        raise ValueError("Manifest is empty (no images found).")

    seen = set()
    missing = []

    for r in rows:
        if not r.image_id:
            raise ValueError("Found empty image_id.")
        if not r.filepath:
            raise ValueError(f"Found empty filepath for image_id={r.image_id}.")
        if r.image_id in seen:
            raise ValueError(f"Duplicate image_id found: {r.image_id}")
        seen.add(r.image_id)

        full_path = dataset_root / Path(r.filepath)
        if not full_path.exists():
            missing.append(str(full_path))

    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} image files referenced by manifest do not exist.\n"
            f"First few missing paths:\n{preview}"
        )


def write_manifest_csv(rows: Sequence[ManifestRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "filepath", "caption", "split", "source"])
        for r in rows:
            writer.writerow([r.image_id, r.filepath, r.caption, r.split, r.source])
