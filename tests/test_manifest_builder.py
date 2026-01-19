from __future__ import annotations

from pathlib import Path

import pytest

from vl_photo_search.data.manifest import (
    build_manifest_rows,
    parse_flickr8k_captions,
    validate_manifest,
    write_manifest_csv,
)


def test_parse_flickr8k_token_format(tmp_path: Path) -> None:
    captions = tmp_path / "Flickr8k.token.txt"
    captions.write_text(
        "img1.jpg#0\tA cat sits.\n"
        "img1.jpg#1\tA small cat on a sofa.\n"
        "img2.jpg#0\tA dog runs.\n",
        encoding="utf-8",
    )

    mapping = parse_flickr8k_captions(captions)
    assert "img1.jpg" in mapping
    assert len(mapping["img1.jpg"]) == 2
    assert mapping["img2.jpg"][0] == "A dog runs."


def test_build_manifest_rows_and_write_csv(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    images_dir = dataset_root / "images"
    images_dir.mkdir(parents=True)

    (images_dir / "img1.jpg").write_bytes(b"")
    (images_dir / "img2.jpg").write_bytes(b"")

    captions = dataset_root / "Flickr8k.token.txt"
    captions.write_text(
        "img1.jpg#0\tA cat sits.\n"
        "img1.jpg#1\tA small cat on a sofa.\n"
        "img2.jpg#0\tA dog runs.\n",
        encoding="utf-8",
    )

    rows = build_manifest_rows(
        dataset_root=dataset_root,
        images_dir="images",
        captions_file="Flickr8k.token.txt",
        dataset_name="flickr8k",
    )

    assert len(rows) == 2
    assert rows[0].filepath.startswith("images/")
    validate_manifest(rows, dataset_root=dataset_root)

    out_csv = tmp_path / "manifest.csv"
    write_manifest_csv(rows, out_csv)
    assert out_csv.exists()


def test_validate_manifest_fails_on_missing_image(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    images_dir = dataset_root / "images"
    images_dir.mkdir(parents=True)

    captions = dataset_root / "Flickr8k.token.txt"
    captions.write_text("missing.jpg#0\tCaption\n", encoding="utf-8")

    (images_dir / "img1.jpg").write_bytes(b"")
    rows = build_manifest_rows(
        dataset_root=dataset_root,
        images_dir="images",
        captions_file="Flickr8k.token.txt",
        dataset_name="flickr8k",
    )

    bad = rows[0].__class__(
        image_id=rows[0].image_id,
        filepath="images/does_not_exist.jpg",
        caption=rows[0].caption,
        split=rows[0].split,
        source=rows[0].source,
    )

    with pytest.raises(FileNotFoundError):
        validate_manifest([bad], dataset_root=dataset_root)
