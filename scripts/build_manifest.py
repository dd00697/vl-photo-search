from __future__ import annotations

import argparse
from pathlib import Path

from vl_photo_search.data.manifest import (
    build_manifest_rows,
    load_yaml_config,
    validate_manifest,
    write_manifest_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build dataset manifest.csv for indexing."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config (default: configs/default.yaml)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_yaml_config(cfg_path)
    config_dir = cfg_path.parent
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

    data_cfg = cfg.get("data", {})
    dataset_name = data_cfg.get("dataset_name", "dataset")
    dataset_root = _resolve_cfg_path(data_cfg["dataset_root"])
    images_dir = data_cfg["images_dir"]
    captions_file = data_cfg["captions_file"]
    manifest_path = _resolve_cfg_path(data_cfg["manifest_path"])

    rows = build_manifest_rows(
        dataset_root=dataset_root,
        images_dir=images_dir,
        captions_file=captions_file,
        dataset_name=dataset_name,
        split_default="train",
    )

    validate_manifest(rows, dataset_root=dataset_root)
    write_manifest_csv(rows, output_path=manifest_path)

    print(f"Wrote manifest: {manifest_path}")
    print(f"images indexed: {len(rows)}")
    print(f"Dataset root: {dataset_root.resolve()}")


if __name__ == "__main__":
    main()
