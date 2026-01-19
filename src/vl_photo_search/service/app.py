from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from vl_photo_search.service.retriever import Retriever


def create_app(config_path: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="VL Photo Search", version="0.1.0")

    cfg_path = (
        Path(config_path)
        if config_path
        else Path(os.environ.get("APP_CONFIG", "configs/default.yaml"))
    )

    # Load once at startup (for real runs)
    retriever = Retriever(cfg_path)

    data_cfg = retriever.cfg["data"]
    dataset_root = Path(data_cfg["dataset_root"])
    images_dir = data_cfg["images_dir"]

    # Serve dataset images at /images/<filename>
    app.mount(
        "/images",
        StaticFiles(directory=str((dataset_root / images_dir).resolve())),
        name="images",
    )

    # Serve the simple web UI
    @app.get("/")
    def home():
        return FileResponse("web/index.html")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/search")
    def search_endpoint(
        q: str = Query(..., min_length=1),
        k: int = Query(10, ge=1, le=50),
    ):
        try:
            results = retriever.search_text(q, top_k=k)
            return {"query": q, "k": k, "results": [r.__dict__ for r in results]}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


def create_app_with_retriever(retriever) -> FastAPI:
    """Test-friendly app factory that avoids loading real artifacts/models."""
    app = FastAPI(title="VL Photo Search", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/search")
    def search_endpoint(
        q: str = Query(..., min_length=1),
        k: int = Query(10, ge=1, le=50),
    ):
        results = retriever.search_text(q, top_k=k)
        return {"query": q, "k": k, "results": [r.__dict__ for r in results]}

    return app
