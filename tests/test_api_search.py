from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient

from vl_photo_search.service.app import create_app_with_retriever


@dataclass(frozen=True)
class DummyResult:
    image_id: str
    filepath: str
    caption: str
    score: float


class DummyRetriever:
    def search_text(self, query: str, top_k: int = 10):
        return [
            DummyResult(
                image_id="img1", filepath="images/img1.jpg", caption="cap", score=0.99
            ),
            DummyResult(
                image_id="img2", filepath="images/img2.jpg", caption="cap2", score=0.50
            ),
        ][:top_k]


def test_search_endpoint_returns_results():
    app = create_app_with_retriever(DummyRetriever())
    client = TestClient(app)

    r = client.get("/search", params={"q": "a dog", "k": 2})
    assert r.status_code == 200
    data = r.json()
    assert data["query"] == "a dog"
    assert len(data["results"]) == 2
    assert data["results"][0]["image_id"] == "img1"
