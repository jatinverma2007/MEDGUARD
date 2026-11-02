"""API tests using FastAPI's TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from medguard.api import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_analyze_text(client: TestClient) -> None:
    response = client.post(
        "/analyze/text",
        json={"text": "Tab Aspirin 75mg and Tab Warfarin 5mg", "language": "en"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "drugs" in body
    assert "findings" in body
    assert "explanations" in body


def test_analyze_text_rejects_empty(client: TestClient) -> None:
    response = client.post("/analyze/text", json={"text": "   "})
    assert response.status_code == 422


def test_analyze_image_rejects_non_image(client: TestClient) -> None:
    response = client.post(
        "/analyze/image",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 415


def test_analyze_image_rejects_empty_file(client: TestClient) -> None:
    response = client.post(
        "/analyze/image",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 422


def test_openapi_schema_available(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/analyze/text" in response.json()["paths"]
