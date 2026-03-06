"""E2E Server Smoke Test.

Validates that the FastAPI application starts correctly and core
endpoints respond. Uses FastAPI TestClient (synchronous httpx wrapper)
so no real server process is spawned.

Usage::

    pytest tests/e2e/test_server_smoke.py -v -m e2e
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.e2e]


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI TestClient from the application factory."""
    import os
    os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))

    from fastapi.testclient import TestClient
    from src.server.app import create_app

    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_has_tools(self, client):
        resp = client.get("/api/health")
        data = resp.json()
        assert data["tools_registered"] > 0


class TestConversationsEndpoint:
    def test_list_conversations_returns_200(self, client):
        resp = client.get("/api/conversations", params={"user_id": "test_smoke"})
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestFeedbackEndpoint:
    def test_submit_feedback_up(self, client):
        resp = client.post(
            "/api/feedback",
            params={
                "user_id": "test_smoke",
                "conversation_id": "conv_smoke",
                "rating": "up",
                "comment": "smoke test",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "feedback_id" in data

    def test_submit_feedback_down(self, client):
        resp = client.post(
            "/api/feedback",
            params={
                "user_id": "test_smoke",
                "conversation_id": "conv_smoke",
                "rating": "down",
            },
        )
        assert resp.status_code == 200

    def test_submit_feedback_invalid_rating(self, client):
        resp = client.post(
            "/api/feedback",
            params={
                "user_id": "test_smoke",
                "conversation_id": "conv_smoke",
                "rating": "neutral",
            },
        )
        assert resp.status_code == 400

    def test_feedback_stats(self, client):
        resp = client.get("/api/feedback/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "up" in data
        assert "down" in data
