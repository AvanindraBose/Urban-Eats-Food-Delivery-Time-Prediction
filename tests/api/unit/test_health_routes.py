import pytest
from unittest.mock import MagicMock
from backend.main import app
from backend.loaders.model_pipeline_loader import load_artifacts


async def test_health_ok_when_model_loaded(async_client):
    app.dependency_overrides[load_artifacts] = lambda: MagicMock()

    response = await async_client.get("/internal/health")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_health_error_when_model_is_none(async_client):
    app.dependency_overrides[load_artifacts] = lambda: None

    response = await async_client.get("/internal/health")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "error"