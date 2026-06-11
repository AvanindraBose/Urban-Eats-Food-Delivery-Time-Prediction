import pytest


async def test_health_returns_ok(integration_client):
    """
    Real model is preloaded via session fixture.
    Health check should pass without hitting MLflow.
    """
    response = await integration_client.get("/internal/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"