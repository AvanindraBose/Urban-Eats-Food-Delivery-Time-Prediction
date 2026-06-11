import pytest
from backend.core.config import settings


async def test_auth_missing_key(integration_client):
    response = await integration_client.post("/auth/token")

    assert response.status_code == 401
    assert response.json()["error"] == "MISSING_API_KEY"


async def test_auth_invalid_key(integration_client):
    response = await integration_client.post(
        "/auth/token",
        headers={"X-API-Key": "totally-wrong-key"}
    )

    assert response.status_code == 403
    assert response.json()["error"] == "INVALID_API_KEY"


async def test_auth_valid_key_returns_token(integration_client):
    response = await integration_client.post(
        "/auth/token",
        headers={"X-API-Key": settings.API_KEY}
    )

    assert response.status_code == 200
    body = response.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"
    assert "expires_in" in body
    assert isinstance(body["access_token"], str)
    assert len(body["access_token"]) > 0