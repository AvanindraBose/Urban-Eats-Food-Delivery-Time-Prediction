import pytest
from unittest.mock import patch, AsyncMock
from backend.core.config import settings


async def test_missing_api_key_returns_401(async_client):
    response = await async_client.post("/auth/token")

    assert response.status_code == 401
    assert response.json()["error"] == "MISSING_API_KEY"


async def test_invalid_api_key_returns_403(async_client):
    response = await async_client.post(
        "/auth/token",
        headers={"X-API-Key": "wrong-key"}
    )

    assert response.status_code == 403
    assert response.json()["error"] == "INVALID_API_KEY"


async def test_valid_api_key_returns_token(async_client):
    with patch(
        "backend.core.security.generate_access_token",
        new_callable=AsyncMock,
        return_value="mocked.jwt.token"
    ):
        response = await async_client.post(
            "/auth/token",
            headers={"X-API-Key": settings.API_KEY}
        )

    assert response.status_code == 200
    body = response.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"
    assert "expires_in" in body