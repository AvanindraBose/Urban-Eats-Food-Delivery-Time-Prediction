from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from backend.main import app
from backend.core.dependencies import get_artifact


@pytest.fixture
def fake_model():
    model = MagicMock()
    model.predict.return_value = [42]
    return model


@pytest.fixture
def unit_app(fake_model):
    app.dependency_overrides[get_artifact] = lambda: fake_model
    yield app
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def async_client(unit_app) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(
        transport=ASGITransport(app=unit_app),
        base_url="http://testserver",
    ) as client:
        yield client