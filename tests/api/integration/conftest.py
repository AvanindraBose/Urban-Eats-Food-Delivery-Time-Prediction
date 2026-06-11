import pytest
import pytest_asyncio
from collections.abc import AsyncIterator
from unittest.mock import MagicMock, patch, AsyncMock
from httpx import ASGITransport, AsyncClient

from backend.main import app
import backend.loaders.model_pipeline_loader as loader


@pytest.fixture(scope="session")
def real_model():
    """
    Simulates a real model pipeline without hitting MLflow/DagsHub.
    Replace MagicMock with joblib.load("path/to/local/model.joblib")
    if you have a local model artifact available.
    """
    model = MagicMock()
    model.predict.return_value = [30]
    return model


@pytest.fixture(autouse=True, scope="session")
def preload_model(real_model):
    """
    Injects the real (or local) model into the module-level cache
    so load_artifacts() never calls MLflow during integration tests.
    """
    loader._model_pipe = real_model
    yield
    loader._model_pipe = None


@pytest_asyncio.fixture
async def integration_client() -> AsyncIterator[AsyncClient]:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client