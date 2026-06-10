import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient


@pytest.fixture(scope="session")
def mlflow_connector() -> MlflowClient:
    dagshub_token = os.getenv("DAGSHUB_PAT")

    if not dagshub_token:
        pytest.skip("DAGSHUB_PAT not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    mlflow.set_tracking_uri(
        "https://dagshub.com/AvanindraBose/Urban-Eats-Food-Delivery-Time-Prediction.mlflow"
    )

    return MlflowClient()


@pytest.fixture(scope="session")
def staging_model_version(mlflow_connector):
    versions = mlflow_connector.get_latest_versions(
        "delivery_time_pred_model_pipe",
        stages=["Staging"]
    )

    if not versions:
        pytest.skip("No staging model found")

    return versions[0]


@pytest.fixture(scope="session")
def model_uri(staging_model_version):
    return (
        f"models:/delivery_time_pred_model_pipe/"
        f"{staging_model_version.version}"
    )


@pytest.fixture(scope="session")
def trained_model(model_uri):
    return mlflow.sklearn.load_model(model_uri)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def interim_test_data_path(repo_root: Path) -> Path:
    path = repo_root / "data" / "interim" / "test.csv"

    if not path.exists():
        pytest.skip(
            f"Interim test dataset not found at {path}"
        )

    return path


@pytest.fixture(scope="session")
def interim_test_df(interim_test_data_path: Path):
    return pd.read_csv(interim_test_data_path)