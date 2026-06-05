import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.utils.logger import CustomLogger, create_log_path


TARGET = "time_taken"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

log_file_path = create_log_path("Data-Preparation-Logs")
preparation_logger = CustomLogger(
    logger_name="Data Preparation",
    log_filename=log_file_path,
)
preparation_logger.save_logs(f"Data Preprocessing Pipeline Started at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}", log_level='info')


def load_data(data_path: Path) -> pd.DataFrame:
    try:
        preparation_logger.save_logs(f"Loading cleaned data from: {data_path}", log_level="info")
        data = pd.read_csv(data_path)
    except Exception as exc:
        preparation_logger.save_logs(f"Failed to load cleaned data from {data_path}: {exc}", log_level="exception")
        raise
    else:
        preparation_logger.save_logs(
            f"Cleaned data loaded successfully. Shape: {data.shape}",
            log_level="info",
        )
        return data


def validate_data(data: pd.DataFrame) -> pd.DataFrame:
    try:
        preparation_logger.save_logs("Validating data before train-test split.", log_level="info")
        if data.empty:
            raise ValueError("Input dataframe is empty.")

        if TARGET not in data.columns:
            raise ValueError(f"Required target column '{TARGET}' is missing.")

        validated_data = data.copy()
    except Exception as exc:
        preparation_logger.save_logs(f"Data validation failed: {exc}", log_level="exception")
        raise
    else:
        preparation_logger.save_logs("Data validation completed successfully.", log_level="info")
        return validated_data


def split_data(data: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        preparation_logger.save_logs(
            f"Splitting data with test_size={test_size}, random_state={random_state}.",
            log_level="info",
        )
        train_data, test_data = train_test_split(
            data,
            test_size=test_size,
            random_state=random_state,
        )
    except Exception as exc:
        preparation_logger.save_logs(f"Failed while splitting data: {exc}", log_level="exception")
        raise
    else:
        preparation_logger.save_logs(
            f"Data split completed. Train shape: {train_data.shape}, Test shape: {test_data.shape}",
            log_level="info",
        )
        return train_data, test_data


def read_params(file_path: Path) -> dict[str, Any]:
    try:
        preparation_logger.save_logs(f"Reading data preparation parameters from: {file_path}", log_level="info")

        if not file_path.exists():
            preparation_logger.save_logs(
                f"Parameters file not found at {file_path}. Using default parameters.",
                log_level="warning",
            )
            params = {
                "test_size": DEFAULT_TEST_SIZE,
                "random_state": DEFAULT_RANDOM_STATE,
            }
        else:
            with file_path.open("r", encoding="utf-8") as file:
                params_file = yaml.safe_load(file) or {}

            params = params_file.get("Data_Preparation", {})
            params = {
                "test_size": params.get("test_size", DEFAULT_TEST_SIZE),
                "random_state": params.get("random_state", DEFAULT_RANDOM_STATE),
            }
    except Exception as exc:
        preparation_logger.save_logs(f"Failed while reading preparation parameters: {exc}", log_level="exception")
        raise
    else:
        preparation_logger.save_logs(f"Preparation parameters resolved: {params}", log_level="info")
        return params


def save_data(data: pd.DataFrame, save_path: Path) -> None:
    try:
        preparation_logger.save_logs(f"Saving dataset to: {save_path}", log_level="info")
        save_path.parent.mkdir(exist_ok=True, parents=True)
        data.to_csv(save_path, index=False)
    except Exception as exc:
        preparation_logger.save_logs(f"Failed to save dataset at {save_path}: {exc}", log_level="exception")
        raise
    else:
        preparation_logger.save_logs(
            f"Dataset saved successfully at {save_path}. Shape: {data.shape}",
            log_level="info",
        )


def perform_data_preparation(
    data: pd.DataFrame,
    test_size: float,
    random_state: int,
    save_train_path: Path,
    save_test_path: Path,
) -> None:
    try:
        preparation_logger.save_logs("Running full data preparation pipeline.", log_level="info")
        validated_data = validate_data(data)
        train_data, test_data = split_data(
            validated_data,
            test_size=test_size,
            random_state=random_state,
        )

        save_data(data=train_data, save_path=save_train_path)
        save_data(data=test_data, save_path=save_test_path)
    except Exception as exc:
        preparation_logger.save_logs(f"Data preparation pipeline failed: {exc}", log_level="exception")
        raise
    else:
        preparation_logger.save_logs("Data preparation pipeline completed successfully.", log_level="info")


def main() -> None:
    try:
        start_time = datetime.now(timezone.utc)
        preparation_logger.save_logs(
            f"Data Preparation Pipeline Started at {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
            log_level="info",
        )

        root_path = Path(__file__).parent.parent.parent
        data_path = root_path / "data" / "cleaned" / "urbaneats-cleaned-dataset.csv"
        save_data_dir = root_path / "data" / "interim"
        params_file_path = root_path / "params.yaml"

        save_train_path = save_data_dir / "train.csv"
        save_test_path = save_data_dir / "test.csv"
        
        data = load_data(data_path)
        parameters = read_params(params_file_path)
        perform_data_preparation(
            data=data,
            test_size=float(parameters["test_size"]),
            random_state=int(parameters["random_state"]),
            save_train_path=save_train_path,
            save_test_path=save_test_path,
        )
    except Exception as exc:
        preparation_logger.save_logs(f"Data Preparation Pipeline Failed: {exc}", log_level="exception")
        raise
    else:
        end_time = datetime.now(timezone.utc)
        preparation_logger.save_logs(
            f"Data Preparation Pipeline Completed at {end_time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
            log_level="info",
        )


if __name__ == "__main__":
    main()
