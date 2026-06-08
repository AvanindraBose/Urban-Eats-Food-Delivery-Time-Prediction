import mlflow
import os
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from backend.logging_fastapi.logger_api import prediction_logger
from backend.core.config import settings
from fastapi.concurrency import run_in_threadpool

load_dotenv()

# Facing Performance Issues hence using Async loader functions

try:
    dagshub_token = os.getenv("DAGSHUB_PAT")

    if dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    else:
        prediction_logger.save_logs("Dagshub Token not found in env file",log_level='warning')

    mlflow.set_tracking_uri(
        settings.MLFLOW_TRACKING_URI
    )
    client = MlflowClient()
except Exception as e:
    prediction_logger.save_logs(f"Error occurred while initializing MLflow: {str(e)}", log_level="error")

_model_pipe = None

# joblib.load(local_path) , client.get_latest_versions("model", stages=["Production"])[0] -> These are not awaitable functions

async def load_artifacts() -> tuple:
    global _model_pipe
    if _model_pipe is not None:
        prediction_logger.save_logs("Model Pipeline is already loaded, using cached versions.", log_level="info")
        return (_model_pipe)
    
    try:
        stage_model = client.get_latest_versions("delivery_time_pred_model_pipe", stages=["Staging"])[0]
        model_name = "delivery_time_pred_model_pipe"
        model_uri = f"models:/{model_name}/{stage_model.version}"

        _model_pipe = await run_in_threadpool(
            mlflow.pyfunc.load_model,
            model_uri
        )
        
    except Exception as e:
        prediction_logger.save_logs(f"Error occurred while loading artifacts: {str(e)}", log_level="error")
        raise e
    else:
        prediction_logger.save_logs("Model Pipeline loaded successfully.", log_level="info")
        return _model_pipe