import numpy as np
import pandas as pd
from backend.core.dependencies import get_artifact
from backend.logging_fastapi.logger_api import prediction_logger
# from backend.core.security import make_cache_key
# from backend.cache.redis_model_cache import get_cached_prediction,set_cached_prediction
from fastapi.concurrency import run_in_threadpool
from scripts.data_clean_utils import perform_data_cleaning

def run_model_prediction(model,cleaned_df):
    
    pred = model.predict(cleaned_df)

    return pred


async def predict_time(data: pd.DataFrame) -> dict:

    prediction_logger.save_logs("Prediction pipeline started", log_level="info")

    # key = make_cache_key(data=data)

    # cached_result = await get_cached_prediction(key)

    # if cached_result :
    #     prediction_logger.save_logs(f"Retrieved cached prediction.", log_level="info")
    #     return cached_result
    
    try:
        df = data
        prediction_logger.save_logs("Input received for prediction", "info")

    except Exception as e:
        prediction_logger.save_logs(
            f"[INPUT ERROR] Failed to read input: {str(e)}", "error"
        )
        raise

    try:
        model_pipe = await get_artifact()
        prediction_logger.save_logs("Model Pipeline Loaded", log_level="info")

    except Exception as e:
        prediction_logger.save_logs(
            f"[MODEL LOAD ERROR] {str(e)}", "error"
        )
        raise
    
    try:
        # Offload CPU-bound cleaning to a threadpool to keep event loop free
        cleaned_df = await run_in_threadpool(perform_data_cleaning, df)
        prediction_logger.save_logs("Successfully Cleaned Input Data", log_level="info")
    except Exception as e:
        prediction_logger.save_logs(
            f"[DATA CLEANING ERROR] {str(e)}", "error"
        )
        raise

    try:

        if cleaned_df.empty:
            prediction_logger.save_logs("Prediction aborted: Input data was filtered out during cleaning.", "warning")
            return {"error": "Invalid input data: record filtered out by cleaning rules (e.g., age < 18 or invalid coords)."}

        prediction = await run_in_threadpool(
            run_model_prediction,
            model_pipe,
            cleaned_df
        )

        prediction_logger.save_logs(
            f"Prediction completed successfully | Output={prediction[0]}",
            "info"
        )

        result = {"prediction": prediction[0]}

        return result

    except Exception as e:
        prediction_logger.save_logs(
            f"[PREDICTION ERROR] {str(e)}", "error"
        )
        raise
