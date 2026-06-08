import pandas as pd
import numpy as np
from fastapi import Depends,HTTPException
from fastapi.security import HTTPBearer,HTTPAuthorizationCredentials
from backend.loaders.model_pipeline_loader import load_artifacts
from backend.core.security import verify_service_token
from backend.logging_fastapi.logger_api import prediction_logger


security = HTTPBearer()

async def get_artifact():
    return await load_artifacts()

async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:

    token = credentials.credentials

    status = await verify_service_token(token)

    if status.get("error") == "expired":

        prediction_logger.save_logs(
            "Token Expired",
            log_level="warning"
        )

        raise HTTPException(
            status_code=401,
            detail={
                "error": "TOKEN_EXPIRED",
                "message": "Token Expired, Please authenticate again",
                "retry": False
            }
        )

    if status.get("error") == "invalid":

        prediction_logger.save_logs(
            "Invalid Token",
            log_level="warning"
        )

        raise HTTPException(
            status_code=401,
            detail={
                "error": "INVALID_TOKEN",
                "message": "Invalid Token, Please authenticate again",
                "retry": False
            }
        )

    if status.get("error") == "invalid_token_type":

        prediction_logger.save_logs(
            "Invalid Token Type",
            log_level="warning"
        )

        raise HTTPException(
            status_code=403,
            detail={
                "error": "INVALID_TOKEN_TYPE",
                "message": "Service Token Required",
                "retry": False
            }
        )

    return status



        