from fastapi import APIRouter, Header
from fastapi.responses import JSONResponse
from backend.core.config import settings
from backend.core.security import generate_access_token
from backend.logging_fastapi.logger_api import auth_logger
from typing import Optional

router = APIRouter(tags=["Auth"])

@router.post("/auth/token")
async def provide_token(api_key: Optional[str] = Header(None,alias="X-API-Key")):
    try:
        if not api_key:
            auth_logger.save_logs("Missing API Key", log_level="warning")
            return JSONResponse(
            status_code=401,
            content={
            "error": "MISSING_API_KEY",
            "message": "API Key is required",
            "retry": False
            }
        )
        auth_logger.save_logs("Token request received", log_level="info")

        if api_key != settings.API_KEY:
            auth_logger.save_logs("Invalid API key provided", log_level="warning")
            return JSONResponse(
                status_code=403,
                content={
                    "error": "INVALID_API_KEY",
                    "message": "Invalid API Key",
                    "retry": False
                }
            )

        token = await generate_access_token()

    except Exception as e:
        auth_logger.save_logs(f"Error generating token: {str(e)}", log_level="error")
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "Failed to generate token",
                "retry": True
            }
        )
    else:
        auth_logger.save_logs("Token generated successfully", log_level="info")
        return JSONResponse(
            status_code=200,
            content={
                "access_token": token,
                "token_type": "bearer",
                "expires_in": int(settings.JWT_EXPIRY_MINUTES) * 60 # Minutes
            }
        )

