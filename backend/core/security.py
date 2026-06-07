import json
import hashlib
from datetime import datetime, timezone ,timedelta
from jose import JWTError , jwt , ExpiredSignatureError
from backend.core.config import settings
from backend.logging_fastapi.logger_api import auth_logger
from dotenv import load_dotenv

load_dotenv()

async def generate_access_token() -> dict:
    try:
        auth_logger.save_logs(f"Started Generating Service Token for api",log_level="info")
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(minutes=int(settings.JWT_EXPIRY_MINUTES))
        
        payload = {
            "sub" : settings.SERVICE_ID, # which service is issuing.
            "token_type" : "service",
            "iat" : now,
            "exp" : expires_at
        }

        token = jwt.encode(
            payload,
            settings.JWT_SECRET_KEY,
            settings.JWT_ALGORITHM  
        )

    except Exception as e:
        auth_logger.save_logs(f"Error creating access token: {str(e)}",log_level="error")
        raise e
    else:
        auth_logger.save_logs(f"Access token created successfully for the api",log_level="info")
        return token

async def verify_service_token(token:str) -> dict:
    try: 
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        if payload.get("token_type") != "service":
            auth_logger.save_logs(
                f"Invalid access token type: {payload.get('token_type')}",
                log_level="warning"
            )
            return {
                "payload": None,
                "error": "invalid_token_type"
            }

    except ExpiredSignatureError:
        auth_logger.save_logs(
            "Access token expired",
            log_level="warning"
        )
        
        return {
            "payload": None,
            "error": "expired"
        }

    except JWTError as e:
        auth_logger.save_logs(
            f"JWTError while verifying access token: {str(e)}",
            log_level="error"
        )
        
        return {
            "payload": None,
            "error": "invalid"
        }
    
    else:
        auth_logger.save_logs(f"Access token verified successfully for api_id: {payload.get('sub')}",log_level="info")
        return {
            "payload": payload,
            "error": None
        }

def make_cache_key(data:dict):
    data_string = json.dumps(data,sort_keys=True)
    return hashlib.sha256(data_string.encode()).hexdigest()