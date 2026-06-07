import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME = "URBANEATS ETA PREDICTION"
    PROJECT_VERSION = "1.0.0"
    API_V1_STR = "/api/v1"
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    JWT_EXPIRY_MINUTES = os.getenv("JWT_EXPIRY_MINUTES")
    SERVICE_ID = os.getenv("SERVICE_ID")
    API_KEY= os.getenv("API_KEY")
    JWT_ALGORITHM= os.getenv("JWT_ALGORITHM")
    REDIS_URL = os.getenv("REDIS_URL")
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    MODEL_NAME= "StackingRegressor"

settings = Settings()