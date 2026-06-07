import mlflow
import os
import json
import logging
from pathlib import Path
from mlflow import MlflowClient
from dotenv import load_dotenv
from src.utils.logger import CustomLogger,create_log_path

load_dotenv()

dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "AvanindraBose"
repo_name = "Urban-Eats-Food-Delivery-Time-Prediction"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
client = MlflowClient()

# Set Up Logging File Configurations
log_file_path = create_log_path("Model-Registration-Logs")
registry_logger = CustomLogger(
    logger_name="Model Registration",
    log_filename=log_file_path,
)

registry_logger.set_log_level(level = logging.INFO)

def load_model_information(file_path):
    try:
        registry_logger.save_logs("Started Loading Model Information File",log_level='info')
        with open(file_path) as f:
            run_info = json.load(f)
    except FileNotFoundError:
        registry_logger.save_logs("The file to load does not exist", log_level="error")
        raise
    except json.JSONDecodeError as e:
        registry_logger.save_logs(f"Failed to parse the JSON file: {e}", log_level='error')
        raise
    except Exception as e:
        registry_logger.save_logs(f"Unexpected error occurred while loading the experiment info: {e}", log_level='error')
        raise
    else:
        registry_logger.save_logs("Model Information File Loaded Successfully",log_level='info')    
        return run_info

def model_registration(model_id:str , model_name:str , client:MlflowClient)-> None:
    try:
        model_uri = f"models:/{model_id}"
        model_version = mlflow.register_model(model_uri, model_name)
        client.transition_model_version_stage(
            name = model_name,
            version = model_version.version,
            stage = "Staging",
            archive_existing_versions = False
        )

        client.update_model_version(
            name = model_name,
            version = model_version.version,
            description = f"Model Pipe version {model_version.version} registered and transitioned to Staging."
        )

        client.set_model_version_tag(
            name = model_name,
            version = model_version.version,
            key = "author",
            value = "Avanindra Bose"
        )

        registry_logger.save_logs(f"Model Pipe registered with ID {model_id} and transitioned to Staging", log_level='info')
    except Exception as e:
        registry_logger.save_logs(f"Failed to register the model pipe : {e}", log_level='error')
        raise

def main():
    try:
        root_path = Path(__file__).parent.parent.parent
        
        # run information file path
        run_info_path = root_path /"reports" / "run_information.json"
        
        # register the model
        run_info = load_model_information(run_info_path)
        
        # get the run id
        model_id = run_info.get("model_id")
        model_name = run_info.get("model_name")

        if model_id:
            model_registration(
                model_id=model_id,
                model_name=model_name,
                client=client,
            )
        else :
            registry_logger.save_logs("Model ID not found in the experiment info.", log_level='error')
    except Exception as e:
        registry_logger.save_logs(f"Failed to complete the model registration process: {e}", log_level='error')
        raise

if __name__ == "__main__":
    main()