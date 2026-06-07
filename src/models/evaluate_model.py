import pandas as pd
import joblib
import logging
import mlflow
import json
import os
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from dotenv import load_dotenv
from src.utils.logger import CustomLogger,create_log_path
from datetime import datetime,timezone
from typing import Any
from sklearn.pipeline import Pipeline

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
mlflow.set_experiment("Food Delivery Time Prediction Pipeline")

TARGET = "time_taken"

log_file_path = create_log_path("Model-Evaluation-Logs")
evaluation_logger = CustomLogger(
    logger_name="Model Evaluation",
    log_filename=log_file_path,
)

evaluation_logger.set_log_level(level = logging.INFO)


def load_data(data_path: Path) -> pd.DataFrame:
    try:
        evaluation_logger.save_logs(f"Started Reading data from: {data_path}", log_level="info")
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        evaluation_logger.save_logs("The file to load does not exist", log_level="error")
        raise
    else:
        evaluation_logger.save_logs("Data Fetched Successfully", log_level="info")
        return df


def make_X_and_y(data:pd.DataFrame, target_column: str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

def load_preprocessor(path: Path) -> Any:
    try:
        evaluation_logger.save_logs(f"Started Loading the preprocessor from: {path}", log_level="info")
        preprocessor = joblib.load(path)
    except Exception as e:
        evaluation_logger.save_logs(f"Error while loading the preprocessor at path {path} due to: {e}",log_level="error")
        raise
    else:
        evaluation_logger.save_logs("Preprocessor Loaded Successfully",log_level="info")
        return preprocessor


def load_model(model_path: Path):
    try:
        evaluation_logger.save_logs(f"Started Loading the model from: {model_path}", log_level="info")
        model = joblib.load(model_path)
    except Exception as e :
        evaluation_logger.save_logs(f"Error while loading the model at path {model_path} due to: {e}",log_level="error")
        raise
    else:
        evaluation_logger.save_logs("Model Loaded Successfully",log_level="info")
        return model

def save_model_info(save_json_path,run_id, artifact_path, model_name,model_id,exp_id) -> None:
    try:
        evaluation_logger.save_logs("Started Saving the Run info",log_level="info")
        info_dict = {
            "run_id": run_id,
            "artifact_path": artifact_path,
            "model_name": model_name,
            "model_id": model_id,
            "experiment_id": exp_id
        }
        with open(save_json_path,"w") as f:
            json.dump(info_dict,f,indent=4)
    except Exception as e:
        evaluation_logger.save_logs(f"Error while saving model info at path {save_json_path} due to: {e}",log_level="error")
        raise
    else:
        evaluation_logger.save_logs("Model Info Saved Successfully",log_level="info")


def main():
    try:
        start_time = datetime.now(timezone.utc)
        evaluation_logger.save_logs(
            f"Model Training Pipeline Started at {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
            log_level="info",
        )
        # root path
        root_path = Path(__file__).parent.parent.parent
        # train data load path
        train_data_path = root_path / "data" / "processed" / "train_trans.csv"
        test_data_path = root_path / "data" / "processed" / "test_trans.csv"
        # model path
        model_path = root_path / "models" / "model.joblib"
        # preprocessor Path
        preprocessor_path = root_path / "models" / "preprocessor.joblib"

        # load preprocessor
        preprocessor = load_preprocessor(preprocessor_path)
        
        # load the training data
        train_data = load_data(train_data_path)
        # load the test data
        test_data = load_data(test_data_path)
        
        # split the train and test data
        X_train, y_train = make_X_and_y(train_data,TARGET)
        X_test, y_test = make_X_and_y(test_data,TARGET)
        
        # load the model
        model = load_model(model_path)
        
        # get the predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # calculate the train and test mae
        train_mae = mean_absolute_error(y_train,y_train_pred)
        test_mae = mean_absolute_error(y_test,y_test_pred)
        
        # calculate the r2 scores
        train_r2 = r2_score(y_train,y_train_pred)
        test_r2 = r2_score(y_test,y_test_pred)
        
        # calculate cross val scores
        cv_scores = cross_val_score(model,
                                    X_train,
                                    y_train,
                                    cv=5,
                                    scoring="neg_mean_absolute_error",
                                    n_jobs=-1)
        
        # mean cross val score
        mean_cv_score = -(cv_scores.mean())
        
        # log with mlflow
        with mlflow.start_run(run_name="Pipeline Model") as run:
            # set tags
            mlflow.set_tag("model","Food Delivery Time Regressor")

            # log parameters
            mlflow.log_params(model.get_params())

            # log metrics
            mlflow.log_metric("train_mae",train_mae)
            mlflow.log_metric("test_mae",test_mae)
            mlflow.log_metric("train_r2",train_r2)
            mlflow.log_metric("test_r2",test_r2)
            mlflow.log_metric("mean_cv_score",mean_cv_score)

            # log individual cv scores
            mlflow.log_metrics({f"CV {num}": score for num, score in enumerate(-cv_scores)})
            
            # mlflow dataset input datatype
            train_data_input = mlflow.data.from_pandas(train_data,targets=TARGET)
            test_data_input = mlflow.data.from_pandas(test_data,targets=TARGET)
            
            # log input
            mlflow.log_input(dataset=train_data_input,context="training")
            mlflow.log_input(dataset=test_data_input,context="validation")
            
            # model signature
            model_signature = mlflow.models.infer_signature(model_input=X_train.sample(20,random_state=42),
                                        model_output=model.predict(X_train.sample(20,random_state=42)))
            
            # log the final model + preprocessor
            model_pipe = Pipeline(
                steps=[
                    ("preprocessor",preprocessor),
                    ("model",model)
                ]
            )
            model_info = mlflow.sklearn.log_model(model_pipe,"delivery_time_pred_model_pipe",signature=model_signature)

            # log stacking regressor
            mlflow.log_artifact(root_path / "models" / "stacking_regressor.joblib")
            
            # log the power transformer
            mlflow.log_artifact(root_path / "models" / "power_transformer.joblib")
            
            # log the preprocessor
            mlflow.log_artifact(root_path / "models" / "preprocessor.joblib")
            
            # get the current run artifact uri
            artifact_uri = mlflow.get_artifact_uri()
        
            
        # get the run id 
        run_id = run.info.run_id
        model_name = "delivery_time_pred_model_pipe"
        exp_id = run.info.experiment_id
        model_id = model_info.model_id
        
        # save the model info
        save_json_path = root_path / "reports" / "run_information.json"
        save_model_info(save_json_path=save_json_path,
                        run_id=run_id,
                        artifact_path=artifact_uri,
                        model_name=model_name,
                        model_id=model_id,
                        exp_id=exp_id)
        evaluation_logger.save_logs("Model Training Pipeline completed successfully.", log_level="info")
    except Exception as e :
        evaluation_logger.save_logs(f"Feature Preprocessing Pipeline Failed: {e}", log_level="exception")
        raise
    else:
        end_time = datetime.now(timezone.utc)
        evaluation_logger.save_logs(
            f"Feature Preprocessing Pipeline Completed at {end_time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
            log_level="info",
        )


if __name__ == "__main__":
    main()
    