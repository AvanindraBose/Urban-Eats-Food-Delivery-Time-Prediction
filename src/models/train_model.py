import pandas as pd
import yaml
import joblib
import logging
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestRegressor,StackingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from pathlib import Path
from src.utils.logger import CustomLogger, create_log_path
from datetime import datetime,timezone
from typing import Any

TARGET = "time_taken"

DEFAULT_RF_PARAMS = {
  "n_estimators": 100,
  "max_depth": None,
  "min_samples_split": 2,
  "min_samples_leaf": 1,
  "max_features": 1.0,
  "random_state": 42,
  "n_jobs": -1,
}

DEFAULT_XGB_PARAMS = {
 "n_estimators": 100,
  "learning_rate": 0.3,
  "max_depth": 6,
  "min_child_weight": 1,
  "gamma": 0,
  "reg_alpha": 0,
  "reg_lambda": 1,
  "random_state": 42,
  "n_jobs": -1
}

log_file_path = create_log_path("Model-Training-Logs")
training_logger = CustomLogger(
    logger_name="Model Training",
    log_filename=log_file_path,
)

training_logger.set_log_level(level = logging.INFO)

def load_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
    
    except FileNotFoundError:
        training_logger.error("The file to load does not exist")
    
    else:
        training_logger.save_logs("Data Fetched Successfully" , log_level="info")
    
        return df

def read_params(file_path: Path) -> dict[str, Any]:
    try:
        training_logger.save_logs(f"Reading data preparation parameters from: {file_path}", log_level="info")

        if not file_path.exists():
            training_logger.save_logs(
                f"Parameters file not found at {file_path}. Using default parameters.",
                log_level="warning",
            )
            params = {
                "RFRegressor": DEFAULT_RF_PARAMS,
                "XGBRegressor": DEFAULT_XGB_PARAMS,
            }
        else:
            with file_path.open("r", encoding="utf-8") as file:
                params_file = yaml.safe_load(file) or {}

            params = params_file.get("Train", {})
        
    except Exception as exc:
        training_logger.save_logs(f"Failed while reading preparation parameters at path {file_path} due to : {exc}", log_level="exception")
    else:
        training_logger.save_logs(f"Preparation parameters resolved: {params}", log_level="info")
        return params


def save_model(model, save_dir: Path, model_name: str) -> None:
    try: 
        training_logger.save_logs("Started Saving the Model",log_level="info")
    # form the save location
        save_location = save_dir / model_name
    except Exception as e :
        training_logger.save_logs(f"Error while saving the model at path {save_dir} due to: {e}",log_level='error')
        raise
    else:
        joblib.dump(value=model,filename=save_location)
        training_logger.save_logs("Model Saved Successfully",log_level="info")
    
    
def save_transformer(transformer, save_dir: Path, transformer_name: str):
    try:
    # form the save location
        save_location = save_dir / transformer_name
    except Exception as e :
        training_logger.save_logs(f"Error while saving the transformer at path {save_dir} due to: {e}",log_level='error')
        raise
    else:
        joblib.dump(transformer, save_location)
        training_logger.save_logs("Transformer Saved Successfully",log_level="info")
    
    
def train_model(model, X_train: pd.DataFrame, y_train):
    try: 
    # fit on the data
        training_logger.save_logs("Started Training the Model",log_level="info")
        model.fit(X_train,y_train)
    except Exception as e :
        training_logger.save_logs(f"Error while training the model: {e}",log_level='error')
        raise
    else:
        training_logger.save_logs("Model Trained Successfully",log_level="info")
        return model


def make_X_and_y(data:pd.DataFrame, target_column: str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

def main():
    try:
        start_time = datetime.now(timezone.utc)
        training_logger.save_logs(
            f"Model Training Pipeline Started at {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
            log_level="info",
        )
        # root path
        root_path = Path(__file__).parent.parent.parent
        # train data load path
        data_path = root_path / "data" / "processed" / "train_trans.csv"
        # parameters file
        params_file_path = root_path / "params.yaml"
    
        # load the training data
        training_data = load_data(data_path)
    
        # split the data into X and y
        X_train, y_train = make_X_and_y(training_data, TARGET)
    
        # model parameters
        model_params = read_params(params_file_path)
    
        # rf_params
        rf_params = model_params['RFRegressor']
    
        # build random forest model
        rf = RandomForestRegressor(**rf_params)
    
        # xgb params
        xgb_params = model_params["XGBRegressor"]
        xgb = XGBRegressor(**xgb_params)
    
        # meta model
        lr = LinearRegression()
    
        # power transformer
        power_transform = PowerTransformer()
    
        # form the stacking regressor
        stacking_reg = StackingRegressor(estimators=[("rf_model",rf),
                                                 ("xgb_model",xgb)],
                                     final_estimator=lr,
                                     cv=5,n_jobs=-1)
    
        # make the model wrapper
        model = TransformedTargetRegressor(regressor=stacking_reg,
                                       transformer=power_transform)
    
        # fit the model on training data
        model = train_model(model,X_train,y_train)
    
        # model name
        model_filename = "model.joblib"
        # directory to save model
        model_save_dir = root_path / "models"
        model_save_dir.mkdir(exist_ok=True)
    
        # extract the model from wrapper
        stacking_model = model.regressor_
        transformer = model.transformer_

    # save the model
        save_model(model=model,
            save_dir=model_save_dir,
            model_name=model_filename)
    
        # save the stacking model
        stacking_filename = "stacking_regressor.joblib"
        save_model(model=stacking_model,
            save_dir=model_save_dir,
            model_name=stacking_filename)
    
        # save the transformer
        transformer_filename = "power_transformer.joblib"
        transformer_save_dir = model_save_dir
        save_transformer(transformer, transformer_save_dir, transformer_filename)

        training_logger.save_logs("Model Training Pipeline completed successfully.", log_level="info")
    
    except Exception as exc:
        training_logger.save_logs(f"Feature Preprocessing Pipeline Failed: {exc}", log_level="exception")
        raise
    else:
        end_time = datetime.now(timezone.utc)
        training_logger.save_logs(
            f"Feature Preprocessing Pipeline Completed at {end_time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
            log_level="info",
        )

if __name__ == "__main__":
    main()