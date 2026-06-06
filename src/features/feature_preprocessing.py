import pandas as pd
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn import set_config
from src.utils.logger import CustomLogger, create_log_path
from datetime import datetime, timezone

# set the transformer outputs to pandas
set_config(transform_output='pandas')

# columns to preprocess in data

num_cols = ["age",
            "ratings",
            "pickup_time_minutes",
            "distance"]

nominal_cat_cols = ['weather',
                    'type_of_order',
                    'type_of_vehicle',
                    "festival",
                    "city_type",
                    "is_weekend",
                    "order_time_of_day"]

ordinal_cat_cols = ["traffic","distance_type"]

target_col = "time_taken"

# generate order for ordinal encoding

traffic_order = ["low","medium","high","jam"]

distance_type_order = ["short","medium","long","very_long"]

log_file_path = create_log_path("Feature-PreProcessing-Logs")
preprocessing_logger = CustomLogger(
    logger_name="Feature Preprocessing",
    log_filename=log_file_path,
)

def load_data(data_path: Path) -> pd.DataFrame:
    try:
        preprocessing_logger.save_logs(f"Loading data from: {data_path}", log_level="info")
        df = pd.read_csv(data_path)
    except Exception as exc:
        preprocessing_logger.save_logs(f"Failed to load data from {data_path}: {exc}", log_level="exception")
        raise
    else:
        return df


def drop_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    try:
        preprocessing_logger.save_logs(
            f"Original dataset shape: {data.shape}. Checking for missing values.", 
            log_level="info"
        )
        df_dropped = data.dropna()
        preprocessing_logger.save_logs(
            f"Dataset shape after dropping NaNs: {df_dropped.shape}", 
            log_level="info"
        )
        
        if df_dropped.isna().sum().sum() > 0:
            raise ValueError("The dataframe still contains missing values after dropna().")
    except Exception as exc:
        preprocessing_logger.save_logs(f"Error during missing value removal: {exc}", log_level="exception")
        raise
    else:
        return df_dropped


def save_transformer(transformer, save_dir: Path, transformer_name: str):
    try:
        save_location = save_dir / transformer_name
        preprocessing_logger.save_logs(f"Saving transformer to: {save_location}", log_level="info")
        save_dir.mkdir(exist_ok=True, parents=True)
        joblib.dump(value=transformer, filename=save_location)
    except Exception as exc:
        preprocessing_logger.save_logs(f"Failed to save transformer: {exc}", log_level="exception")
        raise

def train_preprocessor(preprocessor, data: pd.DataFrame):
    try:
        preprocessing_logger.save_logs("Fitting preprocessor on training data.", log_level="info")
        preprocessor.fit(data)
    except Exception as exc:
        preprocessing_logger.save_logs(f"Preprocessing fit failed: {exc}", log_level="exception")
        raise
    else:
        return preprocessor

def perform_transformations(preprocessor, data: pd.DataFrame):
    try:
        preprocessing_logger.save_logs("Transforming data.", log_level="info")
        transformed_data = preprocessor.transform(data)
    except Exception as exc:
        preprocessing_logger.save_logs(f"Transformation failed: {exc}", log_level="exception")
        raise
    else:
        return transformed_data

def save_data(data: pd.DataFrame, save_path: Path) -> None:
    try:
        preprocessing_logger.save_logs(f"Saving processed data to: {save_path}", log_level="info")
        save_path.parent.mkdir(exist_ok=True, parents=True)
        data.to_csv(save_path, index=False)
    except Exception as exc:
        preprocessing_logger.save_logs(f"Failed to save data at {save_path}: {exc}", log_level="exception")
        raise
    

def make_X_and_y(data:pd.DataFrame, target_column: str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


def join_X_and_y(X: pd.DataFrame, y: pd.Series):
    joined_df = X.join(y,how='inner')
    return joined_df


def main():
    try:
        start_time = datetime.now(timezone.utc)
        preprocessing_logger.save_logs(
            f"Feature Preprocessing Pipeline Started at {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
            log_level="info",
        )

        root_path = Path(__file__).parent.parent.parent
        train_data_path = root_path / "data" / "interim" / "train.csv"
        test_data_path = root_path / "data" / "interim" / "test.csv"
        save_data_dir = root_path / "data" / "processed"
        
        train_trans_filename = "train_trans.csv"
        test_trans_filename = "test_trans.csv"
        save_train_trans_path = save_data_dir / train_trans_filename
        save_test_trans_path = save_data_dir / test_trans_filename
        
        preprocessor = ColumnTransformer(transformers=[
                ("scale", MinMaxScaler(), num_cols),
                ("nominal_encode", OneHotEncoder(drop="first",
                                                handle_unknown="ignore",
                                                sparse_output=False), nominal_cat_cols),
                ("ordinal_encode", OrdinalEncoder(categories=[traffic_order,
                                                              distance_type_order],
                                                encoded_missing_value=-999,
                                                handle_unknown="use_encoded_value",
                                                unknown_value=-1), ordinal_cat_cols)],
                                        remainder="passthrough",
                                        n_jobs=-1,
                                        force_int_remainder_cols=False,
                                        verbose_feature_names_out=False)
        
        # load and clean missing values
        train_df = drop_missing_values(load_data(data_path=train_data_path))
        test_df = drop_missing_values(load_data(data_path=test_data_path))
        
        # split X and y
        X_train, y_train = make_X_and_y(data=train_df, target_column=target_col)
        X_test, y_test = make_X_and_y(data=test_df, target_column=target_col)
        
        # train preprocessor
        preprocessor = train_preprocessor(preprocessor=preprocessor, data=X_train)
        
        # transform
        X_train_trans = perform_transformations(preprocessor=preprocessor, data=X_train)
        X_test_trans = perform_transformations(preprocessor=preprocessor, data=X_test)
        
        # join back
        train_trans_df = join_X_and_y(X_train_trans, y_train)
        test_trans_df = join_X_and_y(X_test_trans, y_test)
        
        # save processed data
        save_data(data=train_trans_df, save_path=save_train_trans_path)
        save_data(data=test_trans_df, save_path=save_test_trans_path)
            
        # save preprocessor
        transformer_filename = "preprocessor.joblib"
        transformer_save_dir = root_path / "models"
        save_transformer(transformer=preprocessor,
                         save_dir=transformer_save_dir,
                         transformer_name=transformer_filename)
        
        preprocessing_logger.save_logs("Feature Preprocessing Pipeline completed successfully.", log_level="info")

    except Exception as exc:
        preprocessing_logger.save_logs(f"Feature Preprocessing Pipeline Failed: {exc}", log_level="exception")
        raise
    else:
        end_time = datetime.now(timezone.utc)
        preprocessing_logger.save_logs(
            f"Feature Preprocessing Pipeline Completed at {end_time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
            log_level="info",
        )
    
    
if __name__ == "__main__":
    main()