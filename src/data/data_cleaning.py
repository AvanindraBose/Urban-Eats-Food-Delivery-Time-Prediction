import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from src.utils.logger import CustomLogger, create_log_path
from datetime import datetime,timezone

log_file_path = create_log_path("Data-Cleaning-Logs")
cleaning_logger = CustomLogger(
    logger_name="Data Cleaning",
    log_filename=log_file_path,
)

cleaning_logger.save_logs(f"Data Preprocessing Pipeline Started at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}", log_level='info')

def load_data(data_path: Path) -> pd.DataFrame:
    try:
        cleaning_logger.save_logs(f"Loading raw data from: {data_path}", log_level="info")
        data = pd.read_csv(data_path)
    except Exception as exc:
        cleaning_logger.save_logs(f"Failed to load raw data from {data_path}: {exc}", log_level="exception")
        raise
    else:
        cleaning_logger.save_logs(
            f"Raw data loaded successfully from {data_path}. Shape: {data.shape}",
            log_level="info",
        )
        return data


def change_column_names(data: pd.DataFrame) -> pd.DataFrame:
    try:
        cleaning_logger.save_logs("Standardizing column names.", log_level="info")
        renamed_data = (
            data.rename(str.lower, axis=1)
            .rename(
                {
                    "delivery_person_id": "rider_id",
                    "delivery_person_age": "age",
                    "delivery_person_ratings": "ratings",
                    "delivery_location_latitude": "delivery_latitude",
                    "delivery_location_longitude": "delivery_longitude",
                    "time_orderd": "order_time",
                    "time_order_picked": "order_picked_time",
                    "weatherconditions": "weather",
                    "road_traffic_density": "traffic",
                    "city": "city_type",
                    "time_taken(min)": "time_taken",
                },
                axis=1,
            )
        )
    except Exception as exc:
        cleaning_logger.save_logs(f"Failed while standardizing column names: {exc}", log_level="exception")
        raise
    else:
        cleaning_logger.save_logs("Column names standardized successfully.", log_level="info")
        return renamed_data


def clean_lat_long(data: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    try:
        cleaning_logger.save_logs("Cleaning invalid latitude and longitude values.", log_level="info")
        cleaned_data = data.copy()
        location_columns = [
            "restaurant_latitude",
            "restaurant_longitude",
            "delivery_latitude",
            "delivery_longitude",
        ]

        for col in location_columns:
            cleaned_data.loc[cleaned_data[col] < threshold, col] = np.nan
    except Exception as exc:
        cleaning_logger.save_logs(f"Failed while cleaning latitude/longitude columns: {exc}", log_level="exception")
        raise
    else:
        cleaning_logger.save_logs("Latitude and longitude columns cleaned successfully.", log_level="info")
        return cleaned_data


def columns_to_drop(data: pd.DataFrame) -> pd.DataFrame:
    try:
        cleaning_logger.save_logs("Dropping columns not required for modeling.", log_level="info")
        columns_drop = [
            "rider_id",
            "restaurant_latitude",
            "restaurant_longitude",
            "delivery_latitude",
            "delivery_longitude",
            "order_date",
            "order_time_hour",
            "order_day",
            "city",
            "order_day_of_week",
            "order_month",
        ]
        cleaned_data = data.drop(columns=columns_drop)
    except Exception as exc:
        cleaning_logger.save_logs(f"Failed while dropping modeling-excluded columns: {exc}", log_level="exception")
        raise
    else:
        cleaning_logger.save_logs(
            f"Columns dropped successfully. Final columns count: {cleaned_data.shape[1]}",
            log_level="info",
        )
        return cleaned_data


def extract_datetime_features(ser: pd.Series) -> pd.DataFrame:
    try:
        cleaning_logger.save_logs("Extracting date-based features.", log_level="info")
        date_col = pd.to_datetime(ser, dayfirst=True)
        date_features = pd.DataFrame(
            {
                "day": date_col.dt.day,
                "month": date_col.dt.month,
                "day_of_week": date_col.dt.day_name(),
                "is_weekend": date_col.dt.day_name().isin(["Saturday", "Sunday"]).astype(int),
            }
        )
    except Exception as exc:
        cleaning_logger.save_logs(f"Failed while extracting date features: {exc}", log_level="exception")
        raise
    else:
        cleaning_logger.save_logs("Date-based features extracted successfully.", log_level="info")
        return date_features


def time_of_day(ser: pd.Series) -> pd.Series:
    try:
        cleaning_logger.save_logs("Creating order time-of-day feature.", log_level="info")
        time_bucket = pd.cut(
            ser,
            bins=[0, 6, 12, 17, 20, 24],
            right=True,
            labels=["after_midnight", "morning", "afternoon", "evening", "night"],
        )
    except Exception as exc:
        cleaning_logger.save_logs(f"Failed while creating time-of-day feature: {exc}", log_level="exception")
        raise
    else:
        cleaning_logger.save_logs("Order time-of-day feature created successfully.", log_level="info")
        return time_bucket


def calculate_haversine_distance(df: pd.DataFrame) -> pd.DataFrame:
    try:
        cleaning_logger.save_logs("Calculating haversine delivery distance.", log_level="info")
        location_columns = [
            "restaurant_latitude",
            "restaurant_longitude",
            "delivery_latitude",
            "delivery_longitude",
        ]

        lat1 = df[location_columns[0]]
        lon1 = df[location_columns[1]]
        lat2 = df[location_columns[2]]
        lon2 = df[location_columns[3]]

        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = 6371 * c
        cleaned_data = df.assign(distance=distance)
    except Exception as exc:
        cleaning_logger.save_logs(f"Failed while calculating haversine distance: {exc}", log_level="exception")
        raise
    else:
        cleaning_logger.save_logs("Haversine delivery distance calculated successfully.", log_level="info")
        return cleaned_data


def create_distance_type(data: pd.DataFrame) -> pd.DataFrame:
    try:
        cleaning_logger.save_logs("Creating distance type feature.", log_level="info")
        cleaned_data = data.assign(
            distance_type=pd.cut(
                data["distance"],
                bins=[0, 5, 10, 15, 25],
                right=False,
                labels=["short", "medium", "long", "very_long"],
            )
        )
    except Exception as exc:
        cleaning_logger.save_logs(f"Failed while creating distance type feature: {exc}", log_level="exception")
        raise
    else:
        cleaning_logger.save_logs("Distance type feature created successfully.", log_level="info")
        return cleaned_data


def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    try:
        cleaning_logger.save_logs(f"Starting core data cleaning. Input shape: {df.shape}", log_level="info")
        cleaned_data = df.copy()

        minors_data = cleaned_data.loc[cleaned_data["age"].astype("float") < 18]
        minors_idx = minors_data.index.to_list()
        six_star_data = cleaned_data.loc[cleaned_data["ratings"] == "6"]
        six_star_idx = six_star_data.index.to_list()

        cleaned_data = cleaned_data.drop(columns=["id"])
        cleaned_data = cleaned_data.drop(index=minors_idx)
        cleaned_data = cleaned_data.drop(index=six_star_idx)
        cleaning_logger.save_logs(
            f"Dropped {len(minors_idx)} minor riders and {len(six_star_idx)} invalid six-star rating records.",
            log_level="info",
        )

        cleaned_data = cleaned_data.replace("NaN ", np.nan)
        cleaned_data["city"] = cleaned_data["rider_id"].str.split("RES").str.get(0)

        cleaned_data["age"] = cleaned_data["age"].astype(float)
        cleaned_data["ratings"] = cleaned_data["ratings"].astype(float)

        cleaned_data["restaurant_latitude"] = cleaned_data["restaurant_latitude"].abs()
        cleaned_data["restaurant_longitude"] = cleaned_data["restaurant_longitude"].abs()
        cleaned_data["delivery_latitude"] = cleaned_data["delivery_latitude"].abs()
        cleaned_data["delivery_longitude"] = cleaned_data["delivery_longitude"].abs()

        cleaned_data["order_date"] = pd.to_datetime(cleaned_data["order_date"], dayfirst=True)
        date_df = extract_datetime_features(cleaned_data["order_date"])
        cleaned_data = cleaned_data.assign(
            order_day=date_df["day"],
            order_month=date_df["month"],
            order_day_of_week=date_df["day_of_week"],
            is_weekend=date_df["is_weekend"],
        )

        cleaned_data["order_time"] = pd.to_datetime(cleaned_data["order_time"], format="mixed")
        cleaned_data["order_picked_time"] = pd.to_datetime(cleaned_data["order_picked_time"], format="mixed")
        cleaned_data["order_time_hour"] = cleaned_data["order_time"].dt.hour
        cleaned_data["pickup_time_minutes"] = (
            cleaned_data["order_picked_time"] - cleaned_data["order_time"]
        ).dt.seconds / 60
        cleaned_data["order_time_of_day"] = time_of_day(cleaned_data["order_time_hour"])
        cleaned_data = cleaned_data.drop(columns=["order_time", "order_picked_time"])

        cleaned_data["weather"] = (
            cleaned_data["weather"]
            .str.replace("conditions ", "")
            .str.lower()
            .replace("nan", np.nan)
        )
        cleaned_data["traffic"] = cleaned_data["traffic"].str.rstrip().str.lower()
        cleaned_data["type_of_order"] = cleaned_data["type_of_order"].str.rstrip().str.lower()
        cleaned_data["type_of_vehicle"] = cleaned_data["type_of_vehicle"].str.rstrip().str.lower()
        cleaned_data["festival"] = cleaned_data["festival"].str.rstrip().str.lower()
        cleaned_data["city_type"] = cleaned_data["city_type"].str.rstrip().str.lower()
        cleaned_data["time_taken"] = cleaned_data["time_taken"].str.replace("(min) ", "").astype(int)
    except Exception as exc:
        cleaning_logger.save_logs(f"Core data cleaning failed: {exc}", log_level="exception")
        raise
    else:
        cleaning_logger.save_logs(
            f"Core data cleaning completed successfully. Output shape: {cleaned_data.shape}",
            log_level="info",
        )
        return cleaned_data


def perform_data_cleaning(df: pd.DataFrame, saved_file_path: str | Path) -> None:
    try:
        saved_file_path = Path(saved_file_path)
        cleaning_logger.save_logs(
            f"Running full data cleaning pipeline. Output path: {saved_file_path}",
            log_level="info",
        )
        cleaned_df = (
            df.pipe(change_column_names)
            .pipe(data_cleaning)
            .pipe(clean_lat_long)
            .pipe(calculate_haversine_distance)
            .pipe(create_distance_type)
            .pipe(columns_to_drop)
        )

        saved_file_path.parent.mkdir(exist_ok=True, parents=True)
        cleaned_df.to_csv(saved_file_path, index=False)
    except Exception as exc:
        cleaning_logger.save_logs(f"Data cleaning pipeline failed: {exc}", log_level="exception")
        raise
    else:
        cleaning_logger.save_logs(
            f"Cleaned dataset saved successfully at {saved_file_path}. Shape: {cleaned_df.shape}",
            log_level="info",
        )


def main() -> None:
    try:
        start_time = datetime.now(timezone.utc)
        cleaning_logger.save_logs(
            f"Data Cleaning Pipeline Started at {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
            log_level="info",
        )

        root_dir = Path(__file__).parent.parent.parent
        data_dir = root_dir / "data" / "raw" / "urbaneats-dataset.csv"
        cleaned_path = root_dir / "data" / "cleaned"
        cleaned_file_name = "urbaneats-cleaned-dataset.csv"

        df = load_data(data_dir)
        perform_data_cleaning(df, cleaned_path / cleaned_file_name)
    except Exception as exc:
        cleaning_logger.save_logs(f"Data Cleaning Pipeline Failed: {exc}", log_level="exception")
        raise
    else:
        end_time = datetime.now(timezone.utc)
        cleaning_logger.save_logs(
            f"Data Cleaning Pipeline Completed at {end_time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
            log_level="info",
        )


if __name__ == "__main__":
    main()
