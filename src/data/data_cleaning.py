import pandas as pd
import numpy as np
from pathlib import Path

def change_column_names(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.rename(str.lower,axis=1)
        .rename({
            "delivery_person_id" : "rider_id",
            "delivery_person_age": "age",
            "delivery_person_ratings": "ratings",
            "delivery_location_latitude": "delivery_latitude",
            "delivery_location_longitude": "delivery_longitude",
            "time_orderd": "order_time",
            "time_order_picked": "order_picked_time",
            "weatherconditions": "weather",
            "road_traffic_density": "traffic",
            "city": "city_type",
            "time_taken(min)": "time_taken"},axis=1)
    )

def clean_lat_long(data: pd.DataFrame, threshold: float=1.0) -> pd.DataFrame:
    location_columns = ['restaurant_latitude',
                        'restaurant_longitude',
                        'delivery_latitude',
                        'delivery_longitude']

    for col in location_columns:
        data.loc[data[col] < threshold , col] = np.nan
    
    return data
    

def extract_datetime_features(ser: pd.Series) -> pd.DataFrame:
    date_col = pd.to_datetime(ser,dayfirst=True)

    return (
        pd.DataFrame(
            {
                "day": date_col.dt.day,
                "month": date_col.dt.month,
                "day_of_week": date_col.dt.day_name(),
                "is_weekend": date_col.dt.day_name().isin(["Saturday","Sunday"]).astype(int)
            }
        ))

def time_of_day(ser: pd.Series):

    return(
        pd.cut(ser,bins=[0,6,12,17,20,24],right=True,
               labels=["after_midnight","morning","afternoon","evening","night"])
    )

def calculate_haversine_distance(df: pd.DataFrame) -> pd.DataFrame:
    location_columns = ['restaurant_latitude',
                        'restaurant_longitude',
                        'delivery_latitude',
                        'delivery_longitude']
    
    lat1 = df[location_columns[0]]
    lon1 = df[location_columns[1]]
    lat2 = df[location_columns[2]]
    lon2 = df[location_columns[3]]

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(
        dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6371 * c

    return (
        df.assign(
            distance = distance)
    )


def create_distance_type(data: pd.DataFrame) -> pd.DataFrame:
    return(
        data
        .assign(
                distance_type = pd.cut(data["distance"],bins=[0,5,10,15,25],
                                        right=False,labels=["short","medium","long","very_long"])
    ))

def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    minors_data = df.loc[df['age'].astype('float') < 18]
    minors_idx = minors_data.index.to_list()
    six_star_data = df.loc[df['ratings'] == '6']
    six_star_idx = six_star_data.index.to_list()

    # drop id's column
    df = df.drop(columns=['id'])

    # drop minors idx and six star idx
    df = df.drop(index=minors_idx)
    df = df.drop(index=six_star_idx)

    # Replace Nan values with np.nan values
    df = df.replace("NaN ",np.nan)

    # extract a new city column from rider id
    df['city'] = df['rider_id'].str.split("RES").str.get(0)

    # Convert age column to float
    df['age'] = df['age'].astype(float)

    #  Convert ratings to float as well.
    df['ratings'] = df['ratings'].astype(float)

    # clean location columns
    df['restaurant_latitude'] = df['restaurant_latitude'].abs()
    df['restaurant_longitude'] = df['restaurant_longitude'].abs()
    df['delivery_latitude'] = df['delivery_latitude'].abs()
    df['delivery_longitude'] = df['delivery_longitude'].abs()

    # order_date to datetime and extract some useful features.
    df['order_date'] = pd.to_datetime(df['order_date'],dayfirst=True)
    date_df = extract_datetime_features(df['order_date'])
    df = df.assign (
            order_day = date_df['day'],
            order_month = date_df['month'],
            order_day_of_week = date_df['day_of_week'],
            is_weekend = date_df['is_weekend'].astype(object)
        )
    
    # Cleaning order_time and order_picked time and extracting only useful info
    df['order_time'] = pd.to_datetime(df['order_time'],format='mixed')
    df['order_picked_time'] = pd.to_datetime(df['order_picked_time'],format = 'mixed')
    df['order_time_hour'] = df['order_time'].dt.hour
    df['pickup_time_minutes'] = (df['order_picked_time'] - df['order_time']).dt.seconds/60
    df['order_time_of_day'] = time_of_day(df['order_time_hour'])
    df = df.drop(columns=['order_time','order_picked_time'])

    #  Cleaning Categorical Columns
    # Weather Column 
    df['weather'] = df['weather'].str.replace("conditions ","").str.lower().replace("nan",np.nan)

    #  Traffic Column
    df['traffic'] = df['traffic'].str.rstrip().str.lower()

    # type_of_order
    df['type_of_order'] = df['type_of_order'].str.rstrip().str.lower()

    # type of vehicle
    df['type_of_vehicle'] = df['type_of_vehicle'].str.rstrip().str.lower()

    # festival cleaning
    df['festival'] = df['festival'].str.rstrip().str.lower()

    # city_type
    df['city_type'] = df['city_type'].str.rstrip().str.lower()

    # Target Column -> Time Taken
    df['time_taken'] = df['time_taken'].str.replace("(min) ","").astype(int)

    return df


def perform_data_cleaning(df: pd.DataFrame , saved_file_path: str) -> None:
    cleaned_df = (
        df
        .pipe(change_column_names)
        .pipe(data_cleaning)
        .pipe(clean_lat_long)
        .pipe(calculate_haversine_distance)
        # .pipe(create_distance_type)
    )


    cleaned_df.to_csv(saved_file_path,index = False)

if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent.parent
    data_dir = root_dir / "data" / "raw" / "urbaneats-dataset.csv"
    cleaned_path = root_dir / "data" / "interim"
    cleaned_path.mkdir(exist_ok=True,parents=True)
    cleaned_file_name = "urbaneats-cleaned-dataset.csv"

    df  = pd.read_csv(data_dir)
    
    perform_data_cleaning(df , cleaned_path / cleaned_file_name)
