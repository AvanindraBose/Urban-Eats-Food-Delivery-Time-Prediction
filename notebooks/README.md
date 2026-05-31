# Notebook Summary: UrbanEats Delivery Time Prediction

This README summarizes the key observations, questions, and decisions documented in the `Data-Cleaning.ipynb` and `EDA.ipynb` notebooks so far. It is intended as a quick revision guide for interviews and project walkthroughs.

## 1. Data-Cleaning Notebook

### Initial Dataset Observations

- Several columns had incorrect data types:
  - `age` should be numeric/integer.
  - `ratings` should be float.
  - Date and time columns should be converted to date/time types.
  - `multiple_deliveries` and the target column should be numeric.
  - `vehicle_condition` appears integer encoded but behaves like a categorical feature.
- Initial missing-value checks using `isna()` and `isnull()` showed no missing values, but manual inspection revealed missing values stored as strings such as `"NaN "`.
- The `weather` column had values like `conditions NaN`, which required separate handling.
- Some latitude and longitude values were `0.000`, pointing to the equator and clearly invalid for Indian delivery locations.
- Some rider ages were recorded as `15`, which raised a data-quality and domain-validity concern.
- The first four characters of `rider_id` contain city information, making it useful for feature extraction.
- The original `city` column had values like `Metropolitan`, `Urban`, and `Semi-Urban`, so it was better treated as `city_type`.

### Missing-Value Analysis

- Missing values were hidden as string values, which is why default null checks initially failed.
- After converting these string-based missing values, the dataset had `8515` missing values.
- Missingness was not random:
  - Rider-related columns were missing together, suggesting missing rider profile data.
  - `order_time` missingness was related to rider data, possibly because of logging or network issues.
  - Weather and road traffic missingness were highly related, meaning if one was missing, the other was likely missing too.
  - Road traffic missingness may be connected to rider-side data capture.
- Around `9%` of rows had at least one missing value before additional location cleaning.

### Cleaning Decisions

- Column names were standardized.
- No duplicate rows were found.
- `id` was unique for every row but not useful for modeling, so it can be dropped.
- `rider_id` had `1320` unique riders and was kept initially because it contains city information and may help with imputation.

### Age and Rider Anomalies

- The `age` column needed conversion to integer.
- Rows with rider age `15` showed multiple issues:
  - Ratings were all `1`.
  - Vehicle condition was very poor.
  - Weather and traffic details were missing.
  - Negative latitude and longitude values were present.
  - Age `15` is below the permissible driving age.
- Because these rows had multiple serious anomalies, removing them was considered more reasonable than trying to fix them.

### Ratings

- A rating of `6` was identified as a likely data error because the expected rating scale is up to `5`.
- Rows with minor riders had rating `1`, which also looked anomalous compared with the normal rating distribution.
- These rating anomalies needed investigation and could be removed if confirmed as invalid.

### Location Cleaning

- Valid Indian coordinates should have positive latitude and longitude values within India's approximate geographic range.
- Some max values were acceptable, but minimum values included negative and zero values.
- `4071` rows had messy latitude/longitude values before cleaning.
- Negative coordinates appeared to be scaling/sign issues, while zero values could not be fixed directly.
- The chosen strategy:
  - Take absolute values for location columns.
  - Treat coordinate values below `1` as missing values.
- After this process, `3640` rows still had messy latitude/longitude values, and all problematic values were below `1`.

### Date and Time Features

- `order_date` needed conversion to date format.
- Time-based feature engineering was performed using `order_time` and `order_picked_time`.
- After extracting useful time features, `order_time` and `order_picked_time` were considered safe to drop.

### Categorical Columns

- `traffic`, `type_of_order`, `type_of_vehicle`, and `city_type` had trailing spaces, so string trimming was required.
- `type_of_order` was a balanced class.
- `type_of_vehicle` was imbalanced, with motorcycles being the most common vehicle type.
- Vehicle type imbalance was marked for further investigation during EDA.

### Cleaning Outcome

- The data-cleaning phase handled hidden missing values, inconsistent data types, invalid coordinates, anomalous riders, string formatting issues, and several feature-engineering steps.
- A useful engineered feature was created from order and pickup times for future model development.

## 2. EDA Notebook

### Purpose of EDA

The EDA focused on univariate, bivariate, and multivariate analysis to support:

- Feature engineering.
- Missing-value imputation.
- Understanding drivers of delivery time.
- Identifying features likely to improve model performance.

### Missing Data After Cleaning

- The initial percentage of rows with missing data was `9.2%`.
- After transforming invalid `0` latitude/longitude values to `NaN`, the missing-row percentage increased to `16.35%`.

### Target Variable: Delivery Time

- The target column is not fully continuous; it behaves more like a discrete time measurement.
- The distribution is bimodal:
  - One peak around `17-18` minutes.
  - Another peak around `26-27` minutes.
- Some deliveries around `50` minutes looked extreme but were not treated as outliers.
- These extreme delivery times were explained by:
  - High or jammed traffic.
  - Longer travel distances.
- Decision: do not remove target outliers because they represent realistic rare events.

### Rider ID

- `rider_id` was kept during EDA because it may help impute rider age and ratings.
- It was not directly useful as a modeling feature without transformation.

### Age

- Rider age is a discrete numeric column.
- Age did not show a meaningful impact on delivery time.

### Ratings

- Higher-rated riders appeared to receive more orders.
- This suggests higher ratings may create more work and income opportunities.
- Vehicle condition affected ratings:
  - Worse vehicle condition generally aligned with lower ratings.
  - Some poor-condition vehicle categories had missing ratings, suggesting customers may skip rating instead of giving very low ratings.
- Vehicle type did not strongly affect ratings.
- Festivals did not show convincing evidence of lowering rider ratings.

### Location Features

- Delivery latitude and longitude correspond clearly to city-level location patterns.
- Location data is useful for understanding city-based delivery behavior and distance.

### Order Date and Festival Effects

- Delivery time varied during festivals.
- Festival deliveries generally took longer and had a shorter range of delivery times, suggesting consistently delayed deliveries during festivals.
- Festivals affected traffic.
- Combining weekend and festival flags did not add much beyond the festival effect; festivals mainly increased delivery time due to traffic congestion.

### Order Time

- Evening orders took longer mainly because of traffic conditions.
- Delivery time was highly dependent on the interaction between time of day and traffic.
- Top ordering hours were analyzed to understand customer demand patterns.

### Pickup Time

- Pickup time did not show a significant effect on delivery time.
- It was considered safe to drop as a direct feature.

### Traffic

- Traffic was one of the most important features.
- Statistical testing indicated traffic depends on `city_type`.
- Traffic affected delivery times.
- Vehicle condition appeared related to longer delivery times in some cases, but this required careful interpretation:
  - Good-condition vehicles may appear to take longer because they are preferred during festivals.
  - The longer time may be caused by festival traffic, not by good vehicle condition itself.

### Multiple Deliveries

- Multiple deliveries affected delivery time.
- The notebook also investigated whether multiple deliveries were associated with longer distances.

### Weather

- Weather affected delivery time.
- Weather also affected traffic.
- Traffic combined with other features acted as a strong discriminatory feature for delivery-time prediction.
- Weather and traffic interaction should be useful for modeling.

### Vehicle Condition and Vehicle Type

- Vehicle condition was analyzed against delivery time using statistical tests.
- Vehicle type was also analyzed for delivery-time impact.
- The relationship between vehicle type and vehicle condition was checked.
- Motorcycles were the dominant vehicle type, so vehicle-type imbalance should be considered during modeling.

### Type of Order

- Order type was analyzed against delivery time, pickup time, ratings, weekends, and festivals.
- This helped check whether certain order categories behave differently across operational conditions.

### City and City Type

- Individual city did not show a significant relationship with delivery time.
- `city_type` was more meaningful than raw city for traffic-related behavior.
- City type was analyzed against:
  - Delivery time.
  - Rider ratings.
  - Vehicle type.
  - Traffic.

### Distance

- Distance was analyzed as a key numeric feature.
- Longer distances were associated with longer delivery times.
- Distance was also analyzed with:
  - Vehicle type.
  - Festivals.
  - City type.
  - Traffic.
- A new distance-based feature was created and checked against delivery time.
- Distance plus traffic is likely an important feature interaction.

## 3. Modeling Implications

- Important likely predictors:
  - Distance.
  - Traffic density.
  - Weather conditions.
  - Festival flag.
  - Time of day.
  - Multiple deliveries.
  - City type.
  - Vehicle condition.
- Features that may be dropped or transformed:
  - `id`: drop because it is only a unique identifier.
  - Raw `order_time` and `order_picked_time`: drop after extracting useful time features.
  - `pickup_time`: likely drop if it does not add predictive value.
  - `rider_id`: avoid using raw ID directly; use it only for feature extraction or imputation.
- Important interactions to consider:
  - Traffic x distance.
  - Traffic x weather.
  - Traffic x time of day.
  - Festival x traffic.
  - City type x traffic.
- Outlier strategy:
  - Do not remove long delivery times automatically, because they are explainable by distance and traffic.
- Missing-value strategy:
  - Treat string `"NaN "` values as missing.
  - Treat invalid coordinates such as `0` or values below `1` as missing.
  - Investigate grouped missingness rather than assuming data is missing completely at random.

## 4. Talking Points

- I did not rely only on `isnull()` checks because missing values were stored as strings. This helped uncover hidden data-quality problems.
- I treated latitude and longitude carefully because invalid coordinates can distort distance-based features.
- I avoided removing target extremes because domain analysis showed that long deliveries were realistic under jammed traffic and longer distances.
- I separated raw city from `city_type` because the original values represented area category, not actual city names.
- I used EDA to decide which features were useful, which features needed engineering, and which features could be dropped.
- I checked feature interactions instead of looking only at one-variable relationships, especially for traffic, weather, festival, and distance.
- I used statistical tests to support conclusions about relationships such as traffic vs city type and delivery time vs traffic.
- I treated data cleaning as a modeling decision, not just formatting, because cleaning choices directly affect missingness, feature quality, and model reliability.
