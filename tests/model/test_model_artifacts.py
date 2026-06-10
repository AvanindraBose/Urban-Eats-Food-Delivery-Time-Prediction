import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def test_model_artifact_exists(trained_model):
    """
    Verify that the model artifact was successfully loaded from MLflow.
    """
    assert trained_model is not None, (
        "Failed to load model from MLflow."
    )


def test_model_exposes_predict_interface(trained_model):
    """
    Verify that the loaded object behaves like a model.
    """
    assert hasattr(trained_model, "predict"), (
        "Loaded model does not implement predict()."
    )


def test_prediction_count_matches_input_rows(
    trained_model,
    interim_test_df,
):
    """
    Ensure the number of predictions equals
    the number of input samples.
    """
    X = interim_test_df.drop(columns=["time_taken"])

    predictions = trained_model.predict(X)

    assert len(predictions) == len(X), (
        "Prediction count does not match input row count."
    )


def test_predictions_are_numeric(
    trained_model,
    interim_test_df,
):
    """
    Ensure predictions are numeric.
    """
    X = interim_test_df.drop(columns=["time_taken"])

    predictions = trained_model.predict(X)

    assert np.issubdtype(predictions.dtype, np.number), (
        "Predictions are not numeric."
    )


def test_predictions_are_finite(
    trained_model,
    interim_test_df,
):
    """
    Ensure predictions do not contain NaN
    or infinite values.
    """
    X = interim_test_df.drop(columns=["time_taken"])

    predictions = trained_model.predict(X)

    assert np.isfinite(predictions).all(), (
        "Predictions contain NaN or infinite values."
    )


def test_model_performance_thresholds(
    trained_model,
    interim_test_df,
):
    """
    Promotion gate:
    The staging model should satisfy the
    predefined business thresholds.
    """

    X = interim_test_df.drop(columns=["time_taken"])
    y = interim_test_df["time_taken"]

    predictions = trained_model.predict(X)

    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)

    MAX_ACCEPTABLE_MAE = 3.10
    MIN_ACCEPTABLE_R2 = 0.80

    assert mae <= MAX_ACCEPTABLE_MAE, (
        f"MAE={mae:.3f} exceeded threshold "
        f"({MAX_ACCEPTABLE_MAE})."
    )

    assert r2 >= MIN_ACCEPTABLE_R2, (
        f"R²={r2:.3f} below threshold "
        f"({MIN_ACCEPTABLE_R2})."
    )