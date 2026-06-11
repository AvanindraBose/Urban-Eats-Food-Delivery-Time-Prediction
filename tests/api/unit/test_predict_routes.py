import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from backend.main import app
from backend.core.dependencies import verify_token, get_artifact


VALID_TOKEN_PAYLOAD = {
    "payload": {"sub": "test-user"}
}

VALID_PREDICT_PAYLOAD = {
    "ID": "1",
    "Delivery_person_ID": "INDORES13DEL02",
    "Delivery_person_Age": "25",
    "Delivery_person_Ratings": "4.5",
    "Restaurant_latitude": 22.745049,
    "Restaurant_longitude": 75.892471,
    "Delivery_location_latitude": 22.765049,
    "Delivery_location_longitude": 75.912471,
    "Order_Date": "01-01-2024",
    "Time_Orderd": "10:00",
    "Time_Order_picked": "10:15",
    "Weatherconditions": "Sunny",
    "Road_traffic_density": "Low",
    "Vehicle_condition": 2,
    "Type_of_order": "Snack",
    "Type_of_vehicle": "motorcycle",
    "multiple_deliveries": "1",
    "Festival": "No",
    "City": "Metropolitian"
}


async def test_predict_success(async_client):
    fake_model = MagicMock()
    fake_model.predict.return_value = [30]

    app.dependency_overrides[verify_token] = lambda: VALID_TOKEN_PAYLOAD

    with patch(
        "backend.services.model_service.get_artifact",
        new_callable=AsyncMock,
        return_value=fake_model
    ), patch(
        "backend.services.model_service.perform_data_cleaning",
        return_value=MagicMock(empty=False)
    ):
        response = await async_client.post("/predict", json=VALID_PREDICT_PAYLOAD)

    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "ETA Prediction Successful"
    assert "time" in body
    assert "lower_limit" in body
    assert "upper_limit" in body


async def test_predict_fails_without_token(async_client):
    response = await async_client.post("/predict", json=VALID_PREDICT_PAYLOAD)

    assert response.status_code in (401, 403)


async def test_predict_internal_error(async_client):
    app.dependency_overrides[verify_token] = lambda: VALID_TOKEN_PAYLOAD

    with patch(
        "backend.services.model_service.get_artifact",
        new_callable=AsyncMock,
        side_effect=Exception("model exploded")
    ):
        response = await async_client.post("/predict", json=VALID_PREDICT_PAYLOAD)

    app.dependency_overrides.clear()

    assert response.status_code == 500
    assert response.json()["error"] == "INTERNAL_SERVER_ERROR"