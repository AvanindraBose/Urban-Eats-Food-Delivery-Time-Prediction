import pytest
from backend.core.config import settings


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


@pytest.fixture
async def auth_token(integration_client) -> str:
    """Gets a real JWT by hitting the real auth route."""
    response = await integration_client.post(
        "/auth/token",
        headers={"X-API-Key": settings.API_KEY}
    )
    return response.json()["access_token"]


async def test_predict_without_token(integration_client):
    response = await integration_client.post(
        "/predict",
        json=VALID_PREDICT_PAYLOAD
    )

    assert response.status_code in (401, 403)


async def test_predict_with_valid_token(integration_client, auth_token):
    response = await integration_client.post(
        "/predict",
        json=VALID_PREDICT_PAYLOAD,
        headers={"Authorization": f"Bearer {auth_token}"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "ETA Prediction Successful"
    assert isinstance(body["time"], int)
    assert body["lower_limit"] < body["time"] < body["upper_limit"]


async def test_predict_with_invalid_token(integration_client):
    response = await integration_client.post(
        "/predict",
        json=VALID_PREDICT_PAYLOAD,
        headers={"Authorization": "Bearer fake.invalid.token"}
    )

    assert response.status_code in (401, 403)