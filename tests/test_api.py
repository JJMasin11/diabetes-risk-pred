from fastapi.testclient import TestClient
from unittest.mock import patch
from src.api.schemas import PredictResponse, PredictRequest
from src.api.main import app

client = TestClient(app)

def make_request(**kwargs):
    request = PredictRequest.model_construct(**{field: 0 for field in PredictRequest.model_fields})
    for key, value in kwargs.items():
        setattr(request, key, value)
    
    return request

def test_valid_payload_returns_200():
    payload = make_request(height=1, weight=1, age=18, high_chol="No").model_dump()

    mock_response = PredictResponse(diabetes_risk=1, diabetes_probability=0.72, shap_values={"bmi": 0.4})

    with patch('src.api.main.PredictPipeline.predict', return_value=mock_response):
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "diabetes_risk" in response.json()
    assert "diabetes_probability" in response.json()
    assert "shap_values" in response.json()

def test_missing_field_returns_422():
    payload = {
        "systolic_bp": 120,
        "diastolic_bp": 80
        # All other required fields omitted
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422

def test_wrong_type_returns_422():
    payload = make_request(mental_health="bad", height=1, high_chol="No").model_dump()
    response = client.post("/predict", json=payload)
    assert response.status_code == 422