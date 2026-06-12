import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.api.schemas import PredictRequest
from src.pipeline.predict_pipeline import PredictPipeline

def make_request(**kwargs):
    request = PredictRequest.model_construct(**{field: 0 for field in PredictRequest.model_fields})
    for key, value in kwargs.items():
        setattr(request, key, value)

    return request

def test_age_boundaries():
    pipeline = PredictPipeline()
    assert pipeline.get_age_category(make_request(age=24)) == 1
    assert pipeline.get_age_category(make_request(age=25)) == 2
    assert pipeline.get_age_category(make_request(age=29)) == 2
    assert pipeline.get_age_category(make_request(age=30)) == 3
    assert pipeline.get_age_category(make_request(age=34)) == 3
    assert pipeline.get_age_category(make_request(age=35)) == 4
    assert pipeline.get_age_category(make_request(age=39)) == 4
    assert pipeline.get_age_category(make_request(age=40)) == 5
    assert pipeline.get_age_category(make_request(age=44)) == 5
    assert pipeline.get_age_category(make_request(age=45)) == 6
    assert pipeline.get_age_category(make_request(age=49)) == 6
    assert pipeline.get_age_category(make_request(age=50)) == 7
    assert pipeline.get_age_category(make_request(age=54)) == 7
    assert pipeline.get_age_category(make_request(age=55)) == 8
    assert pipeline.get_age_category(make_request(age=59)) == 8
    assert pipeline.get_age_category(make_request(age=60)) == 9
    assert pipeline.get_age_category(make_request(age=64)) == 9
    assert pipeline.get_age_category(make_request(age=65)) == 10
    assert pipeline.get_age_category(make_request(age=69)) == 10
    assert pipeline.get_age_category(make_request(age=70)) == 11
    assert pipeline.get_age_category(make_request(age=74)) == 11
    assert pipeline.get_age_category(make_request(age=75)) == 12
    assert pipeline.get_age_category(make_request(age=79)) == 12
    assert pipeline.get_age_category(make_request(age=80)) == 13

def test_income_boundaries():
    pipeline = PredictPipeline()
    assert pipeline.get_income_category(make_request(income=9999)) == 1
    assert pipeline.get_income_category(make_request(income=10000)) == 2
    assert pipeline.get_income_category(make_request(income=14999)) == 2
    assert pipeline.get_income_category(make_request(income=15000)) == 3
    assert pipeline.get_income_category(make_request(income=19999)) == 3
    assert pipeline.get_income_category(make_request(income=20000)) == 4
    assert pipeline.get_income_category(make_request(income=24999)) == 4
    assert pipeline.get_income_category(make_request(income=25000)) == 5
    assert pipeline.get_income_category(make_request(income=34999)) == 5
    assert pipeline.get_income_category(make_request(income=35000)) == 6
    assert pipeline.get_income_category(make_request(income=49999)) == 6
    assert pipeline.get_income_category(make_request(income=50000)) == 7
    assert pipeline.get_income_category(make_request(income=74999)) == 7
    assert pipeline.get_income_category(make_request(income=75000)) == 8

def test_blood_pressure_logic():
    pipeline = PredictPipeline()
    assert pipeline.get_high_bp(make_request(systolic_bp=130, diastolic_bp=0)) == 1
    assert pipeline.get_high_bp(make_request(systolic_bp=129, diastolic_bp=0)) == 0
    assert pipeline.get_high_bp(make_request(systolic_bp=0, diastolic_bp=81)) == 1
    assert pipeline.get_high_bp(make_request(systolic_bp=0, diastolic_bp=80)) == 0

def test_unsure_calls_impute_chol():
    pipeline = PredictPipeline()
    with patch.object(pipeline, 'impute_chol', return_value=(1, 0.8)) as mock_impute:
        pipeline.preprocess(make_request(high_chol="Unsure", height=1))
        mock_impute.assert_called_once()

def test_predict_output():
    pipeline = PredictPipeline()

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

    mock_preprocessor = MagicMock()
    mock_preprocessor.transform.return_value = np.zeros((1, 22))

    with patch('src.pipeline.predict_pipeline.load_from_blob', side_effect=[mock_model, mock_preprocessor, 0.5]), \
         patch.object(pipeline, 'get_top_shap_values', return_value={"bmi": 0.4}), \
         patch.object(pipeline, 'log_to_database'):
        
        response = pipeline.predict(make_request(high_chol="No", height=1))

        assert response.diabetes_risk == 1
        assert response.diabetes_probability == 0.7

def test_top_shap_values():
    pipeline = PredictPipeline()

    mock_model = MagicMock()
    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = np.array([[0.5, 0.3, -0.2, 0.05, 0.4, -0.35, 0.001]])

    input_data = pd.DataFrame(columns=['a', 'b', 'c', 'd', 'e', 'f', 'g'])

    with patch('src.pipeline.predict_pipeline.shap.TreeExplainer', return_value=mock_explainer):
        result = pipeline.get_top_shap_values(mock_model, np.zeros((1, 7)), input_data)

    assert len(result) <= 5
    assert all(abs(v) > 0.01 for v in result.values())

def test_empty_shap():
    pipeline = PredictPipeline()

    mock_model = MagicMock()
    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = np.array([[0.001, 0.005, 0.002, 0.009]])

    input_data = pd.DataFrame(columns=['a', 'b', 'c', 'd'])

    with patch('src.pipeline.predict_pipeline.shap.TreeExplainer', return_value=mock_explainer):
        result = pipeline.get_top_shap_values(mock_model, np.zeros((1, 4)), input_data)

    assert len(result) == 0
