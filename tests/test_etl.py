import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from etl.load_health_indicators import etl_pipeline, validate_schema, rename_columns

def valid_df():
    return pd.DataFrame([{
        "Diabetes_012": 0,
        "HighBP": 1,
        "HighChol": 0,
        "CholCheck": 1,
        "BMI": 28.0,
        "Smoker": 0,
        "Stroke": 0,
        "HeartDiseaseorAttack": 0,
        "PhysActivity": 1,
        "Fruits": 1,
        "Veggies": 1,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "GenHlth": 3,
        "MentHlth": 5,
        "PhysHlth": 5,
        "DiffWalk": 0,
        "Sex": 1,
        "Age": 7,
        "Education": 4,
        "Income": 5
    }])

def invalid_df():
    return pd.DataFrame([{
        "Diabetes_012": 3,
        "HighBP": 2,
        "HighChol": 2,
        "CholCheck": 2,
        "BMI": 28.0,
        "Smoker": 2,
        "Stroke": 2,
        "HeartDiseaseorAttack": 2,
        "PhysActivity": 2,
        "Fruits": 2,
        "Veggies": 2,
        "HvyAlcoholConsump": 2,
        "AnyHealthcare": 2,
        "NoDocbcCost": 2,
        "GenHlth": 6,
        "MentHlth": 31,
        "PhysHlth": 31,
        "DiffWalk": 2,
        "Sex": 2,
        "Age": 14,
        "Education": 7,
        "Income": 9
    }])

def test_valid_schema_passes():
    validate_schema(valid_df()) # Should not raise

def test_rename_columns():
    df = valid_df()
    renamed = rename_columns(df)
    assert "diabetes_012" in renamed.columns
    assert "Diabetes_012" not in renamed.columns
    assert "high_bp" in renamed.columns
    assert "HighBP" not in renamed.columns
    assert "high_chol" in renamed.columns
    assert "HighChol" not in renamed.columns
    assert "chol_check" in renamed.columns
    assert "CholCheck" not in renamed.columns
    assert "bmi" in renamed.columns
    assert "BMI" not in renamed.columns
    assert "smoker" in renamed.columns
    assert "Smoker" not in renamed.columns
    assert "stroke" in renamed.columns
    assert "Stroke" not in renamed.columns
    assert "heart_disease_or_attack" in renamed.columns
    assert "HeartDiseaseorAttack" not in renamed.columns
    assert "phys_activity" in renamed.columns
    assert "PhysActivity" not in renamed.columns
    assert "fruits" in renamed.columns
    assert "Fruits" not in renamed.columns
    assert "veggies" in renamed.columns
    assert "Veggies" not in renamed.columns
    assert "hvy_alcohol_consump" in renamed.columns
    assert "HvyAlcoholConsump" not in renamed.columns
    assert "any_healthcare" in renamed.columns
    assert "AnyHealthcare" not in renamed.columns
    assert "no_doc_bc_cost" in renamed.columns
    assert "NoDocbcCost" not in renamed.columns
    assert "gen_hlth" in renamed.columns
    assert "GenHlth" not in renamed.columns
    assert "ment_hlth" in renamed.columns
    assert "MentHlth" not in renamed.columns
    assert "phys_hlth" in renamed.columns
    assert "PhysHlth" not in renamed.columns
    assert "diff_walk" in renamed.columns
    assert "DiffWalk" not in renamed.columns
    assert "sex" in renamed.columns
    assert "Sex" not in renamed.columns
    assert "age" in renamed.columns
    assert "Age" not in renamed.columns
    assert "education" in renamed.columns
    assert "Education" not in renamed.columns
    assert "income" in renamed.columns
    assert "Income" not in renamed.columns

def test_etl_loads_data():
    with patch('etl.load_health_indicators.get_vault_secret', return_value='test'), \
         patch('etl.load_health_indicators.create_engine') as mock_engine, \
         patch('pandas.read_csv', return_value=valid_df()), \
         patch('pandas.DataFrame.to_sql') as mock_to_sql:
        
        mock_conn = MagicMock()
        mock_conn.execute.return_value.scalar.return_value = 0  # Table is empty
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn

        etl_pipeline()

        mock_to_sql.assert_called_once()

def test_invalid_schema_raises():
    df = invalid_df()
    with pytest.raises(Exception):
        validate_schema(df)

def test_skip_load_if_rows_exist():
    with patch('etl.load_health_indicators.get_vault_secret', return_value='test'), \
         patch('etl.load_health_indicators.create_engine') as mock_engine, \
         patch('pandas.read_csv', return_value=valid_df()), \
         patch('pandas.DataFrame.to_sql') as mock_to_sql:

        mock_conn = MagicMock()
        mock_conn.execute.return_value.scalar.return_value = 100  # Table has rows
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn

        etl_pipeline()

        mock_to_sql.assert_not_called()

def test_missing_column_raises():
    df = valid_df().drop(columns=['HighBP'])
    with pytest.raises(Exception):
        validate_schema(df)