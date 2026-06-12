import pytest
import pandas as pd
from unittest.mock import patch
from src.components.data_transformation import DataTransformation

def get_df():
    return pd.DataFrame({
        "diabetes_012": [0, 1, 2],
        "high_bp": [1, 0, 1],
        "high_chol": [0, 1, 0],
        "chol_check": [1, 1, 0],
        "bmi": [25.0, 30.0, 22.0],
        "smoker": [0, 1, 0],
        "stroke": [0, 0, 1],
        "heart_disease_or_attack": [0, 1, 0],
        "phys_activity": [1, 0, 1],
        "fruits": [1, 1, 0],
        "veggies": [1, 0, 1],
        "hvy_alcohol_consump": [0, 0, 1],
        "any_healthcare": [1, 1, 0],
        "no_doc_bc_cost": [0, 1, 0],
        "gen_hlth": [3, 4, 2],
        "ment_hlth": [5, 10, 0],
        "phys_hlth": [5, 10, 0],
        "diff_walk": [0, 1, 0],
        "sex": [1, 0, 1],
        "age": [7, 5, 9],
        "education": [4, 3, 5],
        "income": [5, 3, 6]
    })

def test_binary_label_conversion():
    train_df = get_df()

    with patch('src.components.data_transformation.save_object'), \
         patch('pandas.read_csv', side_effect=[train_df.copy(), train_df.copy()]):
        
        main_train_arr, _, _, _, _, _, = DataTransformation().initiate_data_transformation('train.csv', 'test.csv')

    labels = main_train_arr[:, -1].tolist()
    assert labels == [0.0, 1.0, 1.0]

def test_correct_shape():
    train_df = get_df()

    with patch('src.components.data_transformation.save_object'), \
         patch('pandas.read_csv', side_effect=[train_df.copy(), train_df.copy()]):
        
        main_train_arr, main_test_arr, chol_train_arr, chol_test_arr, _, _ = DataTransformation().initiate_data_transformation('train.csv', 'test.csv')

        assert main_train_arr.shape == (3, 23)
        assert chol_train_arr.shape == (3, 22)