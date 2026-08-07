import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    main_preprocessor_obj_file_path = os.path.join("artifacts", "main_preprocessor.pkl")
    chol_preprocessor_obj_file_path = os.path.join("artifacts", "chol_preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_main_data_transformer_object(self):
        '''
        This function is responsible for data transformation on the main diabetes prediction model
        '''
        try:
            columns = ['high_bp', 'high_chol', 'chol_check', 'bmi', 'smoker', 'stroke',
                       'heart_disease_or_attack', 'phys_activity', 'fruits', 'veggies',
                       'hvy_alcohol_consump', 'any_healthcare', 'no_doc_bc_cost', 'gen_hlth',
                       'ment_hlth', 'phys_hlth', 'diff_walk', 'sex', 'age', 'education', 'income',
                       'bmi_age']

            pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )

            logging.info("Standard scaling for main preprocessor completed.")

            preprocessor = ColumnTransformer(
                [
                    ("StandardScaler", pipeline, columns)
                ]
            )

            return preprocessor

        except (ValueError, TypeError) as e:
            raise CustomException(e, sys)

    def get_chol_data_transformer_object(self):
        '''
        This function is responsible for data transformation on the auxiliary cholesterol model
        '''
        try:
            columns = ['high_bp', 'chol_check', 'bmi', 'smoker', 'stroke',
                       'heart_disease_or_attack', 'phys_activity', 'fruits', 'veggies',
                       'hvy_alcohol_consump', 'any_healthcare', 'no_doc_bc_cost', 'gen_hlth',
                       'ment_hlth', 'phys_hlth', 'diff_walk', 'sex', 'age', 'education', 'income',
                       'bmi_age']

            pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )

            logging.info("Standard scaling for cholesterol preprocessor completed.")

            preprocessor = ColumnTransformer(
                [
                    ("StandardScaler", pipeline, columns)
                ]
            )

            return preprocessor

        except (ValueError, TypeError) as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed.")

            # Add BMI age interaction
            train_df["bmi_age"] = train_df["bmi"] * train_df["age"]
            test_df["bmi_age"] = test_df["bmi"] * test_df["age"]

            # Convert data from multiclass to binary
            train_df["diabetes_binary"] = (train_df["diabetes_012"] != 0).astype(int)
            train_df.drop(columns=["diabetes_012"], inplace=True)

            test_df["diabetes_binary"] = (test_df["diabetes_012"] != 0).astype(int)
            test_df.drop(columns=["diabetes_012"], inplace=True)

            # Create train and test data for auxiliary cholesterol model
            chol_train_df = train_df.drop(columns=['diabetes_binary'])
            chol_test_df = test_df.drop(columns=['diabetes_binary'])

            logging.info("Obtaining preprocessing object.")

            main_preprocessing_obj = self.get_main_data_transformer_object()
            chol_preprocessing_obj = self.get_chol_data_transformer_object()

            main_target_column_name = "diabetes_binary"
            chol_target_column_name = "high_chol"

            main_input_feature_train_df = train_df.drop(columns=[main_target_column_name], axis=1)
            main_target_feature_train_df = train_df[main_target_column_name]

            main_input_feature_test_df = test_df.drop(columns=[main_target_column_name], axis=1)
            main_target_feature_test_df = test_df[main_target_column_name]

            chol_input_feature_train_df = chol_train_df.drop(columns=[chol_target_column_name], axis=1)
            chol_target_feature_train_df = chol_train_df[chol_target_column_name]

            chol_input_feature_test_df = chol_test_df.drop(columns=[chol_target_column_name], axis=1)
            chol_target_feature_test_df = chol_test_df[chol_target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes.")

            main_input_feature_train_arr = main_preprocessing_obj.fit_transform(main_input_feature_train_df)
            main_input_feature_test_arr = main_preprocessing_obj.transform(main_input_feature_test_df)

            chol_input_feature_train_arr = chol_preprocessing_obj.fit_transform(chol_input_feature_train_df)
            chol_input_feature_test_arr = chol_preprocessing_obj.transform(chol_input_feature_test_df)

            main_train_arr = np.c_[main_input_feature_train_arr, np.array(main_target_feature_train_df)]
            main_test_arr = np.c_[main_input_feature_test_arr, np.array(main_target_feature_test_df)]

            chol_train_arr = np.c_[chol_input_feature_train_arr, np.array(chol_target_feature_train_df)]
            chol_test_arr = np.c_[chol_input_feature_test_arr, np.array(chol_target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            save_object(file_path=self.data_transformation_config.main_preprocessor_obj_file_path, obj=main_preprocessing_obj)
            save_object(file_path=self.data_transformation_config.chol_preprocessor_obj_file_path, obj=chol_preprocessing_obj)

            return main_train_arr, main_test_arr, chol_train_arr, chol_test_arr, self.data_transformation_config.main_preprocessor_obj_file_path, self.data_transformation_config.chol_preprocessor_obj_file_path

        except (OSError, pd.errors.ParserError, KeyError, ValueError) as e:
            raise CustomException(e, sys)