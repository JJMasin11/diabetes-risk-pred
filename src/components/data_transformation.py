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
            columns = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
                       'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                       'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
                       'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income',
                       'BMI_Age']

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
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def get_chol_data_transformer_object(self):
        '''
        This function is responsible for data transformation on the auxiliary cholesterol model
        '''
        try:
            columns = ['HighBP', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
                       'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                       'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
                       'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income',
                       'BMI_Age']
            
            pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )

            logging.info("Standard scaling for cholesterol preprocessor completed.")

            preprocessor = ColumnTransformer(
                [
                    ("StandardScalar", pipeline, columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed.")

            # Add BMI age interaction
            train_df["BMI_Age"] = train_df["BMI"] * train_df["Age"]
            test_df["BMI_Age"] = test_df["BMI"] * test_df["Age"]

            # Convert data from multiclass to binary
            train_df["Diabetes_Binary"] = (train_df["Diabetes_012"] != 0).astype(int)
            train_df.drop(columns=["Diabetes_012"], inplace=True)

            test_df["Diabetes_Binary"] = (test_df["Diabetes_012"] != 0).astype(int)
            test_df.drop(columns=["Diabetes_012"], inplace=True)

            # Create train and test data for auxiliary cholesterol model
            chol_train_df = train_df.drop(columns=['Diabetes_Binary'])
            chol_test_df = test_df.drop(columns=['Diabetes_Binary'])

            logging.info("Obtaining preprocessing object.")

            main_preprocessing_obj = self.get_main_data_transformer_object()
            chol_preprocessing_obj = self.get_chol_data_transformer_object()

            main_target_column_name = "Diabetes_Binary"
            chol_target_column_name = "HighChol"

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

        except Exception as e:
            raise CustomException(e, sys)