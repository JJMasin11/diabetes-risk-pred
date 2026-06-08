import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from sqlalchemy import create_engine
from dotenv import load_dotenv
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        load_dotenv()

        try:
            # Connect to PostgreSQL database and read the data into a DataFrame
            engine = create_engine(f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}")
            query = "SELECT * FROM health_indicators;"

            df = pd.read_sql(query, engine)
            df.drop(columns=['id'], inplace=True)  # Drop the 'id' column if it exists

            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=(df["diabetes_012"] != 0).astype(int))
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of the data is completed")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    main_train_arr, main_test_arr, chol_train_arr, chol_test_arr, _, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    main_precision, main_recall, main_roc_auc, chol_precision, chol_recall, chol_roc_auc = model_trainer.initiate_model_trainer(main_train_arr, main_test_arr, chol_train_arr, chol_test_arr)
    print("Main Model Performance:")
    print(f"Precision: {main_precision}, Recall: {main_recall}, ROC AUC: {main_roc_auc}")
    print("=" * 35)
    print("\n")

    print("Cholesterol Model Performance:")
    print(f"Precision: {chol_precision}, Recall: {chol_recall}, ROC-AUC: {chol_roc_auc}")