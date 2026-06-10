import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from sqlalchemy import create_engine
from src.utils import get_vault_secret


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

        try:
            # Connect to PostgreSQL database and read the data into a DataFrame
            engine = create_engine(f"postgresql+psycopg2://{get_vault_secret('DB-USER')}:{get_vault_secret('DB-PASSWORD')}@{get_vault_secret('DB-HOST')}:{get_vault_secret('DB-PORT')}/{get_vault_secret('DB-NAME')}")
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