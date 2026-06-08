from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging

class TrainPipeline:
    def __init__(self):
        pass

    def train(self):
        logging.info("Starting the training pipeline...")

        # Data Ingestion
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()
        logging.info("Data ingestion completed.")

        # Data Transformation
        data_transformation = DataTransformation()
        main_train_arr, main_test_arr, chol_train_arr, chol_test_arr, _, _ = data_transformation.initiate_data_transformation(train_data, test_data)
        logging.info("Data transformation completed.")

        # Model Training
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_trainer(main_train_arr, main_test_arr, chol_train_arr, chol_test_arr)
        logging.info("Model training completed.")
        logging.info("Training pipeline completed successfully.")

if __name__ == "__main__":
    train_pipeline = TrainPipeline()
    train_pipeline.train()

