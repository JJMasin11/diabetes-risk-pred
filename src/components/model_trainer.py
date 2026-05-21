import os
import sys
from dataclasses import dataclass
from catboost import CatBoostClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    threshold_file_path = os.path.join("artifacts", "threshold.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            x_train, y_train, x_test, y_test = (train_array[:, :-1], train_array[:, -1], test_array[:, :-1], test_array[:, -1])

            model = CatBoostClassifier(learning_rate=0.01, iterations=1000, depth=6, loss_function="Logloss", auto_class_weights="Balanced", random_state=42, verbose=0)
            threshold = 0.48

            model.fit(x_train, y_train)

            # Predict probabilities
            y_proba = model.predict_proba(x_test)[:, 1]

            # Apply threshold to get binary predictions
            y_pred = (y_proba >= threshold).astype(int)

            # Calculate metrics
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=model)
            save_object(file_path=self.model_trainer_config.threshold_file_path, obj=threshold)

            return precision, recall, roc_auc

        except Exception as e:
            raise CustomException(e, sys)