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
    trained_main_model_file_path = os.path.join("artifacts", "main_model.pkl")
    trained_chol_model_file_path = os.path.join("artifacts", "chol_model.pkl")
    threshold_file_path = os.path.join("artifacts", "threshold.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, main_train_array, main_test_array, chol_train_array, chol_test_array):
        try:
            logging.info("Splitting training and testing input data")
            x_train_main, y_train_main, x_test_main, y_test_main = (main_train_array[:, :-1], main_train_array[:, -1], main_test_array[:, :-1], main_test_array[:, -1])
            x_train_chol, y_train_chol, x_test_chol, y_test_chol = (chol_train_array[:, :-1], chol_train_array[:, -1], chol_test_array[:, :-1], chol_test_array[:, -1])

            main_model = CatBoostClassifier(learning_rate=0.01, iterations=1000, depth=6, loss_function="Logloss", auto_class_weights="Balanced", random_state=42, verbose=0)
            threshold = 0.48

            chol_model = CatBoostClassifier(learning_rate=0.05, iterations=600, depth=6, loss_function="Logloss", auto_class_weights="Balanced", random_state=42, verbose=0)

            main_model.fit(x_train_main, y_train_main)
            chol_model.fit(x_train_chol, y_train_chol)

            # Predict probabilities
            y_proba_main = main_model.predict_proba(x_test_main)[:, 1]
            y_proba_chol = chol_model.predict_proba(x_test_chol)[:, 1]

            # Make predictions
            y_pred_main = (y_proba_main >= threshold).astype(int)
            y_pred_chol = chol_model.predict(x_test_chol)

            # Calculate metrics
            main_precision = precision_score(y_test_main, y_pred_main)
            main_recall = recall_score(y_test_main, y_pred_main)
            main_roc_auc = roc_auc_score(y_test_main, y_proba_main)

            chol_precision = precision_score(y_test_chol, y_pred_chol)
            chol_recall = recall_score(y_test_chol, y_pred_chol)
            chol_roc_auc = roc_auc_score(y_test_chol, y_proba_chol)

            save_object(file_path=self.model_trainer_config.trained_main_model_file_path, obj=main_model)
            save_object(file_path=self.model_trainer_config.trained_chol_model_file_path, obj=chol_model)
            save_object(file_path=self.model_trainer_config.threshold_file_path, obj=threshold)

            return main_precision, main_recall, main_roc_auc, chol_precision, chol_recall, chol_roc_auc

        except Exception as e:
            raise CustomException(e, sys)