import os
import sys
from dataclasses import dataclass
from catboost import CatBoostClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import mlflow
from mlflow.models import infer_signature
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dotenv import load_dotenv

@dataclass
class ModelTrainerConfig:
    trained_main_model_file_path = os.path.join("artifacts", "main_model.pkl")
    trained_chol_model_file_path = os.path.join("artifacts", "chol_model.pkl")
    threshold_file_path = os.path.join("artifacts", "threshold.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        load_dotenv()
        if os.getenv("APP_ENV", "local") == "local":
            mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
        else:
            mlflow.set_tracking_uri(os.getenv("MLFLOW_URI"))

    def log_to_mlflow(self, experiment, model, params, metrics, x_train, artifact_path):
        mlflow.set_experiment(experiment)

        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params(params)
            mlflow.log_param("train_size", len(x_train))

            # Log metrics
            mlflow.log_metrics(metrics)

            # Set model type tag
            mlflow.set_tag("model_type", type(model).__name__)

            # Infer model signature
            threshold = params.get("threshold")
            if threshold is not None:
                proba = model.predict_proba(x_train[:5])[:, 1]
                preds = (proba >= threshold).astype(int)
            else:
                preds = model.predict(x_train[:5])

            signature = infer_signature(x_train[:5], preds)

            # Log the model
            mlflow.catboost.log_model(
                cb_model=model,
                artifact_path=artifact_path,
                signature=signature,
                input_example=x_train[:5],
                registered_model_name=experiment
            )
        

    def initiate_model_trainer(self, main_train_array, main_test_array, chol_train_array, chol_test_array):
        main_params = {
            "learning_rate": 0.01,
            "iterations": 1000,
            "depth": 6,
            "loss_function": "Logloss",
            "auto_class_weights": "Balanced",
            "random_state": 42,
            "verbose": 0
        }

        chol_params = {
            "learning_rate": 0.05,
            "iterations": 600,
            "depth": 6,
            "loss_function": "Logloss",
            "auto_class_weights": "Balanced",
            "random_state": 42,
            "verbose": 0
        }

        try:
            logging.info("Splitting training and testing input data")
            x_train_main, y_train_main, x_test_main, y_test_main = (main_train_array[:, :-1], main_train_array[:, -1], main_test_array[:, :-1], main_test_array[:, -1])
            x_train_chol, y_train_chol, x_test_chol, y_test_chol = (chol_train_array[:, :-1], chol_train_array[:, -1], chol_test_array[:, :-1], chol_test_array[:, -1])

            main_model = CatBoostClassifier(**main_params)
            threshold = 0.48

            chol_model = CatBoostClassifier(**chol_params)

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
            main_metrics = {
                "precision": main_precision,
                "recall": main_recall,
                "roc_auc": main_roc_auc
            }

            chol_precision = precision_score(y_test_chol, y_pred_chol)
            chol_recall = recall_score(y_test_chol, y_pred_chol)
            chol_roc_auc = roc_auc_score(y_test_chol, y_proba_chol)
            chol_metrics = {
                "precision": chol_precision,
                "recall": chol_recall,
                "roc_auc": chol_roc_auc
            }

            save_object(file_path=self.model_trainer_config.trained_main_model_file_path, obj=main_model)
            save_object(file_path=self.model_trainer_config.trained_chol_model_file_path, obj=chol_model)
            save_object(file_path=self.model_trainer_config.threshold_file_path, obj=threshold)

            # Log metrics to MLflow
            main_log_params = {**main_params, "threshold": threshold}

            self.log_to_mlflow(
                experiment="diabetes-prediction-model",
                model=main_model,
                params=main_log_params,
                metrics=main_metrics,
                x_train=x_train_main,
                artifact_path="main_model_artifacts"
            )

            self.log_to_mlflow(
                experiment="auxiliary-cholesterol-model",
                model=chol_model,
                params=chol_params,
                metrics=chol_metrics,
                x_train=x_train_chol,
                artifact_path="chol_model_artifacts"
            )

            return self.model_trainer_config.trained_main_model_file_path, self.model_trainer_config.trained_chol_model_file_path, self.model_trainer_config.threshold_file_path

        except Exception as e:
            raise CustomException(e, sys)