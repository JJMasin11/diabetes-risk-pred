import pandas as pd
import mlflow

def log_binary_baseline():
    df = pd.read_csv("etl/binary_baseline_results.csv")
    df.rename(columns={"ROC AUC": "ROC-AUC"}, inplace=True)
    df = df[df["Threshold"] == 0.5]
    df.drop(columns=["Threshold", "F1 Score", "Combined Score"], inplace=True)

    mlflow.set_experiment("diabetes-binary-baseline")

    for _, row in df.iterrows():
        with mlflow.start_run():
            mlflow.set_tag("model_type", row["Model"])
            mlflow.log_metrics({
                "precision": row["Precision"],
                "recall": row["Recall"],
                "roc_auc": row["ROC-AUC"]
            })


def log_cholesterol_baseline():
    df = pd.read_csv("etl/cholesterol_baseline_results.csv")

    mlflow.set_experiment("cholesterol-baseline")

    for _, row in df.iterrows():
        with mlflow.start_run():
            mlflow.set_tag("model_type", row["Model"])
            mlflow.log_metrics({
                "precision": row["Precision"],
                "recall": row["Recall"],
                "roc_auc": row["ROC-AUC"]
            })


def log_experiments():
    mlflow.set_tracking_uri("http://localhost:5000")
    log_binary_baseline()
    log_cholesterol_baseline()


if __name__ == "__main__":
    log_experiments()
