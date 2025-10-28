import pandas as pd
import yaml
import pickle
import os
import logging
import mlflow
import numpy as np
import json

from logger_utils import get_logger
from mlflow_tracking import setup_mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
                            f1_score, confusion_matrix, classification_report


# Load parameters
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Configure logging
logger = get_logger('train_rf', params)

# MLflow_track
setup_mlflow()


def evaluate_rf():
    try:
        logger.info("Starting Random Forest model evaluation")

        # Load processed data
        df = pd.read_csv(params["data"]["processed_file"])
        x = df.drop(columns=["Fraud Category"])
        y = df["Fraud Category"]

        # Label encode if necessary
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Split dataset (same params as training)
        x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=params["rf_model"]["test_size"],
            random_state=params["rf_model"]["random_state"])

        # Load trained model
        model_path = "models/rf_model.pkl"
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Loaded trained model from {model_path}")

        # Predict
        y_pred = model.predict(x_test)

        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        cls_report = classification_report(y_test, y_pred)

        # Log to console and file
        logger.info(f"Accuracy: {acc}")
        logger.info(f"Precision: {prec}")
        logger.info(f"Recall: {rec}")
        logger.info(f"F1 Score: {f1}")
        logger.info(f"Confusion Matrix:\n{cm}")
        logger.info(f"Classification Report:\n{cls_report}")

        # Save metrics locally
        os.makedirs("metrics", exist_ok=True)
        metrics_path = "metrics/rf_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1),
                "confusion_matrix": cm.tolist()
            }, f)
        logger.info(f"Metrics saved to {metrics_path}")

        # Log metrics to MLflow/DagsHub
        with mlflow.start_run(run_name="RandomForest_Evaluation"):
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_artifact(metrics_path)

        logger.info("Evaluation completed and metrics logged successfully")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    evaluate_rf()