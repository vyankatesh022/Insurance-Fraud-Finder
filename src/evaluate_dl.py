import pandas as pd
import yaml
import os
import json
import pickle
import mlflow
import numpy as np

from logger_utils import get_logger
from mlflow_tracking import setup_mlflow

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
                            f1_score, roc_auc_score, classification_report, confusion_matrix




# Load parameters
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Configure logging
logger = get_logger('evaluate_dl', params)

# MLflow_track
setup_mlflow()

def evaluate_dl():
    try:
        logger.info("Load Data in Evaluate Dl")
        df = pd.read_csv(params["data"]["processed_file"])

        X = df.drop(columns=["Fraud Category"])
        y = LabelEncoder().fit_transform(df["Fraud Category"])

        num_classes = len(np.unique(y))
        y_cat = to_categorical(y) if num_classes > 2 else y

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(X)

        xtrain, xtest, ytrain, ytest = train_test_split(x_scaled, y_cat,test_size=params["dl_model"]["test_size"],
                                                    random_state=params["dl_model"]["random_state"])
    
        #  Load trained model 
        model_path = "models/dl_model.pkl"
        if not os.path.exists(model_path):
            logger.error(" Model file not found")
            return

        logger.info(f"Loading model from {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Predictions
        logger.info("Predicting on test data")
        y_pred_probs = model.predict(xtest)

        if num_classes > 2:
            y_pred = np.argmax(y_pred_probs, axis=1)
            y_true = np.argmax(ytest, axis=1)
        else:
            y_pred = (y_pred_probs > 0.5).astype(int).ravel()
            y_true = ytest

        # Metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        try:
            if num_classes > 2:
                auc = roc_auc_score(to_categorical(y_true), y_pred_probs, multi_class="ovr")
            else:
                auc = roc_auc_score(y_true, y_pred_probs)
        except ValueError as e:
            logger.warning(f"AUC could not be calculated: {e}")
            auc = None

        logger.info(f"Metrics -> Accuracy={acc:.4f},Precision={prec:.4f},\
                     Recall={rec:.4f}, F1={f1:.4f}, AUC={auc}")

        cls_report = classification_report(y_true, y_pred)
        logger.info(f"Classification Report:\n{cls_report}")

        conf_mat = confusion_matrix(y_true, y_pred)
        logger.info(f"Confusion Matrix:\n{conf_mat}")

        # Log to MLflow
        with mlflow.start_run(run_name="DL_Evaluation"):
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            if auc is not None:
                mlflow.log_metric("auc", auc)

            metrics_data = {
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1 Score': f1,
                'AUC': auc 
            }

            os.makedirs("metrics", exist_ok=True)
            report_path = "metrics/dl_evaluation_report.json"
            with open(report_path, "w") as f:
                json.dump(metrics_data, f, indent=4)

            mlflow.log_artifact(report_path)
            logger.info(f" Metrics saved to {report_path}")

        logger.info(" Evaluation completed successfully")

    except Exception as e:
        logger.error(f" Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    evaluate_dl()
