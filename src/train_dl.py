import pandas as pd
import yaml
import pickle
import os
import mlflow
import mlflow.keras
import numpy as np


from logger_utils import get_logger
from mlflow_tracking import setup_mlflow
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical



# Load parameters
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Configure logging
logger = get_logger('train_dl', params)

# MLflow_track
setup_mlflow()

def train_dl():
    # Load Data
    df = pd.read_csv(params["data"]["processed_file"])
    x = df.drop(columns=["Fraud Category"])
    y = LabelEncoder().fit_transform(df["Fraud Category"])

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Determine classification type automatically
    num_classes = len(np.unique(y))
    if num_classes > 2:
        y = to_categorical(y)

    xtrain, xtest, ytrain, ytest = train_test_split(x_scaled, y,test_size=params["dl_model"]["test_size"],
                                                    random_state=params["dl_model"]["random_state"])


    with mlflow.start_run(run_name="DL Model"):
        mlflow.log_params(params["dl_model"])

        # Build model
        model = Sequential([
            Dense(128, activation="relu", input_shape=(xtrain.shape[1],)),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer=params["dl_model"]["optimizer"], 
                      loss=params["dl_model"]["loss"], metrics=["accuracy"])


        history = model.fit(
            xtrain, ytrain,
            epochs=params["dl_model"]["epochs"],
            batch_size=params["dl_model"]["batch_size"],
            validation_split=params["dl_model"]["validation_split"],
            verbose=1)

        # Evaluate Model
        loss_val, acc_val = model.evaluate(xtest, ytest)
        mlflow.log_metric("val_loss", loss_val)
        mlflow.log_metric("val_accuracy", acc_val)

        # Log the model in MLflow
        mlflow.keras.log_model(model, 'DL Model')

        os.makedirs("models", exist_ok=True)
        model_path = "models/dl_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact(model_path)

        logger.info(f"Training complete — val_accuracy: {acc_val:.4f}, val_loss: {loss_val:.4f}")

if __name__ == "__main__":
    train_dl()