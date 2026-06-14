import pandas as pd
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
import pickle

from logger_utils import get_logger
from mlflow_tracking import setup_mlflow
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# Load parameters
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Configure logging
logger = get_logger('train_pytorch', params)

# MLflow_track
setup_mlflow()

def train_pytorch():
    # Load Data
    df = pd.read_csv(params["data"]["processed_file"])
    x = df.drop(columns=["Fraud Category"])
    y = LabelEncoder().fit_transform(df["Fraud Category"])

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    xtrain, xtest, ytrain, ytest = train_test_split(
        x_scaled, y,
        test_size=params["pytorch_model"]["test_size"],
        random_state=params["pytorch_model"]["random_state"]
    )
    
    logger.info("Applying RandomOverSampler to balance training data")
    ros = RandomOverSampler(random_state=params['pytorch_model']['random_state'])
    xtrain, ytrain = ros.fit_resample(xtrain, ytrain)

    # Convert to PyTorch tensors
    xtrain_t = torch.tensor(xtrain, dtype=torch.float32)
    ytrain_t = torch.tensor(ytrain, dtype=torch.long)
    xtest_t = torch.tensor(xtest, dtype=torch.float32)
    ytest_t = torch.tensor(ytest, dtype=torch.long)

    train_dataset = TensorDataset(xtrain_t, ytrain_t)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=params["pytorch_model"]["batch_size"], 
        shuffle=True
    )

    input_dim = xtrain.shape[1]
    num_classes = len(pd.Series(y).unique())

    with mlflow.start_run(run_name="PyTorch_Model"):
        logger.info(f"Building PyTorch sequential model with input_dim={input_dim}")
        mlflow.log_params(params["pytorch_model"])

        model = nn.Sequential(
            nn.Linear(input_dim, params["pytorch_model"]["hidden_size1"]),
            nn.ReLU(),
            nn.Dropout(params["pytorch_model"]["dropout"]),
            nn.Linear(params["pytorch_model"]["hidden_size1"], params["pytorch_model"]["hidden_size2"]),
            nn.ReLU(),
            nn.Dropout(params["pytorch_model"]["dropout"]),
            nn.Linear(params["pytorch_model"]["hidden_size2"], num_classes)
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params["pytorch_model"]["learning_rate"])
        
        epochs = params["pytorch_model"]["epochs"]
        
        logger.info(f"Model initialized, starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
        
        # Evaluate Model
        model.eval()
        with torch.no_grad():
            outputs = model(xtest_t)
            loss_val = criterion(outputs, ytest_t).item()
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == ytest_t).sum().item()
            acc_val = correct / ytest_t.size(0)
            
        mlflow.log_metric("val_loss", loss_val)
        mlflow.log_metric("val_accuracy", acc_val)
        
        # Log the model in MLflow
        mlflow.pytorch.log_model(model, name='PyTorch Model', serialization_format='pt2', input_example=xtrain_t[:5].numpy().copy())
        
        os.makedirs("models", exist_ok=True)
        model_path = "models/pytorch_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
            
        logger.info(f"Training complete — val_accuracy: {acc_val:.4f}, val_loss: {loss_val:.4f}")

if __name__ == "__main__":
    train_pytorch()
