import pandas as pd
import yaml
import pickle
import mlflow
import os

from logger_utils import get_logger
from mlflow_tracking import setup_mlflow
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load parameters
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Configure logging
logger = get_logger('train_rf', params)

# MLflow_track
setup_mlflow()

def train_rf():
    try:
        logger.info("Loading processed data")
        df = pd.read_csv(params['data']['processed_file'])

        if 'Fraud Category' not in df.columns:
            logger.error("'Fraud Category' column is missing in the dataset.")
            raise KeyError("'Fraud Category' column is missing in the dataset.")
        
        x = df.drop(columns=['Fraud Category'])
        y = df['Fraud Category']

        # Label encode if needed
        y = LabelEncoder().fit_transform(y)

        logger.info("Splitting data into train/test")
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=params['rf_model']['test_size'],\
                                                             random_state=params['rf_model']['random_state'])


        with mlflow.start_run(run_name="RandomForest_Model"):

            logger.info("Initializing Random Forest Classifier")
            model = RandomForestClassifier(random_state=params['rf_model']['random_state'])

            param_grid=params['rf_model']['param_grid']
            grid_search=GridSearchCV(model, param_grid=param_grid, cv=params['rf_model']['cv_folds'],
                                      n_jobs=-1, verbose=2, scoring='accuracy')

            logger.info("Training RandomForest with GridSearchCV")
            grid_search.fit(X_train, y_train)

            for param, metric in zip(grid_search.cv_results_['params'],grid_search.cv_results_['mean_test_score']):
                with mlflow.start_run(nested=True) as child:
                    logger.info(f"Logging hyperparameters: {param}, Mean Test Score: {metric}")
                    mlflow.log_params(param) 
                    mlflow.log_metric("mean_test_score", metric)

            logger.info("Predicting on test set")
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)

            best_param=grid_search.best_params_
            best_score=grid_search.best_score_

            mlflow.log_params(best_param)

            logger.info(f"Model Accuracy: {best_score}")
            report = classification_report(y_test, y_pred)
            logger.info(f"Classification Report:\n{report}")

            # Log to MLflow
            mlflow.log_metric("accuracy", best_score)
            mlflow.sklearn.log_model(model,'rf_model')
            mlflow.log_metric('best_score', best_score)
            mlflow.sklearn.log_model(best_model, 'rf_model')
            
            # Save model locally
            os.makedirs("models", exist_ok=True)
            model_path = "models/rf_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(best_model, f)
            logger.info(f"Random Forest model saved to {model_path}")

    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    train_rf()
