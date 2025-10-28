import os
import logging
import mlflow
import yaml
import dagshub

# Load parameters from YAML file
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Configure logging
log_file=params['logging']['file']

logging.basicConfig(level=params['logging']['level'])
logger = logging.getLogger('mlflow')

# File handler for logging to a file
if not logger.handlers:
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level=params['logging']['level'])
    file_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))  
    logger.addHandler(file_handler)

def setup_mlflow():
    dagshub_name='vyankatesh'
    dagshub_repro='Insurance-Fraud-Finder'
    
    # Dagshub setup
    dagshub.init(repo_owner=dagshub_name,repo_name=dagshub_repro, mlflow=True)
    
    uri=f'https://dagshub.com/{dagshub_name}/{dagshub_repro}.mlflow'

    mlflow.set_tracking_uri(uri)
    logger.info(f"MLflow tracking URI set to: {uri}")
    return uri