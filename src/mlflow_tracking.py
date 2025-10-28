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
    if os.getenv('GITHUB_ACTIONS')=='true':
        logger.info('Running github → using DagsHub MLflow tracking.')
        dagshub_name = os.getenv('DAGSHUB_NAME', '')
        dagshub_repro = os.getenv('DAGSHUB_REPRO', '')

        if not dagshub_name or not dagshub_repro:
            logger.warning("⚠️ DAGSHUB_NAME or DAGSHUB_REPRO environment variables are not set.")

        # Dagshub setup
        dagshub.init(repo_owner=dagshub_name, \
                    repo_name=dagshub_repro, mlflow=True)

        uri=f'https://dagshub.com/{dagshub_name}/{dagshub_repro}.mlflow'
        os.environ['MLFLOW_USERNAME']=dagshub_name
        os.environ['MLFLOW_SECRET_KEY']=os.getenv("DAGSHUB_KEY", "")
    else:
        logger.info("Running locally → using local MLflow tracking.")
        uri = "http://127.0.0.1:5000"

    mlflow.set_tracking_uri(uri)
    logger.info(f"MLflow tracking URI set to: {uri}")
    return uri