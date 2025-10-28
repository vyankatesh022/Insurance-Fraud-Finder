import pandas as pd
import yaml
import logging
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from logger_utils import get_logger

# Load parameters from YAML file
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

logger=get_logger('data_preprocessing',params)

def preprocess_data():
    logger.info('Starting data preprocessing')
    file_path = params['data']['raw_path']

    try:
        # Load raw data
        df = pd.read_csv(file_path)
        logger.info(f'Loaded data from {file_path}')

        # Drop columns if exist
        drop_cols = [c for c in params['data']['drop_columns'] if c in df.columns]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)
            logger.info(f"Dropped columns: {drop_cols}")
        
        # Convert money columns
        for col in params['data']['money_columns']:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '', regex=False).astype(float)
        logger.info("Converted money columns to float")

        # Impute missing values
        df.fillna(params['data']['missing_value'], inplace=True)
        logger.info(f"Missing values imputed with: {params['data']['missing_value']}")

        # Label encoding
        if params['data']['label_encoding']:
            for col in df.select_dtypes(include='object').columns:
                df[col] = LabelEncoder().fit_transform(df[col])
            logger.info("Label encoding applied to categorical columns")

        # Scaling
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        if params['data']['scaling_method'] == 'StandardScaler':
            df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])
        
        elif params['data']['scaling_method'] == 'MinMaxScaler':
            df[numeric_cols] = MinMaxScaler().fit_transform(df[numeric_cols])
        
        logger.info(f"Scaled features using {params['data']['scaling_method']}")

        # Save processed data
        os.makedirs(params['data']['processed_dir'], exist_ok=True)
        processed_data_path = params['data']['processed_file']
        df.to_csv(processed_data_path, index=False)
        logger.info(f"Processed data saved to {processed_data_path}")

        # DVC versioning
        os.system(f'dvc add {processed_data_path}')
        logger.info("Processed data added to DVC")

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise
if __name__ == "__main__":
    preprocess_data()