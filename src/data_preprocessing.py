import pandas as pd
import yaml
import logging
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
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

        # Feature Engineering: Dates
        date_cols = ['POLICYRISKCOMMENCEMENTDATE', 'Date of Death', 'INTIMATIONDATE']
        for col in date_cols:
            if col in df.columns:
                # Replace '-' with NaN
                df[col] = df[col].replace('-', np.nan)
                df[col] = pd.to_datetime(df[col], format='%d-%m-%Y', errors='coerce')
        
        if 'Date of Death' in df.columns and 'POLICYRISKCOMMENCEMENTDATE' in df.columns:
            df['Days_to_Death'] = (df['Date of Death'] - df['POLICYRISKCOMMENCEMENTDATE']).dt.days
        
        if 'INTIMATIONDATE' in df.columns and 'Date of Death' in df.columns:
            df['Days_to_Intimation'] = (df['INTIMATIONDATE'] - df['Date of Death']).dt.days
            
        # Drop original date columns
        existing_date_cols = [c for c in date_cols if c in df.columns]
        df.drop(columns=existing_date_cols, inplace=True)
        logger.info("Extracted date features and dropped original date columns")

        # Separate target before imputation
        target_col = 'Fraud Category'
        y = None
        if target_col in df.columns:
            y = df.pop(target_col)

        # Impute missing values
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        cat_cols = df.select_dtypes(include=['object']).columns

        if len(num_cols) > 0:
            num_imputer = SimpleImputer(strategy='median')
            df[num_cols] = num_imputer.fit_transform(df[num_cols])
        
        if len(cat_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
        
        logger.info("Missing values imputed with median (numerical) and mode (categorical)")

        if y is not None:
            df[target_col] = y

        # Label encoding
        if params['data']['label_encoding']:
            for col in df.select_dtypes(include='object').columns:
                if col != target_col:
                    df[col] = LabelEncoder().fit_transform(df[col])
            logger.info("Label encoding applied to categorical columns")

        # Scaling
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if target_col in numeric_cols:
            numeric_cols = numeric_cols.drop(target_col)
            
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