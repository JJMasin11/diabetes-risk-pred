import os
import sys
import pandas as pd
from dotenv import load_dotenv
from src.exception import CustomException
from src.logger import logging
from sqlalchemy import create_engine, text

def validate_schema(df):
    binary_columns = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex']

    # Check if binary columns contain only 0 and 1
    try:
        assert df[binary_columns].isin([0, 1]).all().all()
    except Exception as e:
        raise CustomException(e, sys)
    
    # Check if Diabetes_012 column contains only 0, 1, and 2
    try:
        assert df['Diabetes_012'].isin([0, 1, 2]).all()
    except Exception as e:
        raise CustomException(e, sys)
    
    # Check if GenHlth is between 1 and 5
    try:
        assert df['GenHlth'].between(1, 5).all()
    except Exception as e:
        raise CustomException(e, sys)
    
    # Check if MentHlth and PhysHlth are between 0 and 30
    try:
        assert df['MentHlth'].between(0, 30).all()
        assert df['PhysHlth'].between(0, 30).all()
    except Exception as e:
        raise CustomException(e, sys)
    
    # Check if Age is between 1 and 13
    try:
        assert df['Age'].between(1, 13).all()
    except Exception as e:
        raise CustomException(e, sys)
    
    # Check if Education is between 1 and 6
    try:
        assert df['Education'].between(1, 6).all()
    except Exception as e:
        raise CustomException(e, sys)
    
    # Check if Income is between 1 and 8
    try:
        assert df['Income'].between(1, 8).all()
    except Exception as e:
        raise CustomException(e, sys)

    # Check if all columns are of number type
    try:
        assert len(df.select_dtypes(include=['int64', 'float64']).columns) == len(df.columns)
    except Exception as e:
        raise CustomException(e, sys)


def rename_columns(df):
    column_mapping = {
        'Diabetes_012': 'diabetes_012',
        'HighBP': 'high_bp',
        'HighChol': 'high_chol',
        'CholCheck': 'chol_check',
        'BMI': 'bmi',
        'Smoker': 'smoker',
        'Stroke': 'stroke',
        'HeartDiseaseorAttack': 'heart_disease_or_attack',
        'PhysActivity': 'phys_activity',
        'Fruits': 'fruits',
        'Veggies': 'veggies',
        'HvyAlcoholConsump': 'hvy_alcohol_consump',
        'AnyHealthcare': 'any_healthcare',
        'NoDocbcCost': 'no_doc_bc_cost',
        'GenHlth': 'gen_hlth',
        'MentHlth': 'ment_hlth',
        'PhysHlth': 'phys_hlth',
        'DiffWalk': 'diff_walk',
        'Sex': 'sex',
        'Age': 'age',
        'Education': 'education',
        'Income': 'income'
    }
    return df.rename(columns=column_mapping)


def etl_pipeline():
    # Load environment variables
    load_dotenv()

    # Connect to the PostgreSQL database
    engine = create_engine(f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}")
    
    # Check if data has already been loaded, skip if it has
    try:
        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM health_indicators")).scalar()
            if count > 0:
                logging.info("Data already loaded. Skipping ETL process.")
                return
    except Exception as e:
        raise CustomException(e, sys)
    
    # Load health indicators data from CSV
    df = pd.read_csv('notebook/data/diabetes_012_health_indicators_BRFSS2015.csv')
    logging.info(f"Extracted {len(df)} rows from CSV.")

    # Remove duplicates
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates()
        logging.info(f"Removed duplicates. {len(df)} rows remaining.")

    # Schema Enforcement
    validate_schema(df)
    logging.info("Data validation passed. Loading data into database.")

    # Load data into PostgreSQL database
    df = rename_columns(df)

    try:
        df.to_sql('health_indicators', engine, if_exists='append', index=False)
    except Exception as e:
        raise CustomException(e, sys)
    logging.info("Data loaded successfully into the database.")

if __name__ == "__main__":
    etl_pipeline()