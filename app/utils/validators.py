import pandas as pd
import numpy as np
from fastapi import HTTPException
from app.core.config import logger



CONSTANT_SENSORS = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 
                    'sensor_16', 'sensor_18', 'sensor_19']

# Critical sensors for RUL prediction (proven in research)
CRITICAL_SENSORS = ['id','cycle',
    'setting1', 'setting2', 'sensor_2', 'sensor_3', 'sensor_4',
    'sensor_7', 'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12',
    'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20',
    'sensor_21' 
]    

def validate_csv_structure(df: pd.DataFrame) -> bool:
    """
    Validates the structure, columns, and data integrity of the uploaded CSV.
    """
    # 1. Check if DataFrame is empty
    if df.empty:
        logger.error("Validation Failed: Uploaded file is empty.")
        raise HTTPException(
            status_code=400,
            detail="The uploaded CSV file contains no data."
        )

    # 2. Check for Missing Columns
    # We use a set for faster comparison
    current_columns = set(df.columns)
    missing_cols = [col for col in CRITICAL_SENSORS if col not in current_columns]

    if missing_cols:
        logger.error(f"Validation Failed: Missing columns: {missing_cols}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid CSV Schema",
                "missing_columns": missing_cols,
                
            }
        )


    
    logger.info(f"✅ CSV Validation Passed: {len(df)} rows, {len(df['id'].unique())} unique assets.")
    return True
