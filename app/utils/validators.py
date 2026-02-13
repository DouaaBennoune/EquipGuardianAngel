import pandas as pd
import numpy as np
from fastapi import HTTPException
from app.core.config import logger

# The exact schema required by the model pipeline
REQUIRED_COLUMNS = ['sensor_11', 'sensor_14', 'sensor_2', 'sensor_12_norm', 'sensor_12', 
            'sensor_11_norm', 'setting2', 'sensor_15','cycle', 'sensor_15_norm', 'sensor_13',
            'sensor_13_norm', 'id','sensor_7_norm', 'sensor_21_norm', 'sensor_17', 'sensor_21',
            'sensor_4_norm', 'sensor_17_norm', 'sensor_4', 'sensor_9', 'sensor_3_norm',
            'sensor_2_norm', 'sensor_20_norm', 'setting1', 'sensor_9_norm', 'sensor_3',
            'sensor_8', 'sensor_8_norm', 'sensor_20', 'sensor_7']
            

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
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in current_columns]

    if missing_cols:
        logger.error(f"Validation Failed: Missing columns: {missing_cols}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid CSV Schema",
                "missing_columns": missing_cols,
                "required_count": len(REQUIRED_COLUMNS),
                "received_count": len(df.columns)
            }
        )

    # 4. Check for Minimum Sequence Length
    # Since the LSTM model uses a sliding window of 45, an engine with fewer rows cannot be predicted
    min_sequence = 45
    engine_counts = df.groupby('id').size()
    valid_engines = engine_counts[engine_counts >= min_sequence]
    
    if valid_engines.empty:
        logger.warning("Validation Warning: No engines found with sufficient data length (45 cycles).")
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient data: At least one 'id' must have {min_sequence} cycles of history."
        )

    logger.info(f"✅ CSV Validation Passed: {len(df)} rows, {len(df['id'].unique())} unique assets.")
    return True