from pathlib import Path
import numpy as np 
import pandas as pd
import os
import joblib
from backend.core.config import logger
from fastapi import HTTPException
from dotenv import load_dotenv
load_dotenv()

CONSTANT_SENSORS =['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 
                    'sensor_16', 'sensor_18', 'sensor_19']

scaler_path_env = os.getenv("scaler_path")
if scaler_path_env is None:
    raise RuntimeError("Environment variable 'scaler_path' is not set")

scaler_path = Path(scaler_path_env)

print(f"\n{'='*50}\nPREPROCESSING STARTUP DIAGNOSTICS\n{'='*50}")
print(f"SCALER PATH: {scaler_path}[{'EXISTS' if scaler_path.exists() else 'MISSING'}]")

# Load Scaler
if not scaler_path.exists():
    raise FileNotFoundError("Scaler missing!")
scaler = joblib.load(scaler_path)
print(f"{'='*50}\n SUCCESS: Scaler is ready\n{'='*50}\n")


def preprocessing(df) -> tuple[pd.DataFrame, list]:
    scaler_required_cols =[
        'setting1', 'setting2', 'sensor_2', 'sensor_3', 'sensor_4', 
        'sensor_7', 'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12', 
        'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21'
    ]
    
    df = df.copy()
    
    # Drop Constant Sensors
    df = df.drop(columns=CONSTANT_SENSORS, errors='ignore')
    
    # Standard Production null handling
    df = df.dropna(subset=['id', 'cycle'])
    df = df.ffill().bfill()
    df = df.fillna(0)

    # Schema Validation
    missing_cols =[col for col in scaler_required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Validation Failed: Missing columns: {missing_cols}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid CSV Schema. Missing columns required by the model: {missing_cols}"
        )

    # Apply final scaling
    X_input = df[scaler_required_cols]
    X_scaled_values = scaler.transform(X_input)

    # Rebuild the scaled dataframe
    df_scaled = pd.DataFrame(X_scaled_values, columns=scaler_required_cols)
    df_scaled['id'] = df['id'].values
    df_scaled['cycle'] = df['cycle'].values
    
    testing_cols =[col for col in df_scaled.columns if col not in ['id', 'cycle']]

    return df_scaled, testing_cols

def windowings(df_scaled, testing_cols):
    min_sequence = 44  
    
    # Filter out engines that don't have enough history
    engine_counts = df_scaled.groupby('id').size()
    valid_engines = engine_counts[engine_counts >= min_sequence]
    
    if valid_engines.empty:
        logger.warning(f"Validation Warning: No engines found with sufficient data length ({min_sequence} cycles).")
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient data: At least one engine 'id' must have {min_sequence} cycles of history."
        )

    logger.info(f" CSV Validation Passed: {len(df_scaled)} rows, {len(df_scaled['id'].unique())} unique assets.")
    
    X_windows = []
    engine_ids =[]

    for engine_id, engine_data in df_scaled.groupby('id'):
        if len(engine_data) >= min_sequence:
            # Take the last 44 rows for this engine
            window = engine_data[testing_cols].tail(min_sequence).values
            
            # FLATTEN the entire 44x16 window into a single 1D array of 704 features!
            flat_window = window.flatten() 
            
            X_windows.append(flat_window)
            engine_ids.append(engine_id)

    if not X_windows:
        logger.warning("No engines found with enough data.")
        return [],[]
    else: 
        return X_windows, engine_ids