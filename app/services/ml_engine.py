import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.layers import LSTM
from pathlib import Path
from typing import List, Dict
from app.core.config import logger

# FIX: Robustly handle the 'time_major' version mismatch error
@tf.keras.utils.register_keras_serializable()
class CompatibleLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)  # Delete the key that causes the crash
        super().__init__(*args, **kwargs)

class MaintenancePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.original_df = None
        self.seq_length = 45
        self.thresholds = {"critical": 30, "warning": 60}

        # --- DYNAMIC PATH RESOLUTION ---
        # We find the root based on this file's location
        current_file = Path(__file__).resolve()
        # root is 2 levels up from app/services/ml_engine.py
        root_dir = current_file.parent.parent.parent 
        
        # Define paths
        model_path = root_dir / "pkls" / "best_rul_model_LSTM.h5"
        scaler_path = root_dir / "pkls" / "LSTM_MODEL_BEST_SO_FAR.pkl"
        ref_data_path = current_file.parent / "FD001_train_df_normalized_windows.csv"

        print(f"\n{'='*50}\nML ENGINE STARTUP DIAGNOSTICS\n{'='*50}")
        print(f"ROOT DIR: {root_dir}")
        print(f"MODEL PATH: {model_path} [{'EXISTS' if model_path.exists() else 'MISSING'}]")
        print(f"SCALER PATH: {scaler_path} [{'EXISTS' if scaler_path.exists() else 'MISSING'}]")
        print(f"REF DATA: {ref_data_path} [{'EXISTS' if ref_data_path.exists() else 'MISSING'}]")

        try:
            if not model_path.exists() or not scaler_path.exists() or not ref_data_path.exists():
                raise FileNotFoundError("One or more required ML artifacts are missing!")

            # 1. Load Scaler
            self.scaler = joblib.load(scaler_path)
            
            # 2. Load Model (with the LSTM version fix)
            self.model = tf.keras.models.load_model(
                str(model_path), 
                custom_objects={'LSTM': CompatibleLSTM},
                compile=False
            )
            
            # 3. Load Reference Data
            self.original_df = pd.read_csv(ref_data_path)
            
            logger.info("✅ ML Engine loaded successfully.")
            print(f"{'='*50}\n✅ SUCCESS: Model and Scaler are ready\n{'='*50}\n")

        except Exception as e:
            print(f"\n{'!'*50}\n❌ CRITICAL LOAD FAILURE: {e}\n{'!'*50}\n")
            logger.error(f"Failed to load ML artifacts: {e}")
            # We don't raise here to allow the API to start, 
            # but we catch it in predict()

    def _apply_train_normalization(self, test_df, train_df, window_size=45):
        test_df = test_df.copy()
        # Identify sensor columns (exclude id, cycle, and target RUL)
        exclude_cols = ['id', 'cycle', 'setting3']
        sensor_cols = [col for col in train_df.columns if col not in exclude_cols]
        
        global_means = train_df[sensor_cols].mean()
        global_stds = train_df[sensor_cols].std()

        norm_cols = []
        for sensor in sensor_cols:
            if sensor in test_df.columns:
                mean = global_means[sensor]
                std = global_stds[sensor]
                norm_name = f"{sensor}_norm"
                
                # Formula: (x - mean) / std -> then apply rolling mean
                test_df[norm_name] = test_df.groupby("id")[sensor].transform(
                    lambda x: ((x - mean) / (std if std > 1e-6 else 1)).rolling(window=window_size, min_periods=1).mean()
                )
                norm_cols.append(norm_name)

        test_df[norm_cols] = test_df[norm_cols].fillna(0)
        print(norm_cols)
        return test_df, norm_cols

    def predict(self, df: pd.DataFrame) -> List[Dict]:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model or Scaler not loaded.")

        # 1. The EXACT list of columns your scaler expects (in the exact order)
        scaler_required_cols = ['sensor_11', 'sensor_14', 'sensor_2', 'sensor_12_norm', 'sensor_12', 
            'sensor_11_norm', 'setting2', 'sensor_15', 'sensor_15_norm', 'sensor_13',
            'sensor_13_norm', 'sensor_7_norm', 'sensor_21_norm', 'sensor_17', 'sensor_21',
            'sensor_4_norm', 'sensor_17_norm', 'sensor_4', 'sensor_9', 'sensor_3_norm',
            'sensor_2_norm', 'sensor_20_norm', 'setting1', 'sensor_9_norm', 'sensor_3',
            'sensor_8', 'sensor_8_norm', 'sensor_20', 'sensor_7']

        # 2. Filter and reorder the uploaded dataframe to match the scaler
        try:
            # Select only the required columns in the correct order
            X_input = df[scaler_required_cols]

            # Apply scaling
            X_scaled_values = self.scaler.transform(X_input)

            # Create a DataFrame from scaled values (important for the sliding window step)
            df_scaled = pd.DataFrame(X_scaled_values, columns=scaler_required_cols)

            # Ensure 'id' is treated as the original ID for grouping (not scaled)
            df_scaled['id'] = df['id'].values
            df_scaled['cycle'] = df['cycle'].values
            testing_cols = [col for col in df_scaled if col not in ['id']]
        except Exception as e:
            logger.error(f"Scaling failed: {e}")
            raise RuntimeError(f"Data preparation error: {str(e)}")

        # REMOVED: double-normalization was collapsing all features to ~0
        # FD001_test_df_windows, _ = self._apply_train_normalization(df_scaled, self.original_df, window_size=50)

        # 3. Create Sliding Windows
        X_windows = []
        engine_ids = []

        for engine_id, engine_data in df_scaled.groupby('id'):
            if len(engine_data) >= self.seq_length:
                # Take the last window for this engine
                window = engine_data[testing_cols].tail(self.seq_length).values
                X_windows.append(window)
                engine_ids.append(engine_id)

        if not X_windows:
            logger.warning("No engines found with enough data (45 cycles).")
            return []

        # 4. Predict RUL
        try:
            input_array = np.array(X_windows)
            rul_predictions = self.model.predict(input_array)
            rul_preds = rul_predictions.flatten()
        except Exception as e:
            logger.error(f"Prediction crash: {e}")
            raise RuntimeError(f"Model prediction failed: {e}")

        # 5. Format Output
        results = []
        for i, val in enumerate(rul_preds):
            rul_val = val.item()
            rul_val = rul_val * 125.0
            rul = float(rul_val)
            if rul <= self.thresholds['critical']:
                status = "Critical"
            elif rul <= self.thresholds['warning']:
                status = "Warning"
            else:
                status = "Healthy"

            results.append({
                "id": str(engine_ids[i]),
                "status": status,
                "cycles": int(round(rul)),
                "confidence": 0.85
            })

        return results