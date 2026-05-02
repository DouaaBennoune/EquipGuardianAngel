import os
from pathlib import Path
from typing import List, Dict
import joblib
import numpy as np
from dotenv import load_dotenv
from app.core.config import logger

load_dotenv()

class MaintenancePredictor:
    def __init__(self):
        self.model = None
        self.thresholds = {"critical": 30, "warning": 60}

        # Loading model
        model_path = Path(os.getenv("model_path"))
        print(f"MODEL PATH:  {model_path}[{'EXISTS' if model_path.exists() else 'MISSING'}]")

        try:
            if not model_path.exists():
                raise FileNotFoundError("RandomForest model file is missing!")

            self.model = joblib.load(model_path)
            print(f"{'='*50}\n RF model loaded\n{'='*50}\n")

        except Exception as e:
            raise RuntimeError(f"Failed to load ML artifacts: {e}")

    def predict(self, X_windows, engine_ids) -> List[Dict]:
        if self.model is None:
            raise RuntimeError("Model not loaded.")
            
        if not X_windows:
            return[]

        try:
            # For RF, input array will be shape (num_engines, num_features)
            input_array = np.array(X_windows)
            rul_predictions = self.model.predict(input_array)
            rul_preds = rul_predictions.flatten()
        except Exception as e:
            logger.error(f"Prediction crash: {e}")
            raise RuntimeError(f"Model prediction failed: {e}")

        results =[]
        for i, val in enumerate(rul_preds):
            rul_val = float(val) * 125.0  
            if rul_val <= self.thresholds["critical"]:
                status = "Critical"
            elif rul_val <= self.thresholds["warning"]:
                status = "Warning"
            else:
                status = "Healthy"

            results.append({
                "id": str(engine_ids[i]),
                "status": status,
                "cycles": int(round(rul_val)),
                "confidence": 0.85,
            })

        return results