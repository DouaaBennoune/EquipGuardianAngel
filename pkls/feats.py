import joblib
import pandas as pd
from pathlib import Path

# Load your scaler
scaler_path = Path("pkls/SVM_model_scaler.pkl")
scaler = joblib.load(scaler_path)

print(f"✅ Scaler loaded successfully.")
print(f"🔢 Expected Feature Count: {scaler.n_features_in_}")

if hasattr(scaler, 'feature_names_in_'):
    print("\n📋 THE EXACT 32 COLUMNS YOU NEED:")
    print(list(scaler.feature_names_in_))
else:
    print("\n⚠️ Scaler does not have feature names saved.")
    print("This happens if you trained on a numpy array instead of a pandas DataFrame.")
    print("You must manually check your training notebook to see which columns were used.")