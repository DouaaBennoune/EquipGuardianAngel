"""Check which scaler features have raw-scale means vs near-zero means."""
import joblib
from pathlib import Path

scaler = joblib.load(Path(__file__).parent / "pkls" / "LSTM_MODEL_BEST_SO_FAR.pkl")

print(f"{'Feature':<25} {'Mean':>12} {'Scale':>12}  {'Type'}")
print("-" * 70)
for name, mean, scale in zip(scaler.feature_names_in_, scaler.mean_, scaler.scale_):
    ftype = "NORM (near-zero)" if abs(mean) < 1.0 else "RAW (large)"
    print(f"{name:<25} {mean:>12.4f} {scale:>12.6f}  {ftype}")
