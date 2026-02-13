"""Diagnostic script to inspect the LSTM model and scaler."""
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.layers import LSTM
from pathlib import Path

@tf.keras.utils.register_keras_serializable()
class CompatibleLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)
        super().__init__(*args, **kwargs)

root = Path(__file__).parent

# Load model
model = tf.keras.models.load_model(
    str(root / "pkls" / "best_rul_model_LSTM.h5"),
    custom_objects={'LSTM': CompatibleLSTM},
    compile=False
)

# Load scaler
scaler = joblib.load(root / "pkls" / "LSTM_MODEL_BEST_SO_FAR.pkl")

print("=" * 60)
print("MODEL DIAGNOSTICS")
print("=" * 60)
print(f"Model input shape:  {model.input_shape}")
print(f"Model output shape: {model.output_shape}")
model.summary()

print("\n" + "=" * 60)
print("SCALER DIAGNOSTICS")
print("=" * 60)
print(f"Scaler type: {type(scaler).__name__}")
print(f"N features:  {scaler.n_features_in_}")
if hasattr(scaler, 'feature_names_in_'):
    print(f"Feature names: {list(scaler.feature_names_in_)}")
if hasattr(scaler, 'mean_'):
    print(f"Scaler means (first 5): {scaler.mean_[:5]}")
if hasattr(scaler, 'scale_'):
    print(f"Scaler scales (first 5): {scaler.scale_[:5]}")

# Test with random data to see output range
print("\n" + "=" * 60)
print("QUICK PREDICTION TEST")
print("=" * 60)
n_features = model.input_shape[-1]
seq_len = model.input_shape[1] if model.input_shape[1] else 45

# Test 1: zeros
zeros_input = np.zeros((1, seq_len, n_features))
pred_zeros = model.predict(zeros_input, verbose=0)
print(f"Input=zeros       → raw prediction: {pred_zeros.flatten()[0]:.6f}")

# Test 2: ones
ones_input = np.ones((1, seq_len, n_features))
pred_ones = model.predict(ones_input, verbose=0)
print(f"Input=ones        → raw prediction: {pred_ones.flatten()[0]:.6f}")

# Test 3: random normal
rand_input = np.random.randn(1, seq_len, n_features)
pred_rand = model.predict(rand_input, verbose=0)
print(f"Input=random      → raw prediction: {pred_rand.flatten()[0]:.6f}")

# Test 4: large negative (simulating degraded sensors)
neg_input = np.full((1, seq_len, n_features), -2.0)
pred_neg = model.predict(neg_input, verbose=0)
print(f"Input=-2.0        → raw prediction: {pred_neg.flatten()[0]:.6f}")

# Test 5: linearly increasing (simulating degradation)
linear_input = np.linspace(-3, 3, seq_len * n_features).reshape(1, seq_len, n_features)
pred_linear = model.predict(linear_input, verbose=0)
print(f"Input=linear(-3→3)→ raw prediction: {pred_linear.flatten()[0]:.6f}")

print(f"\nAll × 125: zeros={pred_zeros.flatten()[0]*125:.1f}, "
      f"ones={pred_ones.flatten()[0]*125:.1f}, "
      f"rand={pred_rand.flatten()[0]*125:.1f}, "
      f"neg={pred_neg.flatten()[0]*125:.1f}, "
      f"linear={pred_linear.flatten()[0]*125:.1f}")
