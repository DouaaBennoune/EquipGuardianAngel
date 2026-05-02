import requests
import io

# Simulate what Streamlit does with uploaded files
with open('/app/fd002.csv', 'rb') as f:
    csv_bytes = f.read()

# Test 1: Direct file object (what your code does)
print("=== Test 1: Direct file object ===")
with open('/app/fd002.csv', 'rb') as f:
    try:
        r = requests.post('http://equip-backend:8000/api/v1/predict', files={'file': f})
        print(f"Status: {r.status_code}")
    except Exception as e:
        print(f"Error: {e}")

# Test 2: BytesIO (what Streamlit UploadFile might do)
print("\n=== Test 2: BytesIO ===")
bio = io.BytesIO(csv_bytes)
bio.name = 'test.csv'
try:
    r = requests.post('http://equip-backend:8000/api/v1/predict', files={'file': bio})
    print(f"Status: {r.status_code}")
except Exception as e:
    print(f"Error: {e}")

# Test 3: With explicit filename
print("\n=== Test 3: With filename tuple ===")
try:
    r = requests.post('http://equip-backend:8000/api/v1/predict', files={'file': ('fd002.csv', csv_bytes)})
    print(f"Status: {r.status_code}")
except Exception as e:
    print(f"Error: {e}")
