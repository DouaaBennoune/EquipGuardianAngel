from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import io
from app.models.schemas import PredictionResponse
from app.services.ml_engine import MaintenancePredictor
from app.utils.validators import validate_csv_structure
from app.core.config import logger

router = APIRouter()
ml_service = MaintenancePredictor()

@router.post("/predict", response_model=PredictionResponse)
async def predict_equipment_health(file: UploadFile = File(...)):
    """
    Upload a CSV file to get Predictive Maintenance insights.
    
    - **file**: CSV containing sensor data and an 'id' column.
    """
    # Check file extension
    if not file.filename.endswith(('.csv', '.txt')):
        raise HTTPException(status_code=400, detail="File must be a CSV or TXT file")

    try:
        content = await file.read()
        
        # Determine separator (comma for CSV, space for TXT/CMAPSS)
        sep = ',' if file.filename.endswith('.csv') else r'\s+'
        
        # Read Data
        df = pd.read_csv(io.StringIO(content.decode('utf-8')), sep=sep)
        
        # 1. Validate Structure
        validate_csv_structure(df)
        
        # 2. Process & Predict (FIXED METHOD NAME)
        predictions = ml_service.predict(df)
        
        # 3. Aggregate Stats for Dashboard
        counts = {'Healthy': 0, 'Warning': 0, 'Critical': 0}
        for p in predictions:
            status_key = p['status']
            if status_key in counts:
                counts[status_key] += 1
                
        return PredictionResponse(
            healthy_count=counts['Healthy'],
            warning_count=counts['Warning'],
            critical_count=counts['Critical'],
            total_processed=len(predictions),
            equipment=predictions
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(status_code=500, detail=f"Internal processing error: {str(e)}")