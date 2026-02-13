from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class EquipmentPrediction(BaseModel):
    """Schema for individual equipment prediction details"""
    id: str
    status: str
    cycles: float
    confidence: float

class PredictionResponse(BaseModel):
    """Schema for the aggregated API response"""
    healthy_count: int
    warning_count: int
    critical_count: int
    total_processed: int
    equipment: List[EquipmentPrediction]

class ErrorResponse(BaseModel):
    detail: str