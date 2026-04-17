from pydantic import BaseModel
from typing import List, Optional

class GeoPoint(BaseModel):
    lat: float
    lon: float
    probability: float

class ClusterPoint(BaseModel):
    center: List[float]
    density: int

class PredictionResponse(BaseModel):
    status: str
    message: str
    points: List[GeoPoint] = []
    clusters: Optional[List[ClusterPoint]] = None
    metadata: Optional[dict] = None

class PatchInferenceRequest(BaseModel):
    bbox: List[float] # [lon1, lat1, lon2, lat2]
    resolution: int = 10
    date_range: str = "last_3_days"
