from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
import numpy as np
import os
import rasterio

from app.schemas.predict import PredictionResponse
from app.models.inference import run_marine_debris_pipeline

router = APIRouter()

# Global memory to act as a mock database for the dashboard's live-polling feature
live_hotspots = []

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Handle generic image upload metadata.
    """
    return {"status": "success", "message": f"File {file.filename} uploaded successfully"}

@router.post("/predict", response_model=PredictionResponse)
async def predict_debris(file: UploadFile = File(...)):
    """
    Takes an actual uploaded MARIDA `.tif` image, runs physical U-Net inference,
    extracts the geographical CRS, and isolates specific debris latitude and longitudes.
    """
    global live_hotspots
    
    if not file.filename.endswith(('.tif', '.tiff')):
        raise HTTPException(status_code=400, detail="Only GeoTIFF files (.tif) are currently supported for geographic extraction.")
    
    # Safely save the file locally for parsing
    os.makedirs("data/uploads", exist_ok=True)
    temp_path = f"data/uploads/{file.filename}"
    
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
        
    try:
        # Load the actual image array natively
        with rasterio.open(temp_path) as src:
            image_data = src.read().astype(np.float32)
            
            # The model requires strictly 11 bands!
            if image_data.shape[0] > 11:
                image_data = image_data[:11, :, :]
        
        # Execute the true mathematical pipeline over the physical file properties
        results = run_marine_debris_pipeline(image_data, tif_path=temp_path)
        live_hotspots = results  # Update the dashboard polling state!
        
        return PredictionResponse(
            status="success",
            message="Real Inference completed successfully.",
            points=results,
            metadata={"filename": file.filename, "found_hotspots": len(results)}
        )
    except Exception as e:
        print(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/visualization-data", response_model=PredictionResponse)
async def get_visualization_data():
    """
    Returns the real aggregated live-hotspots cached from recent user `.tif` predictions.
    """
    return PredictionResponse(
        status="success",
        message="Live visualization data fetched.",
        points=live_hotspots
    )
