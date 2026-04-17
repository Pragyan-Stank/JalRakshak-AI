import numpy as np
import torch
import torch.nn.functional as F
import os
import rasterio

from typing import Tuple, List

# ── project imports ─────────────────────────────────────────
try:
    from train_unet import LightUNet, Config
except ImportError:
    LightUNet = None
    Config = None

from app.services.fdi import calculate_fdi, combine_fdi_and_predictions
from app.utils.pixel_latlon import pixel_coords_from_prediction

# ── Metadata ───────────────────────────────────────────
CLASS_NAMES = {
    0: "Unlabeled",       1: "Marine Debris",
    2: "Dense Sargassum", 3: "Sparse Sargassum",
    4: "Natural Organic", 5: "Ship",
    6: "Clouds",          7: "Marine Water",
    8: "Sediment Water",  9: "Foam",
   10: "Turbid Water",   11: "Shallow Water",
   12: "Waves",          13: "Cloud Shadows",
   14: "Wakes",          15: "Mixed Water",
}

# ── Normalisation constants ──────────────────────────────────
MEAN = np.array([0.0582, 0.0745, 0.0921, 0.0850, 0.0980, 0.1580,
                 0.1780, 0.1830, 0.1550, 0.0985, 0.0680],
                dtype=np.float32)[:, None, None]

STD  = np.array([0.0200, 0.0310, 0.0360, 0.0420, 0.0410, 0.0680,
                 0.0730, 0.0760, 0.0730, 0.0560, 0.0340],
                dtype=np.float32)[:, None, None]


class UNetInferencer:
    def __init__(self, model_path: str = "trained_models/best_model.pth"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
    
    def _load_model(self):
        if not LightUNet:
            print("Warning: train_unet.py not found in backend/. Running mock inference.")
            return None
            
        print("Loading MARIDA DL DL framework...")
        
        # Load exactly as specified in the provided inference script
        in_channels = Config().IN_CHANNELS if Config else 11
        num_classes = Config().NUM_CLASSES if Config else 16
        features    = Config().FEATURES if Config else [32, 64, 128, 256]
        
        model = LightUNet(
            in_channels=in_channels,
            num_classes=num_classes,
            features=features
        ).to(self.device)
        
        # Fallback to standard path if exact file is missing
        path_to_load = self.model_path if os.path.exists(self.model_path) else "trained_models/unet_model.pth"
        
        if os.path.exists(path_to_load):
            ckpt = torch.load(path_to_load, map_location=self.device)
            state_dict = ckpt["model"] if "model" in ckpt else ckpt
            model.load_state_dict(state_dict)
            best_miou = ckpt.get("val_miou", "?") if isinstance(ckpt, dict) else "?"
            print(f"Loaded weights (val mIoU = {best_miou})")
        else:
            print(f"Model path {path_to_load} missing! Running randomly initialised mock inference.")
            
        model.eval()
        return model
        
    def predict(self, img_raw: np.ndarray) -> np.ndarray:
        """
        Takes raw 11-band optical image array.
        Applies cleaning, z-score standardisation and torch pipeline.
        Returns prediction probabilities for Marine Debris (Class 1).
        """
        if self.model is None:
            # Fallback mock arrays
            return np.random.uniform(0, 1, size=(img_raw.shape[1], img_raw.shape[2]))

        # Exact robust data processing as provided by script
        img_clean = np.nan_to_num(img_raw, nan=0.0, posinf=0.0, neginf=0.0)
        img_norm = np.clip((img_clean - MEAN) / (STD + 1e-8), -5, 5)

        with torch.no_grad():
            x = torch.from_numpy(img_norm).unsqueeze(0).to(self.device)
            logits = self.model(x)                               # (1,16,256,256)
            
            # Index 1 = Marine Debris probabilities
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
            debris_probs = probs[1, :, :]
            return debris_probs

# Global singleton to avoid intensive model re-loading on every API call
_global_inferencer = None

def get_inferencer():
    global _global_inferencer
    if _global_inferencer is None:
        _global_inferencer = UNetInferencer()
    return _global_inferencer

def run_marine_debris_pipeline(image_data: np.ndarray, tif_path: str = None) -> list:
    """
    Simulates the core logic pipeline: UNet + FDI combined scoring.
    image_data: raw optical array shape MUST BE (11, 256, 256) per MARIDA standard.
    """
    # 1. Run inference (standardisation mapping is handled inside predict)
    inferencer = get_inferencer()
    unet_probs = inferencer.predict(image_data)
    
    # 2. Extract specific bands to generate FDI and combine
    if image_data.shape[0] >= 11:
        nir = image_data[7, :, :]      # B8A
        red_edge = image_data[5, :, :] # B6
        swir = image_data[9, :, :]     # B11 (Index 9 in standard MARIDA 11 bands)
        
        fdi_map = calculate_fdi(nir, red_edge, swir)
        final_probs = combine_fdi_and_predictions(fdi_map, unet_probs)
    else:
        final_probs = unet_probs
        
    # 3. Convert successful hits (> 0.5 prob threshold) to coordinates
    results = []
    y_indices, x_indices = np.where(final_probs > 0.5)
    
    # Utilise local raster coordinates if available
    if tif_path and os.path.exists(tif_path):
        with rasterio.open(tif_path) as src:
            transform = src.transform
            crs = src.crs
            transformer = None
            if crs and crs.to_epsg() != 4326:
                from pyproj import Transformer
                transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                
            for y, x in zip(y_indices, x_indices):
                x_coord, y_coord = transform * (x, y)
                if transformer:
                    lon, lat = transformer.transform(x_coord, y_coord)
                else:
                    lon, lat = x_coord, y_coord
                prob = float(final_probs[y, x])
                results.append({"lat": lat, "lon": lon, "probability": prob})
    else:
        # Fallback dummy logic if run as a raw numerical array without spatial extent
        base_lat, base_lon = 37.7749, -122.4194
        for y, x in zip(y_indices, x_indices):
            lat = base_lat + (y - 128) * 0.0001
            lon = base_lon + (x - 128) * 0.0001
            prob = float(final_probs[y, x])
            results.append({"lat": lat, "lon": lon, "probability": prob})
            
    return results
