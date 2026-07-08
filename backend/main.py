import os
import sys
import shutil
import uuid
import json
from datetime import datetime
import pickle
import zipfile
import h5py
import pandas as pd
import numpy as np
import torch
from io import BytesIO
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torch.utils.data import DataLoader

# Add current directory to path to import BEST_WITH_CONCAT modules
sys.path.append(str(Path(__file__).parent))

from mid_fusion_lite.config import MidFusionLiteConfig, get_preset_config, BASE_NUMERIC_COLS
from mid_fusion_lite.tabular_engineering_02 import add_feature_engineering
from mid_fusion_lite.tabular_preprocessor_04 import FoldTabularPreprocessor, select_available_columns
from mid_fusion_lite.data import get_image_transforms
from mid_fusion_lite.model import MidFusionLiteModel
from mid_fusion_lite.train import predict_loader

def shades_of_gray(img_rgb: np.ndarray, power: int = 6, gamma: float | None = None) -> np.ndarray:
    """Apply Shades of Gray color constancy algorithm to normalize lighting."""
    img_dtype = img_rgb.dtype
    img_float = img_rgb.astype(np.float32)
    
    if gamma is not None:
        img_float = np.power(img_float, gamma)
        
    img_power = np.power(img_float, power)
    illum = np.power(np.mean(img_power, axis=(0, 1)), 1.0 / power)
    
    illum_norm = np.sqrt(np.sum(np.square(illum)))
    if illum_norm > 0:
        illum = illum / illum_norm
        
    mean_gray = 1.0 / np.sqrt(3.0)
    
    # Scale channels
    scale = mean_gray / (illum + 1e-8)
    img_normalized = img_float * scale[None, None, :]
    
    # Clip and cast
    return np.clip(img_normalized, 0, 255).astype(img_dtype)

from preprocessor_utils import load_or_create_preprocessors

app = FastAPI(title="Skin Lesion Multi-modal Classification API")

PREDICTION_HISTORY = []

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
)

# Directory configuration
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
UPLOADS_DIR = STATIC_DIR / "uploads"

# Ensure directories exist
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static/assets", StaticFiles(directory=str(BASE_DIR.parent / "frontend" / "dist" / "assets")), name="frontend_assets")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

def read_image_from_hdf5(isic_id: str, hdf5_path: Path) -> Optional[np.ndarray]:
    """Load a medical image from the user-provided HDF5 file."""
    if hdf5_path and hdf5_path.exists():
        try:
            with h5py.File(hdf5_path, "r") as h5:
                if isic_id in h5:
                    raw_obj = h5[isic_id][()]
                    if isinstance(raw_obj, np.ndarray):
                        binary = raw_obj.tobytes()
                    else:
                        binary = bytes(raw_obj)
                    from PIL import Image
                    image = Image.open(BytesIO(binary)).convert("RGB")
                    return np.asarray(image)
        except Exception as e:
            print(f"Error reading {isic_id} from HDF5 {hdf5_path}: {e}")
    return None

# Custom Dataset for inference on folder of images
class FolderImageTabularDataset(torch.utils.data.Dataset):
    def __init__(self, ids, tabular, image_folder, hdf5_path=None, transforms=None, use_shades_of_gray=False):
        self.ids = np.asarray(ids).astype(str)
        self.tabular = np.asarray(tabular, dtype=np.float32)
        self.image_folder = Path(image_folder)
        self.hdf5_path = Path(hdf5_path) if hdf5_path is not None else None
        self.transforms = transforms
        self.use_shades_of_gray = use_shades_of_gray

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        isic_id = self.ids[index]
        
        # Look for the image file in the temporary folder
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            p = self.image_folder / f"{isic_id}{ext}"
            if p.exists():
                img_path = p
                break
                
        if img_path is None:
            # Case insensitive check
            for p in self.image_folder.iterdir():
                if p.stem.lower() == isic_id.lower() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    img_path = p
                    break
                    
        if img_path is not None:
            try:
                from PIL import Image
                image = Image.open(img_path).convert("RGB")
                image_array = np.asarray(image)
            except Exception as e:
                print(f"Error reading image {img_path}: {e}. Using placeholder.")
                image_array = np.ones((224, 224, 3), dtype=np.uint8) * 255
        else:
            # Fallback: check if we have the uploaded HDF5 file
            img_arr = None
            if self.hdf5_path is not None:
                img_arr = read_image_from_hdf5(isic_id, self.hdf5_path)
                
            if img_arr is not None:
                image_array = img_arr
                # Save it to image_folder so that the rest of main.py's preview copying logic can find it
                try:
                    from PIL import Image
                    out_img = Image.fromarray(image_array)
                    out_img.save(self.image_folder / f"{isic_id}.jpg")
                except Exception as e:
                    print(f"Error caching HDF5 image to disk: {e}")
            else:
                print(f"Warning: Image for ID {isic_id} not found in folder or HDF5. Using placeholder.")
                image_array = np.ones((224, 224, 3), dtype=np.uint8) * 255

        if self.use_shades_of_gray:
            image_array = shades_of_gray(image_array)

        if self.transforms is not None:
            image = self.transforms(image=image_array)["image"]
        else:
            image = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0

        return {
            "image": image,
            "tabular": torch.from_numpy(self.tabular[index]),
            "isic_id": isic_id,
        }


def get_available_presets():
    """Detect available trained model configurations from the outputs folder."""
    outputs_roots = [
        Path(__file__).parent / "models",
        Path(__file__).parent / "backbone_image",
        Path(__file__).parent / "fusion_models",
        Path(__file__).parent / "results" / "mid_fusion_lite_outputs",
        Path(__file__).parent / "mid_fusion_lite_outputs",
        Path(__file__).parent.parent / "mid_fusion_lite" / "mid_fusion_lite_outputs",
        Path(__file__).parent.parent / "mid_fusion_lite_outputs",
        Path(__file__).parent.parent / "mid_fusion_lite"
    ]
    presets = []
    
    seen_folders = set()
    
    # Check both outputs roots themselves (e.g. if files are directly inside) and subdirectories
    for outputs_root in outputs_roots:
        if not outputs_root.exists():
            continue
            
        # Check the root directory itself
        root_pts = list(outputs_root.glob("*.pt"))
        if root_pts and outputs_root.name not in seen_folders:
            seen_folders.add(outputs_root.name)
            preset_name = outputs_root.name
            metrics_json = outputs_root / "metrics.json"
            cv_pauc = None
            if metrics_json.exists():
                try:
                    with open(metrics_json, "r") as f:
                        metrics = json.load(f)
                        preset_name = metrics.get("preset", preset_name)
                        cv_pauc = metrics.get("cv_pauc", None)
                except Exception:
                    pass
            presets.append({
                "folder_name": outputs_root.name, # e.g. "mid_fusion_lite_outputs"
                "preset_name": preset_name,
                "checkpoints": [p.name for p in root_pts],
                "cv_pauc": cv_pauc
            })
            
        # Check subdirectories
        for subfolder in outputs_root.iterdir():
            if subfolder.is_dir() and subfolder.name not in seen_folders:
                # Check if there are .pt checkpoints in this folder
                pts = list(subfolder.glob("*.pt"))
                if pts:
                    seen_folders.add(subfolder.name)
                    # Try to load preset name from metrics.json
                    preset_name = subfolder.name
                    metrics_json = subfolder / "metrics.json"
                    cv_pauc = None
                    if metrics_json.exists():
                        try:
                            with open(metrics_json, "r") as f:
                                metrics = json.load(f)
                                preset_name = metrics.get("preset", preset_name)
                                cv_pauc = metrics.get("cv_pauc", None)
                        except Exception:
                            pass
                    presets.append({
                        "folder_name": subfolder.name,
                        "preset_name": preset_name,
                        "checkpoints": [p.name for p in pts],
                        "cv_pauc": cv_pauc
                    })
    return presets


@app.get("/")
async def serve_spa():
    index_path = BASE_DIR.parent / "frontend" / "dist" / "index.html"
    response = FileResponse(index_path)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "fastapi-backend",
        "model_ready": True,
        "model_version": "mid_fusion_lite",
    }

@app.get("/api/models")
def list_models():
    """Endpoint to list all trained model checkpoints available for prediction."""
    presets = get_available_presets()
    return {"models": presets}

@app.post("/api/predict")
async def run_prediction(
    csv_file: Optional[UploadFile] = File(None),
    image_files: List[UploadFile] = File(None),
    zip_file: UploadFile = File(None),
    hdf5_file: UploadFile = File(None),
    model_folder: str = Form("stage3_fusion")
):
    """
    Run pipeline prediction:
    1. Save uploaded CSV and images (extracted from zip, individual files or HDF5)
    2. Read configuration of selected model
    3. Run feature engineering and pre-fit tabular preprocessing
    4. Predict with ensemble of checkpoints
    5. Clean up temporary files
    6. Return predictions
    """
    session_id = str(uuid.uuid4())
    session_dir = UPLOADS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    images_dir = session_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    is_image_only = model_folder == "backbone_image"
    
    # Validation
    if not is_image_only and (csv_file is None or csv_file.filename == ""):
        raise HTTPException(status_code=400, detail="Mô hình Fusion yêu cầu tải lên file metadata CSV.")
        
    if image_files is None and zip_file is None and hdf5_file is None:
        raise HTTPException(status_code=400, detail="Vui lòng tải lên hình ảnh lâm sàng (ZIP/HDF5/Ảnh).")
            
    try:
        # Save CSV if not image-only
        if not is_image_only and csv_file is not None and csv_file.filename != "":
            csv_path = session_dir / "input_metadata.csv"
            with open(csv_path, "wb") as f:
                f.write(await csv_file.read())
            
        # Save HDF5 if provided
        hdf5_path = None
        if hdf5_file is not None and hdf5_file.filename != "":
            hdf5_path = session_dir / "images.hdf5"
            with open(hdf5_path, "wb") as f:
                f.write(await hdf5_file.read())
            
        # Save Images from ZIP if provided
        if zip_file is not None and zip_file.filename != "":
            zip_path = session_dir / "images.zip"
            with open(zip_path, "wb") as f:
                f.write(await zip_file.read())
            
            # Extract ZIP
            with zipfile.ZipFile(zip_path, "r") as zf:
                for file_info in zf.infolist():
                    # Avoid directory traversal and hidden files
                    if "/" in file_info.filename or "\\" in file_info.filename or file_info.filename.startswith("."):
                        filename = os.path.basename(file_info.filename)
                        if not filename:
                            continue
                    else:
                        filename = file_info.filename
                    
                    target_path = images_dir / filename
                    with zf.open(file_info) as source, open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target)
                        
        # Save individual images if provided
        if image_files is not None:
            for img in image_files:
                if img.filename != "":
                    # Save image
                    img_path = images_dir / img.filename
                    with open(img_path, "wb") as f:
                        f.write(await img.read())

        if is_image_only:
            isic_ids = []
            for p in images_dir.iterdir():
                if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    isic_ids.append(p.stem)
            if hdf5_path and hdf5_path.exists():
                try:
                    with h5py.File(hdf5_path, "r") as h5:
                        isic_ids.extend(list(h5.keys()))
                except Exception as e:
                     print(f"Error reading keys from HDF5: {e}")
            isic_ids = sorted(list(set(isic_ids)))
            if not isic_ids:
                raise HTTPException(status_code=400, detail="Không tìm thấy hình ảnh nào hợp lệ để dự đoán.")
            test_df = pd.DataFrame({
                "isic_id": isic_ids,
                "patient_id": ["patient_unknown"] * len(isic_ids)
            })
        else:
            # Load CSV into pandas
            try:
                test_df = pd.read_csv(csv_path)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Cannot parse CSV file: {e}")

            # Check required primary columns
            if "isic_id" not in test_df.columns:
                raise HTTPException(status_code=400, detail="CSV must contain 'isic_id' column.")
            
        # Determine model path
        outputs_roots = [
            Path(__file__).parent / "models",
            Path(__file__).parent / "backbone_image",
            Path(__file__).parent / "fusion_models",
            Path(__file__).parent / "results" / "mid_fusion_lite_outputs",
            Path(__file__).parent / "mid_fusion_lite_outputs",
            Path(__file__).parent.parent / "mid_fusion_lite_outputs",
            Path(__file__).parent.parent / "mid_fusion_lite" / "mid_fusion_lite_outputs",
            Path(__file__).parent.parent / "mid_fusion_lite"
        ]
        
        selected_model_path = None
        for root in outputs_roots:
            candidate = root if model_folder == "mid_fusion_lite_outputs" else root / model_folder
            if candidate.exists() and list(candidate.glob("*.pt")):
                selected_model_path = candidate
                break
                
        if selected_model_path is None:
            # Fallback to the first one just in case
            selected_model_path = outputs_roots[0] if model_folder == "mid_fusion_lite_outputs" else outputs_roots[0] / model_folder
            
        if not selected_model_path.exists():
            # Support absolute paths or exact names if somehow passed
            if Path(model_folder).exists():
                selected_model_path = Path(model_folder)
            else:
                raise HTTPException(status_code=404, detail=f"Model folder '{model_folder}' not found.")
            
        # Read checkpoint files
        checkpoints = list(selected_model_path.glob("*.pt"))
        if not checkpoints:
            raise HTTPException(status_code=404, detail=f"No checkpoints (.pt) found in model folder '{model_folder}'.")
            
        # Read metrics to resolve preset details
        preset_name = "smoke" # default fallback
        n_splits = 2
        fold_indices = [0]
        metrics_json = selected_model_path / "metrics.json"
        
        if metrics_json.exists():
            try:
                with open(metrics_json, "r") as f:
                    metrics = json.load(f)
                    preset_name = metrics.get("preset", "smoke")
                    # Map preset to n_splits
                    if "smoke" in preset_name:
                        preset_name = "smoke"
                        n_splits = 2
                        fold_indices = [0]
                    elif "quick" in preset_name:
                        preset_name = "quick"
                        n_splits = 3
                        fold_indices = [0, 1, 2]
                    elif "full_lite" in preset_name:
                        preset_name = "full_lite"
                        n_splits = 5
                        fold_indices = [0, 1, 2, 3, 4]
                    elif "kaggle_top1_fusion" in preset_name:
                        preset_name = "kaggle_top1_fusion"
                        n_splits = 5
                        fold_indices = [0, 1, 2, 3, 4]
                    elif "image_only_train" in preset_name:
                        preset_name = "image_only_train"
                        n_splits = 5
                        fold_indices = [0, 1, 2, 3, 4]
                    else:
                        print(f"Unknown preset {preset_name}, falling back to full_lite")
                        preset_name = "full_lite"
                        n_splits = 5
                        fold_indices = [0, 1, 2, 3, 4]
            except Exception as e:
                print(f"Error loading metrics.json: {e}")
                
        # Load the configuration of this preset
        try:
            config = get_preset_config(preset_name)
        except Exception:
            config = get_preset_config("smoke")

        # Override configurations to match the trained checkpoints
        if preset_name == "full_lite":
            config = config.with_updates(
                image_size=256,
                tabm_k=16,
                tabm_n_blocks=3,
                tabm_d_block=256,
                tab_dropout=0.15,
                backbone_name="efficientnet_b0",
                tab_hidden_dim=192,
                fusion_hidden_dim=320,
            )
            
        if not is_image_only:
            # Read reference columns from training metadata to align schema
            # We need this to ensure the preprocessor receives exactly the same columns!
            train_metadata_path = Path(__file__).parent.parent.parent / "data" / "train-metadata.csv"
            if not train_metadata_path.exists():
                # Fallback to local tests folder if main data folder is not available
                train_metadata_path = Path(__file__).parent.parent / "tests" / "test-metadata.csv"
                
            if train_metadata_path.exists():
                # Load only 1 row to get column names
                train_header_df = pd.read_csv(train_metadata_path, nrows=1)
                # Add missing columns with NaN to uploaded test_df
                for col in train_header_df.columns:
                    if col not in test_df.columns and col != config.target_col:
                        test_df[col] = np.nan
            else:
                print("Warning: Reference train-metadata.csv not found to align test schema.")

            # Ensure patient_id is present, generate dummy if missing
            if config.group_col not in test_df.columns:
                test_df[config.group_col] = "patient_unknown"
                
            # Fill missing key columns with appropriate defaults
            if "tbp_lv_nevi_confidence" not in test_df.columns:
                test_df["tbp_lv_nevi_confidence"] = 50.0

            # Perform feature engineering
            print("Running Feature Engineering on uploaded metadata...")
            # Make a dummy train_df because add_feature_engineering needs both
            dummy_train_df = test_df.copy()
            if config.target_col not in dummy_train_df.columns:
                dummy_train_df[config.target_col] = 0
                
            _, test_df_engineered = add_feature_engineering(
                train_df=dummy_train_df,
                test_df=test_df,
                group_col=config.group_col,
                categorical_cols=config.categorical_cols,
            )
        else:
            test_df_engineered = test_df.copy()

        # Get device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running prediction on device: {device}")
        
        # Load or create preprocessors
        if not is_image_only:
            preprocessors = load_or_create_preprocessors(preset_name, n_splits, fold_indices)
        else:
            preprocessors = {}
        
        # Run inference for each checkpoint
        test_pred_sum = np.zeros(len(test_df), dtype=np.float32)
        n_ensembles = 0
        
        transforms = get_image_transforms(config.image_size)["valid"]
        
        for cp in checkpoints:
            # Parse fold index from file name e.g. mid_fusion_lite_fold_0.pt
            try:
                fold_idx = int(cp.stem.split("_")[-1])
            except Exception:
                fold_idx = 0 # fallback
                
            if not is_image_only:
                if fold_idx not in preprocessors:
                    print(f"Skipping checkpoint {cp.name} because preprocessor for fold {fold_idx} is missing.")
                    continue
                    
                print(f"Running inference for checkpoint: {cp.name}")
                preprocessor = preprocessors[fold_idx]
                
                # Preprocess tabular data
                x_test = preprocessor.transform(test_df_engineered)
                bins = getattr(preprocessor, "bins", None)
                cat_cardinalities = list(preprocessor.cat_cardinalities) if config.use_tabm_tabular else None
            else:
                print(f"Running inference for checkpoint (Image Only): {cp.name}")
                x_test = np.zeros((len(test_df_engineered), 1), dtype=np.float32)
                bins = None
                cat_cardinalities = None
            
            # Create Dataset and DataLoader
            test_dataset = FolderImageTabularDataset(
                ids=test_df_engineered[config.id_col].astype(str).to_numpy(),
                tabular=x_test,
                image_folder=images_dir,
                hdf5_path=hdf5_path,
                transforms=transforms,
                use_shades_of_gray=getattr(config, "use_shades_of_gray", False)
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=config.valid_batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False
            )

            # Instantiate Model
            model = MidFusionLiteModel(
                backbone_name=config.backbone_name,
                n_tab_features=x_test.shape[1],
                use_tabm_tabular=config.use_tabm_tabular,
                tab_hidden_dim=config.tab_hidden_dim,
                tabm_arch_type=config.tabm_arch_type,
                tabm_k=config.tabm_k,
                tabm_n_blocks=config.tabm_n_blocks,
                tabm_d_block=config.tabm_d_block,
                fusion_hidden_dim=config.fusion_hidden_dim,
                fusion_mode=config.fusion_mode,
                cross_attention_heads=config.cross_attention_heads,
                cross_attention_dropout=config.cross_attention_dropout,
                tab_dropout=config.tab_dropout,
                fusion_dropout=config.fusion_dropout,
                train_image_backbone=config.train_image_backbone,
                download_pretrained_backbone=False,
                modality_dropout_prob=config.modality_dropout_prob,
                modality_mode="image_only" if is_image_only else "both",
                bins=bins,
                d_embedding=config.d_embedding,
                cat_cardinalities=cat_cardinalities,
            )
            
            # Load state dict
            state_dict = torch.load(cp, map_location=device)
            # Patch state_dict for backward compatibility with older checkpoints
            if "cross_modal_fusion.image_bottleneck.weight" in state_dict:
                state_dict["cross_modal_fusion.tab_bottleneck.weight"] = state_dict.pop("cross_modal_fusion.image_bottleneck.weight")
                state_dict["cross_modal_fusion.tab_bottleneck.bias"] = state_dict.pop("cross_modal_fusion.image_bottleneck.bias")
            
            # Initialize lazy parameters (needed before loading state_dict)
            model.eval()
            with torch.no_grad():
                dummy_images = torch.zeros(2, 3, config.image_size, config.image_size)
                dummy_tabular = torch.zeros(2, x_test.shape[1])
                _ = model(dummy_images, dummy_tabular)
                
            model.load_state_dict(state_dict, strict=False)
            model = model.to(device)
            
            # Run predict
            fold_preds = predict_loader(
                model=model,
                loader=test_loader,
                device=device,
                use_amp=False,
                progress_name=f"api_predict_fold_{fold_idx}",
                log_interval=0
            )
            
            test_pred_sum += fold_preds
            n_ensembles += 1
            
        if n_ensembles == 0:
            raise HTTPException(status_code=500, detail="Failed to run inference on any checkpoint.")
            
        # Calculate ensemble average
        test_pred = test_pred_sum / float(n_ensembles)
        
        # Post-processing feature was removed, use raw predictions
        test_pred_processed = test_pred.copy()
            
        # Copy matching images to session static directory for browser preview and generate Grad-CAM
        image_web_paths = {}
        cam_web_paths = {}
        from gradcam_utils import generate_gradcam_overlay
        
        for i, isic_id in enumerate(test_df[config.id_col]):
            copied = False
            for ext in [".jpg", ".jpeg", ".png"]:
                src = images_dir / f"{isic_id}{ext}"
                if src.exists():
                    web_dest_dir = STATIC_DIR / "session_images" / session_id
                    web_dest_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Copy original image
                    dest = web_dest_dir / f"{isic_id}{ext}"
                    shutil.copy2(src, dest)
                    image_web_paths[isic_id] = f"/static/session_images/{session_id}/{isic_id}{ext}"
                    
                    # Generate Grad-CAM heatmaps
                    try:
                        from PIL import Image
                        original_pil = Image.open(src).convert("RGB")
                        
                        # Get tensors from dataset
                        sample = test_dataset[i]
                        img_tensor = sample["image"].unsqueeze(0).to(device)
                        tab_tensor = sample["tabular"].unsqueeze(0).to(device)
                        
                        # Generate overlay
                        cam_pil = generate_gradcam_overlay(
                            model=model,
                            image_tensor=img_tensor,
                            tabular_tensor=tab_tensor,
                            original_img_pil=original_pil,
                            alpha=0.4,
                            is_image_only=is_image_only
                        )
                        cam_dest = web_dest_dir / f"{isic_id}_cam{ext}"
                        cam_pil.save(cam_dest)
                        cam_web_paths[isic_id] = f"/static/session_images/{session_id}/{isic_id}_cam{ext}"
                    except Exception as e:
                        print(f"Grad-CAM generation failed for {isic_id}: {e}")
                        cam_web_paths[isic_id] = None
                        
                    copied = True
                    break
            if not copied:
                image_web_paths[isic_id] = "/static/images/placeholder.png"
                cam_web_paths[isic_id] = None
                
        # Build response payload
        results = []
        
        for i, row in test_df.iterrows():
            isic_id = str(row["isic_id"])
            prob = float(test_pred_processed[i])
            
            # Save to history DB
            risk_level = "low"
            risk_text = "Nguy cơ thấp / theo dõi định kỳ"
            if prob >= 0.560058:
                risk_level = "high"
                risk_text = "Nguy cơ cao / đề nghị sinh thiết ngay"
            elif prob >= 0.395996:
                risk_level = "medium"
                risk_text = "Nguy cơ trung bình / nên kiểm tra thêm"
                
            PREDICTION_HISTORY.insert(0, {
                "prediction_id": str(uuid.uuid4()),
                "patient_code": str(row.get("patient_id", "Unknown")),
                "image_name": f"{isic_id}.jpg",
                "probability_malignant": prob,
                "risk_level": risk_level,
                "risk_text": risk_text,
                "created_at": datetime.now().isoformat()
            })

            results.append({
                "isic_id": isic_id,
                "patient_id": str(row.get("patient_id", "Unknown")),
                "age_approx": float(row.get("age_approx", 0.0)) if not pd.isna(row.get("age_approx")) else None,
                "sex": str(row.get("sex", "Unknown")) if not pd.isna(row.get("sex")) else None,
                "anatom_site_general": str(row.get("anatom_site_general", "Unknown")) if not pd.isna(row.get("anatom_site_general")) else None,
                "clin_size_long_diam_mm": float(row.get("clin_size_long_diam_mm", 0.0)) if not pd.isna(row.get("clin_size_long_diam_mm")) else None,
                "probability_raw": float(test_pred[i]),
                "probability_processed": prob,
                "risk_level": risk_level,
                "risk_text": risk_text,
                "image_url": image_web_paths.get(isic_id, "/static/images/placeholder.png"),
                "cam_image_url": cam_web_paths.get(isic_id, None)
            })
            
        # Clean up temporary folders EXCEPT session_images that frontend needs
        try:
            shutil.rmtree(str(images_dir))
            if (session_dir / "images.zip").exists():
                (session_dir / "images.zip").unlink()
        except Exception as e:
            print(f"Cleanup warning: {e}")
            
        return JSONResponse({
            "status": "success",
            "results": results,
            "session_id": session_id
        })
        
    except HTTPException as he:
        # Re-raise HTTP exceptions
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.post("/api/initialize")
def initialize_preset(preset_name: str = Form("smoke"), n_splits: int = Form(2)):
    """Pre-fit and cache the preprocessors for a given preset so inference is fast."""
    try:
        if "smoke" in preset_name:
            preset_name = "smoke"
            fold_indices = [0]
        elif "quick" in preset_name:
            fold_indices = [0, 1, 2]
        elif "full_lite" in preset_name:
            fold_indices = [0, 1, 2, 3, 4]
        else:
            fold_indices = [0]
            
        load_or_create_preprocessors(preset_name, n_splits, fold_indices)
        return {"status": "success", "message": f"Preprocessors for preset '{preset_name}' initialized successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Initialization failed: {e}")

@app.get("/predictions")
def get_prediction_history():
    """Return the mock in-memory database of previous predictions."""
    return {
        "items": PREDICTION_HISTORY,
        "total": len(PREDICTION_HISTORY)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
