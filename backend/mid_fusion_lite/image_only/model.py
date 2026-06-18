import timm
import torch
import torch.nn as nn
from pathlib import Path

class StandaloneImageModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        use_pretrained: bool = False,
        weights_path: str | None = None,
    ) -> None:
        super().__init__()
        
        create_kwargs = {
            "model_name": backbone_name,
            "num_classes": 1,  # Binary classification
            "global_pool": "avg",
        }
        
        self.model = timm.create_model(pretrained=False, **create_kwargs)
        
        if weights_path:
            self._load_local_weights(weights_path)
        elif use_pretrained:
            try:
                self.model = timm.create_model(pretrained=True, **create_kwargs)
            except Exception as e:
                print(f"Warning: Failed to load pretrained timm weights: {e}. Falling back to random initialization.")
                
    def _load_local_weights(self, weights_path: str) -> None:
        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(f"Image model weights file not found: {weights_path}")
            
        raw_obj = torch.load(path, map_location="cpu")
        state_dict = raw_obj
        if isinstance(raw_obj, dict):
            if "state_dict" in raw_obj:
                state_dict = raw_obj["state_dict"]
            elif "model" in raw_obj:
                state_dict = raw_obj["model"]
                
        # Strip prefixes if model was trained as part of mid_fusion_lite
        normalized: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("image_encoder."):
                key = key[len("image_encoder."):]
            if key.startswith("model.image_encoder."):
                key = key[len("model.image_encoder."):]
            normalized[key] = value
            
        # Try loading state dict
        missing, unexpected = self.model.load_state_dict(normalized, strict=False)
        print(f"Loaded local weights from {weights_path} | missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.model(images)
        if logits.ndim > 1:
            logits = logits.squeeze(1)
        return logits
