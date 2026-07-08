import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'backend'))
import torch
from backend.mid_fusion_lite.model import MidFusionLiteModel
from backend.gradcam_utils import generate_gradcam_overlay
from PIL import Image

# Dummy kwargs
kwargs = {
    'modality_mode': 'both', 'backbone_name': 'efficientnet_b0', 'train_image_backbone': False, 
    'use_tabm_tabular': False, 'n_tab_features': 610, 'tab_hidden_dim': 128, 'tabm_arch_type': 'model1', 'tabm_k': 32, 
    'tabm_n_blocks': 2, 'tabm_d_block': 32, 'fusion_mode': 'cross_attention', 'fusion_hidden_dim': 256, 
    'cross_attention_heads': 8, 'cross_attention_dropout': 0.1, 'tab_dropout': 0.1, 'fusion_dropout': 0.1, 
    'modality_dropout_prob': 0.1, 'keep_tabm_ensemble': True, 'download_pretrained_backbone': False,
}

model = MidFusionLiteModel(**kwargs)
model.eval()

img = torch.randn(1, 3, 224, 224)
tab = torch.randn(1, 610)
pil_img = Image.new("RGB", (224, 224))

try:
    generate_gradcam_overlay(model, img, tab, pil_img)
    print("SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
