import timm
import torch

m = timm.create_model(
    "efficientnet_b0", pretrained=True, num_classes=0, global_pool="avg"
)
torch.save(m.state_dict(), "efficientnet_b0_backbone.pth")
