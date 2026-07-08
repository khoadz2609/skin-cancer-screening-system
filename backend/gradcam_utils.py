import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend for headless server support
from PIL import Image

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None
        self.handlers = []
        
        # Đăng ký hook thu thập đặc trưng và đạo hàm
        self.handlers.append(self.target_layer.register_forward_hook(self._save_features))
        self.handlers.append(self.target_layer.register_backward_hook(self._save_gradients))

    def _save_features(self, module, input, output):
        self.features = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        for handler in self.handlers:
            handler.remove()

    def __call__(self, image_tensor: torch.Tensor, tabular_tensor: torch.Tensor) -> np.ndarray:
        """Sinh bản đồ nhiệt Grad-CAM kích thước 2D.
        
        Args:
            image_tensor: Tensor ảnh đầu vào (1, 3, H, W)
            tabular_tensor: Tensor bảng lâm sàng (1, D)
            
        Returns:
            Ma trận Heatmap 2D chuẩn hóa [0, 1] dạng numpy array.
        """
        # Đảm bảo bật tính toán gradient
        with torch.enable_grad():
            # Set requires_grad=True cho tensor ảnh để bắt buộc PyTorch xây dựng computation graph qua backbone
            img_in = image_tensor.clone().detach()
            img_in.requires_grad = True
            
            tab_in = tabular_tensor.clone().detach()
            
            # Temporary override of train_image_backbone to allow gradients through backbone
            old_train_backbone = getattr(self.model, "train_image_backbone", False)
            self.model.train_image_backbone = True
            
            self.model.zero_grad()
            
            try:
                # Chạy forward pass
                output = self.model(img_in, tab_in)
                
                # Thực hiện backward pass trên giá trị logit dự đoán
                if isinstance(output, tuple):
                    output = output[0]
                if output.dim() > 0:
                    output = output.mean()
                output.backward()
            finally:
                # Restore original train_image_backbone state
                self.model.train_image_backbone = old_train_backbone
            
            if self.gradients is None or self.features is None:
                # Trả về ma trận 0 nếu không thu thập được đặc trưng (tránh crash)
                return np.zeros((14, 14), dtype=np.float32)
                
            # Tính trọng số trung bình toàn cục của gradients (Global Average Pooling)
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
            
            # Tính tổng có trọng số của các bản đồ đặc trưng
            cam = torch.sum(weights * self.features, dim=1).squeeze(0)  # (H_feat, W_feat)
            
            # Áp dụng ReLU để chỉ giữ lại các đặc trưng làm tăng nguy cơ ác tính
            cam = torch.clamp(cam, min=0)
            
            # Tránh lỗi chia cho 0 nếu cam toàn bộ là 0
            cam_max = cam.max()
            if cam_max > 0:
                cam = cam / cam_max
                
            return cam.cpu().numpy()


def find_last_conv_layer(model: nn.Module) -> nn.Module:
    """Tự động tìm kiếm lớp nn.Conv2d cuối cùng trong nhánh ảnh của mô hình."""
    last_conv = None
    # Nếu mô hình có image_encoder (như MidFusionLiteModel) thì tìm trong đó
    target_module = getattr(model, "image_encoder", model)
    
    for module in target_module.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
            
    if last_conv is None:
        raise RuntimeError("Không tìm thấy bất kỳ lớp Conv2d nào trong mô hình ảnh.")
    return last_conv


def generate_gradcam_overlay(
    model: nn.Module,
    image_tensor: torch.Tensor,
    tabular_tensor: torch.Tensor,
    original_img_pil: Image.Image,
    alpha: float = 0.5,
    colormap_name: str = "jet",
    is_image_only: bool = False
) -> Image.Image:
    """Tạo ảnh overlay kết hợp giữa ảnh gốc và heatmap Grad-CAM.
    
    Args:
        model: Mô hình PyTorch MidFusionLiteModel đã load weights.
        image_tensor: Tensor ảnh (1, 3, H, W)
        tabular_tensor: Tensor bảng (1, D)
        original_img_pil: Ảnh gốc dạng PIL Image.
        alpha: Độ trong suốt của heatmap (0.0 đến 1.0).
        colormap_name: Tên bản đồ màu của matplotlib ('jet', 'hot', 'viridis').
        
    Returns:
        PIL Image của ảnh gốc đã được đè heatmap.
    """
    model.eval()
    
    # 1. Tìm layer Conv cuối cùng
    target_layer = find_last_conv_layer(model)
    
    # 2. Khởi tạo GradCAM
    gradcam = GradCAM(model, target_layer)
    
    try:
        # 3. Sinh heatmap 2D
        heatmap2d = gradcam(image_tensor, tabular_tensor)
    finally:
        # Giải phóng hooks để tránh tràn bộ nhớ
        gradcam.remove_hooks()
        
    # 4. Phóng to heatmap khớp kích thước ảnh gốc
    width, height = original_img_pil.size
    
    # Sử dụng matplotlib để gán colormap sang hệ màu RGB (Hỗ trợ tương thích cả phiên bản cũ và mới)
    try:
        colormap = matplotlib.colormaps[colormap_name]
    except AttributeError:
        import matplotlib.cm as cm
        colormap = cm.get_cmap(colormap_name)
    heatmap_rgba = colormap(heatmap2d)  # Kết quả dạng float [0, 1] kích thước (H_feat, W_feat, 4)
    
    # Chuyển đổi sang PIL Image hệ màu RGBA [0, 255]
    heatmap_img = Image.fromarray((heatmap_rgba * 255).astype(np.uint8))
    heatmap_img = heatmap_img.resize((width, height), Image.BILINEAR)
    
    # 5. Thực hiện trộn (Blend) ảnh gốc với heatmap
    original_rgba = original_img_pil.convert("RGBA")
    
    # Chỉ áp dụng độ mờ alpha cho các kênh màu của heatmap, giữ nguyên độ đục của ảnh gốc
    overlay_img = Image.blend(original_rgba, heatmap_img, alpha=alpha)
    
    return overlay_img.convert("RGB")
