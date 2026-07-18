# Mid Fusion Lite — ISIC 2024 Multimodal Classification

`mid_fusion_lite` là pipeline phân loại tổn thương da kết hợp hai modality:

- Ảnh dermoscopy được mã hóa bằng EfficientNet.
- Metadata lâm sàng và đặc trưng được thiết kế được mã hóa bằng TabM.
- Hai nhánh được kết hợp bằng concatenation, GMU hoặc cross-attention.

Pipeline hỗ trợ huấn luyện độc lập từng nhánh, chỉ huấn luyện Stage 3 từ checkpoint có sẵn, huấn luyện progressive ba stage và huấn luyện end-to-end. Metric lựa chọn mô hình là partial AUC trong miền TPR ≥ 0.80, phù hợp với ISIC 2024.

> Đây là mã nghiên cứu/thực nghiệm. Kết quả không phải chẩn đoán y khoa và chưa được xác nhận bằng external validation.

## 1. Kiến trúc

```text
Ảnh HDF5
   │
   ▼
EfficientNet-B0 ── spatial image tokens ──────────────┐
                                                     │
Metadata CSV                                         ▼
   │                                         Multimodal fusion
   ├── feature engineering                           │
   ├── fold-safe preprocessing                       ▼
   └── TabM ensemble ── tabular representation ── Fusion head
                                                     │
                                                     ▼
                                              Xác suất target
```

### Nhánh ảnh

- Backbone mặc định: `efficientnet_b0`.
- Ảnh được đọc từ HDF5 theo `isic_id`.
- Training augmentation gồm biến đổi hình học nhẹ, màu sắc, blur/noise và coarse dropout.
- Validation chỉ resize và normalize.
- Khi dùng cross-attention, feature map không được global-pool trước fusion mà được chuyển thành chuỗi spatial token.

### Nhánh bảng

- Encoder mặc định: TabM với ensemble `K=32`.
- Numerical features sử dụng piecewise-linear embeddings.
- Categorical features được mã hóa thành integer code; code `0` dành cho unknown/unseen.
- Feature engineering bao gồm các nhóm ABCD, patient normalization, anatomical context, area aggregation, ugly-duckling và LOF.
- Các thống kê toàn cục được fit trên fold-train rồi áp dụng cho fold-valid/test để hạn chế leakage.

Thứ tự feature là một phần của checkpoint TabM. Stage 3 phải dùng đúng cùng tập cột, cùng thứ tự và cùng preprocessing như lúc huấn luyện nhánh tabular.

## 2. Cross-attention hiện tại

Với image tokens `I ∈ R^(B×L×D)` và một tabular token `t ∈ R^(B×1×D)`, model tính hai hướng:

```text
Tabular query  ──attend──> L image keys/values
L image query ──attend──> 1 tabular key/value
```

Chiều tabular-query-image là attention đúng nghĩa:

```text
tab_ctx = Attention(query=tab_token, key=image_tokens, value=image_tokens)
```

Tabular representation có thể phân bố trọng số trên các vùng ảnh khác nhau. Với TabM ensemble, mỗi thành viên `K` có thể tạo một cách tổng hợp image tokens khác nhau.

Chiều image-query-tabular chỉ có một key/value:

```text
image_ctx = Attention(query=image_tokens, key=tab_token, value=tab_token)
```

Vì `softmax` trên một key luôn bằng `1`, chiều này không thực hiện lựa chọn attention. Nó hoạt động gần với phép chiếu rồi broadcast tabular context vào mọi image token. Cơ chế hiện tại vì vậy nên được hiểu là **asymmetric cross-modal attention**, không phải co-attention hai chiều hoàn chỉnh.

Hai representation sau residual được ghép lại và đưa vào fusion head:

```text
fused = concat(mean_pool(image_out), tab_out)
```

### Giới hạn và hướng phát triển

- Nếu ưu tiên phần cứng: thay chiều image-query-single-tab-token bằng FiLM và giữ chiều tabular-query-image. Đây là hướng phù hợp nhất với TabM `K=32`.
- Nếu ưu tiên nghiên cứu kiến trúc: chia metadata thành nhiều token theo nhóm như demographic, anatomy, color, geometry và patient context. Khi đó co-attention hai chiều mới có nhiều key để lựa chọn.
- Multi-token attention chưa được triển khai trong phiên bản hiện tại. Với `M` tabular token, chi phí attention tăng gần tuyến tính theo `M`; nên bắt đầu với `M=4` và giảm TabM xuống `K=8` hoặc `K=16` nếu VRAM hạn chế.
- Model hiện không trả attention weights (`need_weights=False`), vì vậy chưa hỗ trợ trực tiếp việc trực quan hóa attention map.

## 3. Modality dropout

`modality_dropout_prob` là xác suất tổng để một sample fusion bị bỏ đúng một modality:

- `p / 2`: bỏ ảnh, giữ bảng.
- `p / 2`: giữ ảnh, bỏ bảng.
- `1 - p`: giữ cả hai.
- Không bao giờ bỏ đồng thời cả hai modality.

Với `modality_dropout_prob=0.15`, phân phối kỳ vọng là:

```text
85.0%  image + tabular
 7.5%  image only
 7.5%  tabular only
```

Mask được lấy theo patient/sample và dùng chung cho toàn bộ `K` thành viên TabM. Mask chỉ áp dụng trong fusion training; evaluation và các mode unimodal không sử dụng modality dropout. Residual và cross-modal context đều được mask để bias của Linear/MHA không tái tạo tín hiệu giả cho modality bị thiếu.

## 4. Các chế độ chạy

Chỉ nên dùng `run_mode` làm selector công khai:

| `run_mode` | Hành vi |
|---|---|
| `tabular_only` | Chỉ chạy nhánh bảng, tương ứng Stage 1 |
| `image_only` | Chỉ chạy nhánh ảnh, tương ứng Stage 2 |
| `fusion` | Bắt buộc nạp checkpoint hai nhánh và chỉ chạy Stage 3 |
| `progressive` | Nạp checkpoint nếu có; nếu thiếu thì huấn luyện Stage 1/2 trước Stage 3 |
| `end_to_end` | Khởi tạo full model và huấn luyện hai nhánh cùng fusion từ đầu |

`modality_mode`, `use_staged_pretraining` và `stage_to_run` chỉ được giữ để tương thích cấu hình cũ. Không nên dùng chúng trong notebook hoặc config mới.

### Stage 3

Stage 3 gồm hai pha:

1. **Stage 3A**: đóng băng image encoder và tabular encoder, huấn luyện các projection/fusion layer.
2. **Stage 3B**: mở tabular encoder và, nếu `train_image_backbone=True`, mở image encoder để joint fine-tuning.

Stage 3B hỗ trợ differential learning rate:

```text
image backbone LR = stage3b_lr
tabular/fusion LR = stage3b_lr × stage3b_head_lr_mult
```

Nếu Stage 3A đạt pAUC cao hơn Stage 3B, pipeline trả checkpoint tốt nhất của Stage 3A thay vì checkpoint cuối.

## 5. Cấu trúc dữ liệu

Các file đầu vào bắt buộc:

```text
data/
├── train-metadata.csv
├── test-metadata.csv
├── sample_submission.csv
├── train-image.hdf5
└── test-image.hdf5
```

Các cột tối thiểu trong train metadata:

- `isic_id`: khóa ảnh trong HDF5.
- `target`: nhãn nhị phân.
- `patient_id`: khóa group để chia fold.

Nếu metadata không có cột `fold`, runner tạo fold bằng `StratifiedGroupKFold`, stratify theo `target` và group theo `patient_id`. Seed mặc định là `42`.

Có thể truyền đường dẫn tuyệt đối qua `MidFusionLiteConfig` thay vì dùng thư mục `data/`:

```python
config = MidFusionLiteConfig(
    train_metadata_path="/path/train-metadata.csv",
    test_metadata_path="/path/test-metadata.csv",
    sample_submission_path="/path/sample_submission.csv",
    train_hdf5_path="/path/train-image.hdf5",
    test_hdf5_path="/path/test-image.hdf5",
)
```

## 6. Cài đặt

Các dependency chính:

```text
torch
timm
tabm
rtdl-num-embeddings
numpy
pandas
scikit-learn
h5py
pillow
albumentations
```

Ví dụ cài đặt:

```bash
pip install torch timm tabm rtdl-num-embeddings numpy pandas scikit-learn h5py pillow albumentations
```

Nên sử dụng CUDA cho image/fusion training. CPU chỉ phù hợp với smoke test hoặc kiểm tra pipeline. Phiên bản thư viện dùng để inference nên khớp với môi trường huấn luyện checkpoint.

## 7. Chạy notebook

Ba notebook chính:

- `Notebook/tabular-only.ipynb`: huấn luyện checkpoint nhánh bảng.
- `Notebook/image-only.ipynb`: huấn luyện checkpoint nhánh ảnh.
- `Notebook/fusion.ipynb`: nạp hai bộ checkpoint và chạy Stage 3.

Quy trình khuyến nghị:

```text
tabular-only.ipynb ──> tabm_fold_0.pt ... tabm_fold_4.pt ──┐
                                                           ├──> fusion.ipynb
image-only.ipynb  ──> image_only_fold_0.pt ... fold_4.pt ──┘
```

Trước khi chạy `fusion.ipynb`, cần sửa:

```python
pretrained_tabular_dir = "/kaggle/input/.../tabular-checkpoints"
pretrained_image_dir = "/kaggle/input/.../image-checkpoints"
output_dir = "/kaggle/working/fusion_output"
```

Không sử dụng output cũ được lưu sẵn trong notebook để báo cáo kết quả sau khi source/config đã thay đổi. Hãy clear output, restart kernel và run all để bảo đảm source và log cùng một phiên bản thí nghiệm.

## 8. Chạy bằng Python

### Tabular-only

```python
from backend.mid_fusion_lite import MidFusionLiteConfig, run_mid_fusion_cv

config = MidFusionLiteConfig(
    run_mode="tabular_only",
    n_splits=5,
    stage1_epochs=30,
    output_dir="./outputs/tabular",
)

results = run_mid_fusion_cv(config)
```

### Image-only

```python
from backend.mid_fusion_lite import MidFusionLiteConfig, run_mid_fusion_cv

config = MidFusionLiteConfig(
    run_mode="image_only",
    n_splits=5,
    image_size=224,
    stage2_epochs=20,
    stage2_train_backbone=True,
    output_dir="./outputs/image",
)

results = run_mid_fusion_cv(config)
```

### Fusion từ hai bộ checkpoint

```python
from backend.mid_fusion_lite import MidFusionLiteConfig, run_mid_fusion_cv
from backend.mid_fusion_lite.config import build_ablation_columns

numeric_cols, categorical_cols = build_ablation_columns(
    use_engineered_abcd=True,
    use_patient_norm=True,
    use_ugly_duckling_adv=True,
    use_categorical_context=True,
)

config = MidFusionLiteConfig(
    run_mode="fusion",
    n_splits=5,
    image_size=224,
    backbone_name="efficientnet_b0",
    numeric_cols=numeric_cols,
    categorical_cols=categorical_cols,
    fusion_mode="cross_attention",
    fusion_hidden_dim=256,
    cross_attention_heads=4,
    modality_dropout_prob=0.15,
    pretrained_tabular_dir="./checkpoints/tabular",
    pretrained_image_dir="./checkpoints/image",
    stage3_freeze_epochs=3,
    stage3_unfreeze_epochs=5,
    stage3a_lr=1e-4,
    stage3b_lr=1e-5,
    stage3b_head_lr_mult=5.0,
    output_dir="./outputs/fusion",
)

results = run_mid_fusion_cv(config)
print(results["cv_pauc"])
```

Trong `run_mode="fusion"`, checkpoint thiếu sẽ gây lỗi ngay. Runner không tự huấn luyện lại nhánh bị thiếu và không ghi checkpoint vào thư mục input chỉ đọc của Kaggle.

### Chạy nhanh bằng CLI

```bash
python backend/mid_fusion_lite/runner.py \
  --preset smoke \
  --run-mode end_to_end \
  --folds 0 \
  --output-dir ./outputs/smoke
```

CLI phù hợp cho smoke test và các override đơn giản. Với thí nghiệm fusion chính thức cần khai báo đầy đủ feature schema và checkpoint, nên dùng Python config hoặc `Notebook/fusion.ipynb` để config được lưu rõ ràng cùng thí nghiệm.

## 9. Checkpoint theo fold

Tên file khuyến nghị, đánh số fold từ `0`:

```text
tabular/
├── tabm_fold_0.pt
├── tabm_fold_1.pt
├── tabm_fold_2.pt
├── tabm_fold_3.pt
└── tabm_fold_4.pt

image/
├── image_only_fold_0.pt
├── image_only_fold_1.pt
├── image_only_fold_2.pt
├── image_only_fold_3.pt
└── image_only_fold_4.pt
```

Runner cũng hỗ trợ:

- `mid_fusion_lite_fold_{fold}.pt`.
- Template chứa `{fold}` cho index bắt đầu từ 0.
- Template chứa `{fold_1}` cho index bắt đầu từ 1.
- Tìm file trong các thư mục con của Kaggle Dataset.

Ví dụ:

```python
pretrained_tabular_dir="/weights/tabm_fold_{fold}.pt"
pretrained_image_dir="/weights/image_fold_{fold_1}.pt"
```

Hai nhánh phải được huấn luyện với cùng fold definition. Cùng số fold nhưng khác seed hoặc khác cách chia patient không được xem là checkpoint tương ứng.

`pretrained_image_dir` yêu cầu checkpoint đầy đủ của nhánh image-only. Nếu chỉ có raw timm backbone weights, hãy dùng `backbone_weights_path`.

## 10. Xử lý mất cân bằng và metric

- Training có thể dùng dynamic negative undersampling theo `negative_ratio`.
- Sampler lấy tập negative mới ở mỗi epoch nhưng vẫn tái lập từ seed.
- Classifier bias được khởi tạo theo tỷ lệ âm/dương đã cấu hình.
- `pos_weight` tự động dùng căn bậc hai của tỷ lệ âm/dương; có thể override qua config.
- Early stopping và checkpoint selection dựa trên validation pAUC.
- TabM ensemble tính loss riêng trên `K` logits và inference bằng trung bình `K` xác suất.

Pipeline lưu cả trung bình pAUC theo fold và overall OOF pAUC. Vì pAUC là metric phi tuyến, hai giá trị không bắt buộc bằng nhau. Overall OOF pAUC là con số phù hợp hơn khi báo cáo chất lượng CV tổng thể.

## 11. Output

Mỗi lần chạy tạo:

```text
output_dir/
├── mid_fusion_lite_fold_0.pt
├── ...
├── oof_predictions.csv
├── submission_mid_fusion_lite.csv
└── metrics.json
```

`results` trả về:

```python
results["cv_pauc"]
results["fold_metrics"]
results["oof_df"]
results["submission_df"]
results["oof_path"]
results["submission_path"]
results["metrics_path"]
```

Runner hiện không trả `cv_auc`; chỉ AUROC theo epoch được ghi trong training log. Không nên in `results["cv_auc"]` nếu chưa bổ sung metric này vào runner.

## 12. Cảnh báo thường gặp

### `ShiftScaleRotate is a special case of Affine`

Đây là cảnh báo deprecation từ Albumentations, không làm dừng huấn luyện. Có thể chuyển sang `A.Affine` trong bản cập nhật sau.

### `feature has just two bin edges`

Feature đó gần như hằng trong fold nên piecewise-linear encoding chỉ tạo một bin hữu ích. Pipeline vẫn chạy, nhưng nên kiểm tra variance và cân nhắc loại feature suy biến trong ablation.

### `missing=N, unexpected=0` khi load branch

Checkpoint branch chỉ chứa một phần của full fusion model nên log có thể báo nhiều missing key thuộc nhánh còn lại hoặc fusion head. Tuy nhiên cần xác nhận checkpoint được tạo từ đúng kiến trúc và feature schema; `strict=False` không thay thế kiểm tra tương thích thí nghiệm.

### Kaggle báo `Read-only file system`

Không ghi checkpoint vào `/kaggle/input`. Đặt `output_dir` dưới `/kaggle/working`. Với `run_mode="fusion"`, hai checkpoint phải tồn tại trước khi chạy.

## 13. Tái lập và báo cáo khoa học

Để so sánh image-only, tabular-only và fusion một cách hợp lệ:

1. Dùng cùng `seed`, `n_splits`, patient groups và fold indices.
2. Không thay đổi feature schema giữa Stage 1 và Stage 3.
3. Dùng cùng preprocessing/augmentation cho checkpoint ảnh và fusion.
4. Lưu config, phiên bản code và đường dẫn/hash checkpoint cùng kết quả.
5. So sánh trên cùng OOF rows.
6. Chạy nhiều seed hoặc bootstrap OOF để ước lượng khoảng tin cậy.
7. Thực hiện ablation giữa concat, GMU, one-way attention, cross-attention và modality dropout.

Một cấu hình phức tạp hoặc nhiều feature không tự động trở thành bằng chứng khoa học. Kết luận về đóng góp của cross-attention hay modality dropout chỉ nên được đưa ra sau ablation có kiểm soát.

## 14. Kiểm thử

Chạy từ workspace root:

```bash
python -m unittest discover -s backend/mid_fusion_lite/tests -v
```

Test hiện bao phủ:

- Phân phối và semantics của modality dropout.
- Mask dùng chung cho TabM ensemble.
- Không rò tín hiệu qua Linear/MHA bias khi một modality bị thiếu.
- Config tối thiểu cho image-only và tabular-only.
- Routing `run_mode` và tương thích config cũ.
- Tìm checkpoint theo tên fold.
- Stage 2 trả đúng history và score của chính nó.

## 15. Cấu trúc mã nguồn

```text
mid_fusion_lite/
├── config.py                    # Dataclass config, preset, feature schema
├── data.py                      # Dataset HDF5, augmentation, sampler
├── model.py                     # EfficientNet, TabM, GMU, cross-attention
├── train.py                     # Train/evaluate/predict, AUROC và pAUC
├── runner.py                    # CV orchestration và staged training
├── tabular_prep_01.py           # Chuẩn bị numerical base
├── tabular_engineering_02.py    # Feature engineering theo fold
├── tabular_selection_03.py      # Null-importance selection
├── tabular_preprocessor_04.py   # Scaling/encoding/quantile transform
├── run_stage3_kaggle.py         # Entry point Stage 3 trên Kaggle
├── Notebook/
│   ├── tabular-only.ipynb
│   ├── image-only.ipynb
│   └── fusion.ipynb
└── tests/
    └── test_modality_dropout.py
```
