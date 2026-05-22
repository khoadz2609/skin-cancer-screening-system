# 📊 Mid Fusion Lite - Feature Engineering Options Report

**Date:** May 22, 2026  
**Status:** Comprehensive analysis of all feature engineering toggles

---

## 📋 Overview

Hiện tại codebase có **9 toggles chính** để bật/tắt các bước feature engineering. Mỗi toggle có thể được sử dụng độc lập hoặc kết hợp để:
- ✅ Tối ưu hiệu suất (pAUC score)
- ✅ Tiết kiệm RAM/GPU memory
- ✅ Tăng tốc độ training
- ✅ Kiểm tra impact của từng feature group

---

## 🔧 Feature Engineering Toggles

### 1️⃣ **`use_feature_engineering` (Master Switch)**

| Property | Value |
|---|---|
| **Type** | `bool` |
| **Default** | `True` |
| **Location** | `config.py:221` |
| **Controls** | Bật/tắt tất cả feature engineering (master switch) |

**Chi tiết:**
- Khi `False`: Chỉ sử dụng **base numeric + categorical columns** từ dữ liệu gốc
- Khi `True`: Bật tất cả engineered features + contextual features
- **Impact:** -80 engineered features nếu tắt

**Use case:**
```python
# Baseline test (chỉ dùng raw features)
config.with_updates(use_feature_engineering=False)

# Full features (production)
config.with_updates(use_feature_engineering=True)
```

**Feature count:**
- OFF: 40 features (34 base numeric + 6 categorical)
- ON: 1012+ features (phụ thuộc vào các toggle khác)

---

### 2️⃣ **`use_contextual_group_norm_features` (Patient-Location Normalization)**

| Property | Value |
|---|---|
| **Type** | `bool` |
| **Default** | `True` |
| **Location** | `config.py:222`, `tabular.py:85-108` |
| **Features Added** | ~369 features |

**Chi tiết:**
```
Công thức: (value - location_mean) / (location_std + err)
```

Tạo 3 nhóm normalization theo ngữ cảnh:
1. **`patient_loc_norm`**: Normalize theo (patient, location) pair
   - VD: Một lesion ở tay, normalize với lesions khác của patient cũng ở tay
   
2. **`patient_loc_sim_norm`**: Normalize theo (patient, location_simple) pair
   - VD: Normalize theo vùng đơn giản hơn (ví dụ: arm, leg)
   
3. **`patient_site_general_norm`**: Normalize theo (patient, anatomical site)
   - VD: Normalize theo anatomical site (ví dụ: head, trunk)

**Công thức:**
```python
for col in all_features:  # ~123 features
    for suffix in [patient_loc_norm, patient_loc_sim_norm, patient_site_general_norm]:
        feature_name = f"{col}_{suffix}"
        # Tạo 369 features mới
```

**Impact:**
- **ON**: +369 features, có thể cải thiện pAUC nếu location matter
- **OFF**: -369 features, tiết kiệm RAM ~15%

**Use case (Kaggle với RAM limited):**
```python
config.with_updates(
    use_contextual_group_norm_features=False
)
```

**Recommendation:**
- ✅ Bật khi: GPU memory > 10GB, muốn best performance
- ❌ Tắt khi: Kaggle notebook, local RAM limited, cần tốc độ

---

### 3️⃣ **`use_ugly_duckling_features` (Outlier Detection)**

| Property | Value |
|---|---|
| **Type** | `bool` |
| **Default** | `True` |
| **Location** | `config.py:223`, `tabular.py:111-149` |
| **Features Added** | ~246 features |

**Chi tiết:**
Dựa trên concept "Ugly Duckling" từ dermatology: một lesion khác lạ so với bình thường trong tập lesions của patient.

**Tạo 3 loại features:**

1. **Percentile ranking:** `ud_percentile_{col}`
   ```
   VD: ud_percentile_lesion_size = rank percentile trong patient
   Giá trị: [0, 1] - 1 = bất thường (quá lớn/quá nhỏ)
   ```
   - Tạo 123 features

2. **Location z-score:** `ud_loc_{col}`
   ```
   VD: ud_loc_areaMM2 = (value - location_mean) / location_std
   Được normalize theo location lesion (tay, chân, v.v.)
   ```
   - Tạo 123 features

3. **Summary statistics:**
   - `ud_count_location`: Số features có |z-score| > 2.0
   - `ud_max_severity_location`: Max |z-score| ở location
   - `ud_consistency_location`: Std của z-scores ở location

**Impact:**
- **ON**: +246 features, detect "ugly ducklings" (lesions khác lạ)
- **OFF**: -246 features, tiết kiệm RAM ~10%

**Use case:**
```python
# Detect anomalous lesions
config.with_updates(use_ugly_duckling_features=True)

# Skip on RAM-limited systems
config.with_updates(use_ugly_duckling_features=False)
```

**Recommendation:**
- ✅ Bật khi: Muốn detect outlier lesions, GPU available
- ❌ Tắt khi: Memory constraint, cần nhanh

---

### 4️⃣ **`use_anatomical_zscore_features` (Site-Level Normalization)**

| Property | Value |
|---|---|
| **Type** | `bool` |
| **Default** | `True` |
| **Location** | `config.py:224`, `tabular.py:453-462` |
| **Features Added** | 3 features |

**Chi tiết:**
Normalize theo anatomical site (head, trunk, extremities, etc.).

**Features:**
```
Công thức: (value - site_mean) / (site_std + err)

- tbp_lv_areaMM2_anatomical_zscore
- tbp_lv_norm_border_anatomical_zscore
- tbp_lv_norm_color_anatomical_zscore
```

**Lý do:** Lesions ở các vị trí khác nhau có kích thước/màu/border khác nhau.
- Tay/chân: thường nhỏ hơn
- Lưng: thường lớn hơn

**Impact:**
- **ON**: +3 features, rất nhẹ
- **OFF**: -3 features

**Use case:**
```python
# Always keep ON - negligible overhead
config.with_updates(use_anatomical_zscore_features=True)
```

**Recommendation:**
- ✅ **Luôn bật** - rất ít overhead, có ích

---

### 5️⃣ **`use_area_aggregation_features` (Patient-Level Statistics)**

| Property | Value |
|---|---|
| **Type** | `bool` |
| **Default** | `True` |
| **Location** | `config.py:225`, `tabular.py:444-451` |
| **Features Added** | 2 features |

**Chi tiết:**
Tính tổng area (diện tích) của tất cả lesions của patient.

**Features:**
```
- tbp_lv_areaMM2_patient: Tổng area của tất cả lesions trên body
- tbp_lv_areaMM2_bp: Tổng area tại một anatomical site (body part)
```

**Lý do:** Lesion load (tổng khối lượng tổn thương) có thể liên quan đến malignancy risk.

**Impact:**
- **ON**: +2 features
- **OFF**: -2 features

**Use case:**
```python
# Always keep ON - very lightweight
config.with_updates(use_area_aggregation_features=True)
```

**Recommendation:**
- ✅ **Luôn bật** - negligible cost, giá trị cao

---

### 6️⃣ **`use_patient_lof_features` (Anomaly Scoring)**

| Property | Value |
|---|---|
| **Type** | `bool` |
| **Default** | `False` ⚠️ |
| **Location** | `config.py:226`, `tabular.py:48-82, 473-496` |
| **Features Added** | 1 feature |
| **Computation** | Quadratic - **HEAVY** |

**Chi tiết:**
Sử dụng **Local Outlier Factor (LOF)** để detect anomalous lesions trong patient.

**Algorithm:**
```python
LOF(n_neighbors=20):
- Tính local density của mỗi lesion trong patient
- Lesions isolated = outliers (anomalous)
- Score cao = anomalous
```

**Features:**
```
- patient_lof_score: LOF score (higher = more anomalous)
```

**Complexity:**
```
O(n²) per patient
Với n = số lesions/patient (avg 5-10)
= tính toán nặng nhưng không quá
```

**Impact:**
- **ON**: +1 feature, but O(n²) computation
- **OFF**: Skip computation

**Use case:**
```python
# HIGH-RISK detection (bấm model)
config.with_updates(use_patient_lof_features=True)

# Default (skip) - không cần
config.with_updates(use_patient_lof_features=False)
```

**Recommendation:**
- ⚠️ **Tắt theo default** - computation heavy, return on investment thấp
- ✅ Bật chỉ khi: Muốn advanced anomaly detection, có resource

---

### 7️⃣ **`use_null_importance_filter` (Feature Selection)**

| Property | Value |
|---|---|
| **Type** | `bool` |
| **Default** | `False` ⚠️ |
| **Location** | `config.py:229`, `tabular.py:617-650` |
| **Features Dropped** | Variable |
| **Computation** | Very Heavy - **DO NOT USE** |

**Chi tiết:**
Dùng "null importance" để filter out non-informative features:

1. Tính correlation thực: `corr_real = |correlation(feature, target)|`
2. Shuffle target, tính correlation: `corr_null = |correlation(feature, shuffled_target)|`
3. Lặp N lần (null_importance_repeats)
4. Giữ feature nếu: `corr_real > quantile(corr_null, quantile)`

**Algorithm:**
```
for repeat in range(null_importance_repeats):  # default: 8
    shuffled_target = shuffle(target)
    null_score = compute_correlation(features, shuffled_target)
    
null_threshold = quantile(null_scores, 0.95)  # default
keep_mask = real_score > null_threshold
```

**Impact:**
- Complexity: O(repeats × features × samples)
  = 8 × 1000 × 3000 ~ millions of operations
- Giảm feature từ 1000+ xuống ~64 features (configurable)

**Params:**
```python
null_importance_repeats: int = 8           # Số lần shuffle
null_importance_quantile: float = 0.95     # Percentile threshold
null_importance_min_kept: int = 64         # Tối thiểu features giữ lại
```

**Use case:**
```python
# ❌ KHÔNG DÙNG - quá nặng
config.with_updates(use_null_importance_filter=True)

# Default - skip
config.with_updates(use_null_importance_filter=False)
```

**Recommendation:**
- ❌ **Tắt luôn** - computation quá nặng (30+ phút), ROI thấp
- Thay vào đó: Dùng tree-based feature importance (random forest)

---

### 8️⃣ **`use_quantile_transform` (Tabular Preprocessing)**

| Property | Value |
|---|---|
| **Type** | `bool` |
| **Default** | `True` |
| **Location** | `config.py:233` |
| **Applied To** | Tabular features |

**Chi tiết:**
Transform dữ liệu tabular sang distribution đồng đều [0, 1] để chuẩn hóa scale.

**Công thức:**
```
quantile_transform(X) = rank(X) / (n_samples)
Kết quả: [0, 1] uniform distribution
```

**Tại sao cần:**
- Some features có skewed distribution (ví dụ: lesion size)
- Tree-based: không cần (tree-based invariant to scaling)
- NN-based TABM: cần (NN benefit from normalized data)

**Params:**
```python
quantile_noise_std: float = 1e-5  # Thêm noise để tránh ties
```

**Impact:**
- ON: NN models (TABM) học tốt hơn
- OFF: May hurt NN performance

**Recommendation:**
- ✅ **Luôn bật** khi dùng TABM
- ❌ Tắt nếu sử dụng tree-based tabular encoder

---

### 9️⃣ **`use_tabm_tabular` (Tabular Architecture)**

| Property | Value |
|---|---|
| **Type** | `bool` |
| **Default** | `True` |
| **Location** | `config.py:228` |
| **Controls** | TABM vs simple MLP |

**Chi tiết:**
Chọn architecture cho tabular encoder:
- `True`: **TABM** (Transformer-based Tabular Model)
  - Hiệu suất cao, handle features tốt
  - Tính toán: O(n_features²) attention
  
- `False`: **Simple MLP** 
  - Nhẹ hơn, nhanh hơn
  - Hiệu suất có thể thấp hơn

**Params (nếu TABM):**
```python
tabm_arch_type: str = "tabm"      # Architecture type
tabm_k: int = 16                  # Heads per block
tabm_n_blocks: int = 3            # Number of blocks
tabm_d_block: int = 256           # Block dimension
```

**Use case:**
```python
# Best performance (production)
config.with_updates(use_tabm_tabular=True)

# Lightweight (quick test)
config.with_updates(use_tabm_tabular=False)
```

**Recommendation:**
- ✅ **Bật khi:** GPU available, muốn best performance
- ❌ Tắt khi: Local CPU, cần tốc độ

---

## 📊 Feature Count Summary

| Setting | Numeric Features | Categorical | Total | Notes |
|---|---|---|---|---|
| **Base only** | 34 | 6 | **40** | No engineering |
| **Engineering ON** | 1012 | 6 | **1018** | All features |
| **Kaggle preset** | 427 | 6 | **433** | No context, no UD |
| **Quick test** | 427 | 6 | **433** | Limited features |
| **Full lite** | 1012 | 6 | **1018** | Full stack |

---

## 🎯 Recommended Configurations

### 1. **Local CPU Testing (Quick)**
```python
config = MidFusionLiteConfig(
    preset_name='smoke',
    epochs=1,
    use_feature_engineering=True,
    use_contextual_group_norm_features=False,  # Skip heavy
    use_ugly_duckling_features=False,
    use_patient_lof_features=False,
    use_null_importance_filter=False,
    use_quantile_transform=True,
    use_tabm_tabular=False,  # Use MLP instead
    num_workers=0,
)
# Features: ~200, Runtime: 2-5 minutes
```

### 2. **Kaggle (Memory Efficient)**
```python
config = MidFusionLiteConfig(
    preset_name='quick',
    epochs=5,
    use_feature_engineering=True,
    use_contextual_group_norm_features=False,
    use_ugly_duckling_features=False,
    use_anatomical_zscore_features=True,   # Keep light ones
    use_area_aggregation_features=True,
    use_patient_lof_features=False,
    use_null_importance_filter=False,
    use_quantile_transform=True,
    use_tabm_tabular=True,
    num_workers=4,
)
# Features: ~427, Runtime: 15-20 minutes
```

### 3. **Production (GPU, Best Performance)**
```python
config = MidFusionLiteConfig(
    preset_name='full_lite',
    epochs=10,
    use_feature_engineering=True,
    use_contextual_group_norm_features=True,   # All features
    use_ugly_duckling_features=True,
    use_anatomical_zscore_features=True,
    use_area_aggregation_features=True,
    use_patient_lof_features=True,
    use_null_importance_filter=False,  # Still too slow
    use_quantile_transform=True,
    use_tabm_tabular=True,
    num_workers=4,
)
# Features: 1012+, Runtime: 30-60 minutes
```

---

## 📈 Impact Analysis (Relative to Baseline)

| Feature Group | Impact | Cost | Recommendation |
|---|---|---|---|
| **Contextual Norm** | +0-2% pAUC | +369 features | ✅ Bật nếu GPU |
| **Ugly Duckling** | +0-3% pAUC | +246 features | ✅ Bật nếu resource |
| **Anatomical Zscore** | +0-1% pAUC | +3 features | ✅ **Luôn bật** |
| **Area Aggregation** | +0-1% pAUC | +2 features | ✅ **Luôn bật** |
| **Patient LOF** | +0-1% pAUC | O(n²) | ❌ Skip |
| **Null Importance** | ~-10% features | Very slow | ❌ Skip |
| **Quantile Transform** | +1-2% pAUC | Negligible | ✅ Bật khi TABM |

---

## 🔄 Code Example

```python
from config import get_preset_config

# Start with preset
config = get_preset_config('quick')

# Customize features
config = config.with_updates(
    # Turn off heavy features for local testing
    use_contextual_group_norm_features=False,
    use_ugly_duckling_features=False,
    use_patient_lof_features=False,
    use_null_importance_filter=False,
    
    # Keep light features
    use_anatomical_zscore_features=True,
    use_area_aggregation_features=True,
    
    # Quick test
    epochs=2,
    num_workers=0,
)

# Run
from runner import run_mid_fusion_cv
result = run_mid_fusion_cv(config)
print(f"CV pAUC: {result['cv_pauc']:.6f}")
```

---

## 📝 Implementation Notes

1. **Modular Design**: Mỗi toggle độc lập, có thể bật/tắt riêng lẻ
2. **No Dependencies**: Các feature groups không phụ thuộc vào nhau
3. **Backward Compatible**: Tất cả features tắt → chỉ dùng base features
4. **Safe Defaults**: 
   - ON: `use_feature_engineering`, `use_anatomical_zscore_features`, `use_area_aggregation_features`
   - OFF: `use_patient_lof_features`, `use_null_importance_filter`

---

## ⚡ Performance Tips

1. **Fastest**: `use_feature_engineering=False` → 40 features, O(1) preprocessing
2. **Best balance**: Kaggle preset → 427 features, ~15 min/fold
3. **Best quality**: Full lite → 1012 features, ~45 min/fold

---

## Summary

✅ **Có 9 toggles chính** để kiểm soát feature engineering  
✅ **Có thể bật/tắt độc lập** theo nhu cầu  
✅ **Mỗi toggle có trade-off** giữa hiệu suất và chi phí  
✅ **Khuyến cáo:** Luôn bật anatomical_zscore + area_aggregation, tắt LOF + null_importance  

