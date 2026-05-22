# 🏗️ Mid Fusion Lite - Fusion Architecture Report

**Status:** Detailed analysis of current fusion strategy  
**Date:** May 22, 2026

---

## ⚠️ **Short Answer: ĐÂY LÀ "INTERMEDIATE FUSION" (Mid-Fusion), KHÔNG PHẢI JOINT**

| Aspect | Intermediate/Mid-Fusion | Joint Fusion | Current Code |
|---|---|---|---|
| **Khi fusion xảy ra** | **Giữa quá trình** (sau encoding) | **Ở cuối** (concatenation trước head) | ✅ **INTERMEDIATE** |
| **Branches độc lập đến lúc** | Encoding xong → features | Đầu vào nguyên bản → features | ✅ **INTERMEDIATE** |
| **Fusion method** | MultiheadAttention / Concatenation | Simple concat | ✅ **INTERMEDIATE** |

---

## 🏗️ Current Architecture Flow

```
Input: Image (160×160) + Tabular (432 features)
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
   IMAGE BRANCH          TABULAR BRANCH
        ↓                       ↓
  ConvNeXt V2 Nano        TABM Encoder
  (backbone only)         (Transformer)
        ↓                       ↓
  image_head             tabular_encoder
  Linear→GELU→Dropout    features (B, tab_hidden_dim)
        ↓                       ↓
  image_features         tabular_features
  (B, fusion_hidden_dim) (B, tab_hidden_dim)
        ↓                       ↓
        └───────────┬───────────┘
                    ↓
        **MID-FUSION LAYER** ← Happens here!
        - Cross-Modal Attention
        - OR Simple Concat
                    ↓
            fused_features
            (B, 2*fusion_hidden_dim)
                    ↓
            fusion_head
            Linear→BN→GELU→Dropout→Linear
                    ↓
                 logits (B, 1)
```

---

## 📊 Detailed Architecture Breakdown

### **Branch 1: Image Branch**

```python
# model.py:144-160
self.image_encoder = timm.create_model(
    model_name="convnextv2_nano",
    num_classes=0,
    global_pool="avg"
)
# Output: (B, 768)  [ConvNeXt Nano output features]

self.image_head = nn.Sequential(
    nn.LazyLinear(fusion_hidden_dim),  # 768 → 256
    nn.GELU(),
    nn.Dropout(fusion_dropout)
)
# Output: (B, 256)
```

### **Branch 2: Tabular Branch**

```python
# model.py:162-177
if use_tabm_tabular:
    self.tabular_encoder = TabMTabularEncoder(
        n_num_features=432,
        d_out=tab_hidden_dim,  # 192
        arch_type="tabm",
        k=16,
        n_blocks=3,
        d_block=256
    )
else:
    self.tabular_encoder = TabularEncoder(MLP)

# Output: (B, 192)
```

### **Fusion Layer: "MID-FUSION"**

```python
# model.py:179-199
if fusion_mode == "cross_attention":
    # Project tabular to same dimension as image
    self.tab_to_fusion = nn.Linear(tab_hidden_dim, fusion_hidden_dim)
    # 192 → 256
    
    self.cross_modal_fusion = CrossModalAttentionFusion(
        hidden_dim=256,
        num_heads=4,
        dropout=0.1
    )
    # Output: (B, 256*2) = (B, 512)
else:
    # Simple concatenation
    fused = torch.cat([image_features, tabular_features], dim=1)
    # Output: (B, 256+192) = (B, 448)
```

### **Fusion Head: Classification**

```python
# model.py:205-211
self.fusion_head = nn.Sequential(
    nn.Linear(fusion_in_dim, fusion_hidden_dim),  # 512 → 256
    nn.BatchNorm1d(256),
    nn.GELU(),
    nn.Dropout(fusion_dropout),
    nn.Linear(256, 1)  # → logit
)
# Output: (B, 1)
```

---

## 🔄 Two Fusion Modes Available

### **Mode 1: Concatenation (Simple)**

```python
fusion_mode = "concat"

Flow:
Image (256) ──┐
              ├→ Concat ──→ (448) ──→ Fusion Head ──→ Logit
Tabular (192)─┘

Pros:
  ✅ Simple, fast
  ✅ All information preserved
  ✅ No cross-interaction

Cons:
  ❌ No inter-modal learning
  ❌ Less expressive
```

**Current Kaggle:** `fusion_mode = "cross_attention"` (NOT concat)

### **Mode 2: Cross-Modal Attention (Current)**

```python
fusion_mode = "cross_attention"

Flow:
Image (256) ──┐
              ├→ LayerNorm ──→ MultiheadAttention ──→ Image'
              │    ↓              ↑
              │  Attention    ↙───┴───→ Tabular'
              │    ↑           Cross-modal
Tabular (192)─┴→ Linear(→256) interaction

Result: (Image' || Tabular') = (512)
                    ↓
            Fusion Head ──→ Logit

Mechanism:
  Image attends to Tabular → learns tabular context
  Tabular attends to Image → learns image context
  Both enriched with cross-modal information

Pros:
  ✅ Inter-modal learning
  ✅ Richer representations
  ✅ Better capture of relationships

Cons:
  ⚠️ Slightly slower (MultiheadAttention overhead)
  ⚠️ More parameters
```

**Current Implementation: CrossModalAttentionFusion (model.py:78-119)**

```python
def forward(self, image_features, tab_features):
    # Normalize
    norm_image = self.image_norm(image_features)
    norm_tab = self.tab_norm(tab_features)
    
    # Cross-attention
    image_ctx, _ = self.image_to_tab(
        query=norm_image,      # Image attends to...
        key=norm_tab,          # ...Tabular
        value=norm_tab
    )
    tab_ctx, _ = self.tab_to_image(
        query=norm_tab,        # Tabular attends to...
        key=norm_image,        # ...Image
        value=norm_image
    )
    
    # Residual connection + Concatenate
    image_out = image_features + self.dropout(image_ctx)
    tab_out = tab_features + self.dropout(tab_ctx)
    return torch.cat([image_out, tab_out], dim=1)
```

---

## 📈 Fusion Types Comparison

| Type | Fusion Point | Architecture | Expressivity | Speed | When to use |
|---|---|---|---|---|---|
| **Early** | Input layer | Image + Tab raw data → Single network | Low | ⚡⚡⚡ Fast | Quick baseline |
| **INTERMEDIATE (Current)** | After encoding | Image encoder + Tab encoder → Fusion | **High** | ⚡⚡ Medium | **BEST** |
| **Late** | Output layer | Image head + Tab head → Avg/Max | Medium | ⚡ Slow | Ensemble-like |
| **Joint** | All layers | Shared representations throughout | High | ⚡ Slow | Complex datasets |

---

## ✅ Why Intermediate/Mid-Fusion is Correct

### **Reasons:**

1. **Separate processing branches**
   - Image: ConvNeXt V2 specializes in visual features
   - Tabular: TABM specializes in structured data
   - ✅ Each learns independently

2. **Fusion after sufficient encoding**
   - Not too early (raw pixel + feature mismatch)
   - Not too late (lost learning opportunities)
   - ✅ Balanced

3. **Cross-modal interaction**
   - Image can learn tabular context (size, age, location)
   - Tabular can learn image patterns (color, shape)
   - ✅ Rich fusion

4. **Interpretability**
   - Can analyze image branch separately
   - Can analyze tabular branch separately
   - Can see attention weights for fusion
   - ✅ Easier to debug

---

## 🎯 Current Configuration (Kaggle)

```python
{
    'fusion_mode': 'cross_attention',  # ← Intermediate Fusion
    'cross_attention_heads': 4,
    'cross_attention_dropout': 0.1,
    'fusion_hidden_dim': 256,
    'tab_hidden_dim': 192,
}
```

### **Flow Summary:**

```
Image Input (160×160)
    ↓
ConvNeXt V2 Nano backbone
    ↓
image_head: Linear(→256) + GELU
    ↓
image_features: (B, 256)
    ↓
           ┌────── MID-FUSION ──────┐
           ↓                        ↓
Cross-Modal Attention        Simple Concat
- image_to_tab               (Alternative)
- tab_to_image
           ↓
fused: (B, 512)  [with cross_attention]
           ↓
fusion_head: Linear + BN + GELU + Linear(→1)
           ↓
logits: (B, 1)
```

---

## 🔄 So Sánh: Intermediate vs Joint vs Late

### **Intermediate (Current) ✅**
```
Image Backbone → Image Encoder → [FUSION] → Logit
Tabular Backbone → Tabular Encoder →
```
- Branches separate → fuse mid-way
- **This is what you have**

### **Joint (Not Current)**
```
[Input: Image + Tabular] → Shared Network → Logit
```
- Fuse from start
- Would need different architecture

### **Late (Not Current)**
```
Image Encoder → Image Head → [FUSION] → Logit
Tabular Encoder → Tabular Head →
```
- Fuse after classification heads
- Less information sharing

---

## 📝 Summary

**Câu trả lời:** ✅ **YES, đó là INTERMEDIATE FUSION (Mid-Fusion)**

- ❌ **NOT Joint Fusion** (Joint would merge inputs at beginning)
- ✅ **YES Intermediate Fusion** (Fuse after each branch encodes)
- ✅ **YES Good choice** (balances expressivity vs complexity)

**Current implementation:**
- Image branch: ConvNeXt V2 Nano → 256 dims
- Tabular branch: TABM → 192 dims
- **Fusion layer: CrossModalAttention** (not simple concat)
- Classification: Linear head on fused (512) features

**Two modes available:**
1. `fusion_mode="cross_attention"` → Advanced (current)
2. `fusion_mode="concat"` → Simple alternative

---

## 🚀 Advanced Alternatives (Not Implemented)

Nếu muốn thử khác:

1. **Gating fusion** - Learn weighted combination
2. **Bilinear fusion** - Capture interactions
3. **Tensor fusion** - Higher-order interactions
4. **Self-attention** - Within-modal before cross-attention

Nhưng **intermediate + cross-attention (hiện tại) là solid choice!** ✅

