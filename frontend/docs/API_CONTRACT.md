# API Contract – ISIC Skin Cancer Demo

Repo frontend chỉ phụ trách UI/UX và gọi API. Backend, database và model inference triển khai riêng.

Base URL:

```text
http://localhost:8000/api/v1
```

Frontend đọc từ:

```env
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

---

## 1. Luồng UX đã chốt

Giao diện Predict có 2 vùng chính:

```text
Bên trái:
- Một ô upload duy nhất cho metadata JSON hoặc ảnh QR
- Hiển thị thông tin bệnh nhân read-only
- Hiển thị metadata ảnh read-only

Bên phải:
- Một ô upload ảnh tổn thương hoặc file HDF5
- Khi chưa upload: ô hiển thị nút upload
- Khi đã upload: chính ô đó hiển thị tên file + nút X
- Nếu là ảnh thông dụng: preview ảnh trong khung cố định
- Nếu là HDF5: chỉ hiển thị tên file, không preview
- Nút prediction và kết quả nằm cùng một hàng
```

Bác sĩ không nhập tay hồ sơ bệnh nhân hoặc metadata quan trọng để tránh sai lệch dữ liệu.

---

## 2. Upload metadata JSON/QR

Frontend dùng cùng một ô upload cho metadata JSON hoặc ảnh QR:

```text
.json
.png
.jpg
.jpeg
```

Nguyên tắc chung:

```text
Thiết bị/hệ thống quang học xuất metadata JSON hoặc QR
→ frontend upload file lên backend
→ backend resolve patient/metadata
→ frontend chỉ hiển thị dữ liệu read-only
```

Frontend không tự fill thông tin bệnh nhân từ JSON để tránh sai lệch dữ liệu.

Nếu là `.json`, frontend gọi:

```text
POST /metadata/json/resolve
```

Nếu là ảnh QR, frontend gọi:

```text
POST /metadata/qr/decode
```

Backend xử lý JSON/QR, tra cứu hoặc lưu thông tin bệnh nhân trong DB, chuẩn hóa metadata và trả dữ liệu về frontend.

---

## 3. Metadata JSON resolve

### `POST /metadata/json/resolve`

Content-Type:

```text
multipart/form-data
```

### Request fields

| Field           | Type | Required | Mô tả                                            |
| --------------- | ---: | -------: | ------------------------------------------------ |
| `metadata_file` | File |      Yes | File `.json` xuất từ thiết bị/hệ thống quang học |

### JSON metadata mẫu

Metadata demo dựa trên file `test-metadata.csv`.

```json
{
  "metadata_mode": "processed",
  "source": "optical_device",
  "patient_code": "IP_6074337",
  "isic_id": "ISIC_0015657",
  "metadata": {
    "isic_id": "ISIC_0015657",
    "patient_id": "IP_6074337",
    "age_approx": 45,
    "sex": "male",
    "anatom_site_general": "posterior torso",
    "clin_size_long_diam_mm": 2.7,
    "image_type": "TBP tile: close-up",
    "tbp_tile_type": "3D: XP",
    "tbp_lv_areaMM2": 3.846876,
    "tbp_lv_perimeterMM": 9.387248,
    "tbp_lv_location": "Torso Back Top Third",
    "tbp_lv_location_simple": "Torso Back",
    "attribution": "Memorial Sloan Kettering Cancer Center",
    "copyright_license": "CC-BY"
  }
}
```

### Response

```json
{
  "payload": {
    "metadata_mode": "processed",
    "source": "optical_device",
    "patient_code": "IP_6074337",
    "isic_id": "ISIC_0015657"
  },
  "patient": {
    "patient_code": "IP_6074337",
    "patient_name": "Demo Patient 6074337",
    "contact": "0900000000",
    "note": "Demo patient loaded from backend DB"
  },
  "resolved_metadata": {
    "metadata_mode": "processed",
    "source": "optical_device",
    "isic_id": "ISIC_0015657",
    "patient_id": "IP_6074337",
    "age_approx": 45,
    "sex": "male",
    "anatom_site_general": "posterior torso",
    "clin_size_long_diam_mm": 2.7,
    "image_type": "TBP tile: close-up",
    "tbp_tile_type": "3D: XP",
    "tbp_lv_areaMM2": 3.846876,
    "tbp_lv_perimeterMM": 9.387248,
    "tbp_lv_location": "Torso Back Top Third",
    "tbp_lv_location_simple": "Torso Back",
    "attribution": "Memorial Sloan Kettering Cancer Center",
    "copyright_license": "CC-BY"
  }
}
```

Backend là nơi quyết định thông tin bệnh nhân cuối cùng. Frontend chỉ nhận response và hiển thị read-only.

---

## 4. QR metadata

QR không được tạo thủ công trên frontend. Thiết bị/hệ thống quang học sẽ sinh QR.

QR nên chứa mã tham chiếu hoặc thông tin định danh ngắn, không nên chứa full metadata quá dài.

### Payload QR khuyến nghị

```json
{
  "metadata_mode": "lookup",
  "source": "device_or_hospital_system",
  "patient_code": "IP_6074337",
  "isic_id": "ISIC_0015657",
  "metadata_ref": "META_ISIC_0015657"
}
```

Backend decode QR, sau đó tra cứu hoặc chuẩn hóa metadata và trả về cùng format với JSON resolve.

---

## 5. Health Check

### `GET /health`

### Response

```json
{
  "status": "ok",
  "service": "isic-skin-backend",
  "model_ready": true,
  "model_version": "mobilenetv3-fusion-v1",
  "threshold": {
    "screening_threshold": 0.1791,
    "high_risk_threshold": 0.5,
    "target_recall": 0.95
  }
}
```

---

## 6. Prediction

### `POST /predictions`

Content-Type:

```text
multipart/form-data
```

### Request fields

| Field           |        Type | Required | Mô tả                                                                                                                |
| --------------- | ----------: | -------: | -------------------------------------------------------------------------------------------------------------------- |
| `image`         |        File |      Yes | Ảnh hoặc file test model. Hỗ trợ `.jpg`, `.jpeg`, `.png`, `.webp`, `.h5`, `.hdf5`. File gốc, không resize ở frontend |
| `patient`       | JSON string |       No | Thông tin bệnh nhân do backend resolve từ metadata/DB                                                                |
| `metadata`      | JSON string |      Yes | Metadata đã được backend resolve từ JSON/QR                                                                          |
| `metadata_mode` |      string |       No | `raw`, `processed`, `lookup`                                                                                         |
| `source`        |      string |       No | `json`, `qr`, `lookup`                                                                                               |

Nếu file là `.h5` hoặc `.hdf5`, frontend không preview ảnh. Backend/model tự xử lý nội dung HDF5.

### Response

```json
{
  "prediction_id": "pred_20260627_001",
  "model_version": "mobilenetv3-fusion-v1",
  "mode": "real_model",
  "probability_malignant": 0.8872,
  "screening_threshold": 0.1791,
  "high_risk_threshold": 0.5,
  "risk_level": "high",
  "risk_text": "Nguy cơ cao / cần kiểm tra thêm",
  "recommendation": "Nên được bác sĩ/chuyên gia kiểm tra thêm.",
  "metrics_context": {
    "holdout_auc": 0.9163,
    "holdout_average_precision": 0.7551,
    "holdout_recall_at_screening_threshold": 0.9667,
    "holdout_precision_at_screening_threshold": 0.2843,
    "holdout_false_negative": 1,
    "holdout_false_positive": 73
  },
  "metadata_used": {
    "isic_id": "ISIC_0015657",
    "metadata_mode": "processed"
  },
  "patient": {
    "patient_code": "IP_6074337",
    "patient_name": "Demo Patient 6074337",
    "contact": "0900000000"
  },
  "created_at": "2026-06-27T10:30:00Z"
}
```

---

## 7. QR decode

### `POST /metadata/qr/decode`

Content-Type:

```text
multipart/form-data
```

### Request fields

| Field      | Type | Required | Mô tả                                         |
| ---------- | ---: | -------: | --------------------------------------------- |
| `qr_image` | File |      Yes | Ảnh QR do thiết bị/hệ thống quang học xuất ra |

### Response

```json
{
  "payload": {
    "metadata_mode": "lookup",
    "source": "device_or_hospital_system",
    "patient_code": "IP_6074337",
    "isic_id": "ISIC_0015657",
    "metadata_ref": "META_ISIC_0015657"
  },
  "patient": {
    "patient_code": "IP_6074337",
    "patient_name": "Demo Patient 6074337",
    "contact": "0900000000",
    "note": "Demo patient loaded from backend DB"
  },
  "resolved_metadata": {
    "metadata_mode": "processed",
    "source": "optical_device",
    "isic_id": "ISIC_0015657",
    "patient_id": "IP_6074337",
    "age_approx": 45,
    "sex": "male",
    "anatom_site_general": "posterior torso",
    "clin_size_long_diam_mm": 2.7
  }
}
```

---

## 8. Patient lookup

### `GET /patients/{patient_code}`

Endpoint này dùng khi cần tra cứu thông tin bệnh nhân theo mã.

Trong flow chính, frontend không bắt buộc gọi endpoint này trực tiếp vì backend đã trả `patient` trong response của `/metadata/json/resolve` hoặc `/metadata/qr/decode`.

### Response

```json
{
  "patient_code": "IP_6074337",
  "patient_name": "Demo Patient 6074337",
  "contact": "0900000000",
  "note": "Demo patient loaded from backend DB"
}
```

---

## 9. Metadata lookup

### `GET /metadata/{isic_id}?mode=processed`

Hoặc lookup bằng `metadata_ref`:

### `GET /metadata/by-ref/{metadata_ref}`

Trong flow chính, frontend không bắt buộc gọi endpoint này trực tiếp vì backend đã trả `resolved_metadata` trong response của `/metadata/json/resolve` hoặc `/metadata/qr/decode`.

### Response

```json
{
  "isic_id": "ISIC_0015657",
  "mode": "processed",
  "patient": {
    "patient_code": "IP_6074337",
    "patient_name": "Demo Patient 6074337",
    "contact": "0900000000"
  },
  "metadata": {
    "metadata_mode": "processed",
    "source": "optical_device",
    "isic_id": "ISIC_0015657",
    "patient_id": "IP_6074337",
    "age_approx": 45,
    "sex": "male",
    "anatom_site_general": "posterior torso",
    "clin_size_long_diam_mm": 2.7
  }
}
```

---

## 10. Error response

Backend nên trả lỗi theo format thống nhất:

```json
{
  "detail": "Invalid metadata JSON file"
}
```

Các mã lỗi tối thiểu:

| Status | Trường hợp                                                   |
| -----: | ------------------------------------------------------------ |
|  `400` | File không hợp lệ, sai định dạng, JSON parse lỗi             |
|  `404` | Không tìm thấy `patient_code`, `isic_id` hoặc `metadata_ref` |
|  `422` | Input không đủ điều kiện để model inference                  |
|  `500` | Lỗi backend/model inference                                  |

---

## 11. Không dùng các cột leakage

Backend không được dùng các cột sau trong inference:

```text
target
iddx_full
iddx_1
iddx_2
iddx_3
iddx_4
iddx_5
mel_mitotic_index
mel_thick_mm
tbp_lv_dnn_lesion_confidence
```

Các cột này chỉ dùng cho phân tích dữ liệu hoặc đánh giá offline, không đưa vào input model khi inference.

---

## 12. Disclaimer

Ứng dụng chỉ phục vụ demo học thuật và hỗ trợ sàng lọc. Kết quả model không được dùng như chẩn đoán y khoa cuối cùng.