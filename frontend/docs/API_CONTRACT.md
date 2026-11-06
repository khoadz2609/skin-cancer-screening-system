# API contract hiện tại

Frontend trong repo này gọi trực tiếp backend FastAPI tại:

```text
http://localhost:8000
```

Đặt trong `.env`:

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_ENABLE_MOCK=false
```

## Health check

`GET /health`

```json
{
  "status": "ok",
  "service": "fastapi-backend",
  "model_ready": true,
  "model_version": "mid_fusion_lite"
}
```

`model_ready` chỉ là `true` khi backend tìm thấy checkpoint `.pt` trong
`backend/models` hoặc `backend/backbone_image`.

## Danh sách model

`GET /api/models`

Trả về các thư mục model có checkpoint, ví dụ:

```json
{
  "models": [
    {
      "folder_name": "models",
      "preset_name": "kaggle_top1_fusion",
      "checkpoints": ["mid_fusion_lite_fold_0.pt"],
      "cv_pauc": 0.1753
    }
  ]
}
```

## Dự đoán hàng loạt

`POST /api/predict` với `multipart/form-data`.

| Field | Bắt buộc | Mô tả |
|---|---:|---|
| `model_folder` | Có | Tên thư mục model do `/api/models` trả về. Không truyền đường dẫn tuyệt đối. |
| `csv_file` | Fusion | CSV phải có cột `isic_id`; nên có `patient_id` và metadata tương ứng. |
| `image_files` | Một trong các field ảnh | Một hoặc nhiều ảnh `.jpg`, `.jpeg`, `.png`, `.webp`. |
| `zip_file` | Một trong các field ảnh | ZIP chứa các file ảnh. Đường dẫn con bị làm phẳng và chỉ nhận file ảnh. |
| `hdf5_file` | Một trong các field ảnh | HDF5 chứa dataset ảnh theo `isic_id`. |

Giới hạn hiện tại: mỗi file tối đa 100 MB, tổng nội dung giải nén ZIP tối đa
250 MB.

Response thành công:

```json
{
  "status": "success",
  "session_id": "uuid",
  "results": [
    {
      "isic_id": "ISIC_0015657",
      "probability_raw": 0.42,
      "probability_processed": 0.42,
      "risk_level": "medium",
      "risk_text": "...",
      "image_url": "/static/session_images/<uuid>/ISIC_0015657.jpg",
      "cam_image_url": "/static/session_images/<uuid>/ISIC_0015657_cam.jpg"
    }
  ]
}
```

Preview và Grad-CAM trong `static/session_images` được tự động xóa sau 7 ngày.

Ngưỡng rủi ro dùng chung:

- `low`: dưới `0.395996`
- `medium`: từ `0.395996` đến dưới `0.560058`
- `high`: từ `0.560058`

## Khởi tạo preprocessor

`POST /api/initialize`

Form fields:

- `preset_name`: `smoke`, `quick` hoặc `full_lite`
- `n_splits`: số fold tương ứng

## Lịch sử dự đoán

`GET /predictions`

Trả về tối đa 1.000 kết quả gần nhất trong bộ nhớ tiến trình. Dữ liệu này
không phải database bền vững và sẽ mất khi backend khởi động lại.

## Phạm vi hiện tại

Các endpoint metadata JSON/QR và `POST /predictions` được mô tả trong các bản
contract cũ chưa được backend hiện tại triển khai. Flow đang chạy của frontend
là upload CSV + ảnh/HDF5/ZIP qua `POST /api/predict`.

Ứng dụng chỉ phục vụ demo học thuật và không thay thế chẩn đoán y khoa.
