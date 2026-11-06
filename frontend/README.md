# ISIC Skin Cancer UI

Repo frontend riêng cho đồ án **ISIC 2024 Skin Cancer Classification**.

Repo này chỉ phụ trách UI/UX và kết nối API. Backend, database, model inference và model training được triển khai ở repo riêng.

Ứng dụng hướng đến luồng hỗ trợ bác sĩ/nhân viên y tế sàng lọc tổn thương da. Ứng dụng không dành cho bệnh nhân tự sử dụng và không thay thế chẩn đoán y khoa.

---

## Luồng sử dụng chính

```text
1. Bác sĩ upload metadata JSON hoặc QR do thiết bị/hệ thống quang học xuất ra.
2. Frontend gửi metadata JSON/QR lên backend.
3. Backend đọc mã bệnh nhân, lưu hoặc tra cứu database, rồi trả về thông tin bệnh nhân và metadata đã chuẩn hóa.
4. Frontend hiển thị thông tin bệnh nhân ở dạng read-only.
5. Bác sĩ upload ảnh tổn thương hoặc file HDF5.
6. Frontend gửi ảnh/HDF5 kèm metadata đã resolve cho backend model.
7. Backend trả kết quả dự đoán để frontend hiển thị.
```

Bác sĩ không nhập tay thông tin bệnh nhân hoặc metadata quan trọng trong flow chính để tránh sai lệch dữ liệu.

---

## Chạy local

Cài dependencies:

```bash
npm install
```

Tạo file môi trường:

```bash
cp .env.example .env
```

Chạy development server:

```bash
npm run dev
```

Mở:

```text
http://localhost:5173
```

---

## Chạy Docker

```bash
docker compose up --build
```

Mở:

```text
http://localhost:5173
```

---

## Kết nối backend

Sửa file `.env`:

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_ENABLE_MOCK=false
```

Khi backend chưa sẵn sàng, có thể bật mock mode:

```env
VITE_ENABLE_MOCK=true
```

API contract gửi backend nằm ở:

```text
docs/API_CONTRACT.md
```

---

## Chức năng frontend

* Upload metadata JSON từ thiết bị/hệ thống quang học.
* Upload QR metadata từ thiết bị/hệ thống quang học.
* Gửi JSON/QR lên backend để resolve thông tin bệnh nhân và metadata.
* Hiển thị thông tin bệnh nhân ở dạng read-only.
* Hiển thị metadata ảnh ở dạng read-only.
* Upload ảnh tổn thương: `.jpg`, `.jpeg`, `.png`, `.webp`.
* Hỗ trợ file test model dạng `.h5`, `.hdf5`.
* Gửi ảnh/HDF5 kèm metadata đã resolve cho backend prediction API.
* Hiển thị kết quả dự đoán với 3 mức nguy cơ: `low`, `medium`, `high`.
* Có mock mode để test UI khi backend chưa hoàn thiện.

---

## Metadata mẫu

Metadata demo dựa trên file `test-metadata.csv`.

Một sample rút gọn:

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

README chỉ hiển thị sample rút gọn để dễ đọc. Khi chạy demo hoặc kết nối backend/model, metadata JSON có thể chứa đầy đủ các cột trong `test-metadata.csv`. Frontend sẽ giữ nguyên các field này trong payload gửi prediction.

Các mã demo trong file metadata test:

| isic_id      | patient_id | age_approx | sex    | anatom_site_general |
| ------------ | ---------- | ---------: | ------ | ------------------- |
| ISIC_0015657 | IP_6074337 |         45 | male   | posterior torso     |
| ISIC_0015729 | IP_1664139 |         35 | female | lower extremity     |
| ISIC_0015740 | IP_7142616 |         65 | male   | posterior torso     |

---

## Ghi chú về thông tin bệnh nhân

File metadata chỉ nên chứa mã bệnh nhân hoặc mã tham chiếu. Các thông tin như tên, liên hệ, ghi chú bệnh nhân nên được backend lấy từ database.

Flow chính:

```text
metadata JSON/QR → backend resolve patient/metadata → frontend hiển thị read-only
```

Frontend không cho bác sĩ nhập tay thông tin bệnh nhân trong flow chính.

---

## Ghi chú về HDF5

Frontend hỗ trợ upload file `.h5` hoặc `.hdf5` để phục vụ test model.

Với HDF5:

* Frontend không preview ảnh.
* Frontend chỉ hiển thị tên file.
* File gốc được gửi nguyên vẹn cho backend.
* Backend/model chịu trách nhiệm đọc và xử lý nội dung file.

---

## Build kiểm tra

```bash
npm run build
```

---

## Disclaimer

Ứng dụng chỉ phục vụ demo học thuật và hỗ trợ sàng lọc. Kết quả model không được dùng như chẩn đoán y khoa cuối cùng.
