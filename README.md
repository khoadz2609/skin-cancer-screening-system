# Skin Cancer Screening & Diagnostic System (Web Demo)

Hệ thống hỗ trợ sàng lọc và dự đoán nguy cơ ung thư da sử dụng trí tuệ nhân tạo đa phương thức (**Multimodal Fusion** và **Vision Only**). Dự án tích hợp mô hình học sâu phân tích hình ảnh tổn thương da lâm sàng và dữ liệu bệnh nhân (metadata) để đưa ra tỉ lệ ác tính và phân cấp rủi ro.

---

## 📌 Tổng quan dự án

Hệ thống cung cấp 2 chế độ dự đoán chính:
1. **Mô hình Kết hợp (Multimodal Fusion):** Kết hợp cả ảnh chụp lâm sàng da và thông tin metadata của bệnh nhân (tuổi, giới tính, vị trí tổn thương, kích thước...) thông qua kiến trúc **EfficientNet-B0 + TabM (Tabular Encoder)** và cơ chế **Cross-Attention Fusion**.
2. **Mô hình chỉ dùng ảnh (Vision Only):** Phù hợp khi chỉ nhập ảnh đơn lẻ mà không có thông tin lâm sàng kèm theo. Chế độ này sử dụng Vision Backbone để dự đoán trực tiếp dựa trên đặc trưng hình ảnh.

Hệ thống còn tích hợp bản đồ nhiệt **Grad-CAM (Gradient-weighted Class Activation Mapping)** để giải thích trực quan (Explainable AI) vùng ảnh hưởng mà mô hình chú ý khi đưa ra quyết định.

---

## 🛠️ Kiến trúc công nghệ

* **Frontend:** React.js (TypeScript), Vite, TailwindCSS, Lucide Icons.
* **Backend:** FastAPI (Python), PyTorch, Timm, TabM, Scikit-learn, Pandas, H5Py.

---

## 📂 Cấu trúc thư mục dự án

```text
final_web_demo/
├── backend/                  # Mã nguồn FastAPI Web Backend
│   ├── backbone_image/       # Chứa các checkpoint mô hình chỉ dùng ảnh (.pt)
│   ├── models/               # Chứa các checkpoint mô hình kết hợp fusion (.pt)
│   ├── mid_fusion_lite/      # Module huấn luyện & định nghĩa mô hình (được tối ưu hóa)
│   ├── static/               # Lưu trữ ảnh tải lên và heatmap Grad-CAM tạm thời
│   ├── main.py               # API chính phục vụ dự đoán và lấy lịch sử
│   └── gradcam_utils.py      # Bộ công cụ sinh bản đồ nhiệt Grad-CAM
├── frontend/                 # Mã nguồn React Frontend
│   ├── src/
│   │   ├── api/              # Chứa các hàm giao tiếp http/endpoints
│   │   ├── features/         # Các trang giao diện (Prediction, History)
│   │   └── App.tsx           # Thành phần chính điều phối SPA
│   ├── package.json          # Quản lý thư viện Node.js
│   └── vite.config.ts        # Cấu hình build Vite
├── .gitignore                # Ngăn git theo dõi các file nặng/cache
├── requirements.txt          # Thư viện Python cần cài đặt
└── start_demo.bat            # Script khởi động nhanh cả Frontend và Backend (Windows)
```

---

## 📥 Hướng dẫn tải và cài đặt Trọng số mô hình (Weights)

Do các file trọng số mô hình `.pt` có kích thước lớn và không được khuyến khích đẩy trực tiếp lên GitHub, bạn cần tải chúng từ tập dữ liệu công khai trên Kaggle và đặt vào đúng vị trí:

1. **Trọng số cho mô hình Kết hợp (Multimodal Fusion - EfficientNet + TabM):**
   * Tải các file checkpoint từ Kaggle Dataset công khai: [fusion-backbone](https://www.kaggle.com/datasets/khoangdz/fusion-backbone)
   * Tạo thư mục `backend/models/` (nếu chưa có) và sao chép các file `.pt` này vào đó.

2. **Trọng số cho mô hình chỉ dùng ảnh (Vision Only - EfficientNet-B0):**
   * Tải các file checkpoint từ Kaggle Dataset công khai: [isic-2024-effb0-backbone](https://www.kaggle.com/datasets/khoangdz/isic-2024-effb0-backbone)
   * Tạo thư mục `backend/backbone_image/` (nếu chưa có) và sao chép các file `.pt` này vào đó.

*(Lưu ý: Đảm bảo trong mỗi thư mục đều có file `metrics.json` đi kèm tương ứng để backend nhận diện đúng preset).*

---

## 🚀 Hướng dẫn khởi chạy dự án

### Cách 1: Khởi chạy nhanh (Dành cho Windows)
Nếu bạn đang chạy trên hệ điều hành Windows, chỉ cần nhấp đúp vào file:
```bash
start_demo.bat
```
Script này sẽ tự động khởi tạo backend FastAPI (cổng `8000`) và frontend Vite (cổng `5173`) cùng lúc.

---

### Cách 2: Khởi chạy thủ công

#### Bước 1: Khởi động Backend
1. Di chuyển vào thư mục `backend`:
   ```bash
   cd backend
   ```
2. Cài đặt các thư viện cần thiết (khuyên dùng môi trường ảo venv):
   ```bash
   pip install -r ../requirements.txt
   ```
3. Chạy backend FastAPI:
   ```bash
   python main.py
   ```
   *Backend sẽ lắng nghe tại địa chỉ: `http://127.0.0.1:8000`*

#### Bước 2: Khởi động Frontend
1. Di chuyển vào thư mục `frontend`:
   ```bash
   cd frontend
   ```
2. Cài đặt các gói thư viện Node.js:
   ```bash
   npm install
   ```
3. Chạy môi trường phát triển (Dev server) hoặc biên dịch production:
   * Chạy dev: `npm run dev`
   * Biên dịch và cập nhật giao diện vào backend: `npm run build`
   *Frontend sẽ chạy tại địa chỉ: `http://localhost:5173`*

---

## 🔬 Minh họa sử dụng
1. Truy cập giao diện web tại `http://localhost:5173` (hoặc `http://127.0.0.1:8000` nếu đã chạy build).
2. Tại phần **Cấu hình mô hình dự đoán**:
   * Chọn **Mô hình kết hợp** để tải lên cả file metadata (CSV) và ảnh lâm sàng.
   * Chọn **Mô hình chỉ dùng ảnh** để chỉ cần kéo thả hình ảnh tổn thương da đơn lẻ mà không cần bất kỳ thông tin nào khác.
3. Bấm **Tiến hành Sàng lọc** để nhận tỉ lệ ác tính, mức độ cảnh báo và bản đồ nhiệt giải thích vị trí tổn thương da.

## Download and verify model artifacts

The checkpoints are intentionally kept outside Git because of their size. With the Kaggle CLI authenticated, download the two public datasets into the expected folders:

```bash
kaggle datasets download -d khoangdz/fusion-backbone -p backend/models --unzip
kaggle datasets download -d khoangdz/isic-2024-effb0-backbone -p backend/backbone_image --unzip
```

Verify the local files before starting the API:

```bash
python scripts/check_weights.py
python scripts/check_weights.py --require-preprocessors
```

The verifier checks all five folds, `metrics.json`, and (optionally) the cached fusion preprocessors. Checkpoints and caches are intentionally ignored by Git; the Kaggle datasets above are the official download sources for this demo.

Install the CLI if needed with `pip install kaggle`, then configure `~/.kaggle/kaggle.json` using your Kaggle account credentials.
