import os

import pandas as pd
import numpy as np
import h5py
import os

def create_demo_dataset():
    # Đường dẫn chuẩn trên Kaggle (Đã bao gồm file chứa cột fold)
    input_csv = '/kaggle/working/train_metadata_with_folds.csv'
    input_h5 = '/kaggle/input/competitions/isic-2024-challenge/train-image.hdf5'
    
    # Nơi lưu trữ bản Demo
    output_dir = '/kaggle/working/demo_data'
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, 'demo_metadata.csv')
    output_h5 = os.path.join(output_dir, 'demo_image.hdf5')
    
    # [Fallback nếu chạy thử ở local]
    if not os.path.exists(input_csv):
        print(f"[Cảnh báo] Không tìm thấy {input_csv}. Dùng dữ liệu local...")
        input_csv = 'train_metadata_with_folds.csv'
        input_h5 = 'train-image.hdf5'
        if not os.path.exists(input_csv):
            print("Vui lòng chạy script kaggle_split_data.py trước để tạo file chia Fold!")
            return

    print("1. Đang tải Metadata (đã chia Folds)...")
    df = pd.read_csv(input_csv)
    
    print("2. Lọc chỉ lấy tập Validation (Fold 0)...")
    # LỌC CHỈ LẤY FOLD 0 (Vì fold 0 thường được dùng làm Validation trong demo của bạn)
    df_val = df[df['fold'] == 0]
    print(f"   Tổng số ảnh trong tập Validation (Fold 0): {len(df_val)}")
    
    print("3. Phân tích bệnh nhân (có bệnh) trong tập Validation...")
    # Lấy danh sách các patient_id CÓ ít nhất 1 ảnh BỊ BỆNH (target == 1) TỪ TẬP VALIDATION
    patients_with_disease = df_val[df_val['target'] == 1]['patient_id'].unique()
    print(f"   Tìm thấy {len(patients_with_disease)} bệnh nhân có bệnh trong tập Val.")
    
    # Chỉ giữ lại dataframe của những bệnh nhân này (tất cả ảnh của họ trong fold 0)
    df_positive_patients = df_val[df_val['patient_id'].isin(patients_with_disease)]
    
    # Đếm tổng số ảnh (cả bệnh và không bệnh) của từng bệnh nhân
    patient_image_counts = df_positive_patients.groupby('patient_id').size()
    
    # Lấy mức TRUNG VỊ (median) lượng ảnh để đại diện cho "trung bình"
    median_count = int(patient_image_counts.median())
    print(f"   Mức lượng ảnh trung vị của nhóm này là: {median_count} ảnh/bệnh nhân.")
    
    # Tìm 1 bệnh nhân có tổng số lượng ảnh GẦN VỚI MỨC TRUNG VỊ NHẤT
    patient_image_counts_diff = (patient_image_counts - median_count).abs()
    selected_patient = patient_image_counts_diff.idxmin()
    exact_count = patient_image_counts[selected_patient]
    
    print(f"4. Đã TÌM ĐƯỢC 1 Bệnh nhân thỏa mãn:")
    print(f"   -> ID: {selected_patient}")
    print(f"   -> Tổng số ảnh: {exact_count}")
    print(f"   -> Số ảnh có bệnh (target=1): {df_val[(df_val['patient_id'] == selected_patient) & (df_val['target'] == 1)].shape[0]}")
    
    # Trích xuất riêng data của bệnh nhân này
    demo_df = df_val[df_val['patient_id'] == selected_patient].copy()
    
    # Lưu ra CSV nhỏ
    demo_df.to_csv(output_csv, index=False)
    print(f"   Đã lưu file Metadata thu gọn tại: {output_csv}")
    
    # --- TRÍCH XUẤT ẢNH TỪ HDF5 KHỔNG LỒ SANG HDF5 MINI ---
    isic_ids_to_extract = demo_df['isic_id'].tolist()
    
    print(f"5. Đang trích xuất ảnh từ file HDF5 gốc sang HDF5 mini...")
    try:
        with h5py.File(input_h5, 'r') as h5_in:
            with h5py.File(output_h5, 'w') as h5_out:
                for idx, isic_id in enumerate(isic_ids_to_extract):
                    # Copy binary data của hình ảnh trực tiếp sang file mới
                    image_data = h5_in[isic_id][()]
                    h5_out.create_dataset(isic_id, data=image_data)
                    
                    if (idx + 1) % 10 == 0 or (idx + 1) == len(isic_ids_to_extract):
                        print(f"   Đã copy {idx + 1}/{len(isic_ids_to_extract)} ảnh...")
                        
        print(f"   Đã lưu HDF5 thu gọn tại: {output_h5}")
        print("\nHOÀN TẤT! BỘ DỮ LIỆU DEMO (Chỉ thuộc Validation Fold 0) ĐÃ SẴN SÀNG.")
        print("Hãy tải 2 file trong /kaggle/working/demo_data/ về máy tính.")
    except Exception as e:
        print(f"Lỗi đọc HDF5: {e}")

if __name__ == '__main__':
    create_demo_dataset()
