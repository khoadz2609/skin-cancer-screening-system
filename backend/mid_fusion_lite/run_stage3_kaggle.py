import os
import shutil
import sys
from pathlib import Path

# ========================================================
# 0. CẤU HÌNH THƯ MỤC CODE & DỮ LIỆU
# ========================================================
is_kaggle = os.path.exists("/kaggle/working")

if is_kaggle:
    # Nếu chạy trên Kaggle, add thư mục chứa code vào sys.path
    # TODO: Đổi tên dataset thư mục code của bạn trên Kaggle nếu cần
    code_dir_candidates = [
        Path("/kaggle/input/mid-fusion-lite-code"),
        Path("/kaggle/input/isic-01-07"),  # Ví dụ dataset code của bạn
    ]
    code_dir = next(
        (c for c in code_dir_candidates if (c / "config.py").exists()), None
    )
    if code_dir:
        sys.path.insert(0, str(code_dir))
        print(f"[+] Đã trỏ sys.path tới thư mục code: {code_dir}")
    else:
        print("[!] CẢNH BÁO: Không tìm thấy thư mục chứa code (config.py).")

from config import MidFusionLiteConfig, build_ablation_columns
from runner import run_mid_fusion_cv


def resolve_kaggle_data_paths(config):
    if is_kaggle:
        print("\n[+] Đang tự động tìm kiếm bộ dữ liệu ISIC 2024 trên Kaggle...")
        dataset_root = None

        # Quét toàn bộ thư mục /kaggle/input để tìm file train-metadata.csv
        for root, dirs, files in os.walk("/kaggle/input"):
            if "train-metadata.csv" in files:
                dataset_root = Path(root)
                break

        if dataset_root:
            print(f"    -> Đã tìm thấy dữ liệu ISIC tại: {dataset_root}")
            return config.with_updates(
                train_metadata_path=str(dataset_root / "train-metadata.csv"),
                test_metadata_path=str(dataset_root / "test-metadata.csv"),
                sample_submission_path=str(dataset_root / "sample_submission.csv"),
                train_hdf5_path=str(dataset_root / "train-image.hdf5"),
                test_hdf5_path=str(dataset_root / "test-image.hdf5"),
            )
        else:
            print(
                "    [!] LỖI NGHIÊM TRỌNG: Không tìm thấy file train-metadata.csv trong /kaggle/input/"
            )
            print(
                "    [!] Vui lòng đảm bảo bạn đã 'Add Data' bộ dataset của cuộc thi ISIC 2024 vào notebook!"
            )
    return config


def setup_and_run_fusion():
    print("=" * 60)
    print("  KAGGLE STAGE 3: TABM + TOP-1 IMAGE BACKBONE FUSION")
    print("=" * 60)

    # ========================================================
    # 1. ĐỒNG BỘ WEIGHTS TABM
    # ========================================================
    # Kiểm tra xem đang chạy trên Kaggle hay Local
    is_kaggle = os.path.exists("/kaggle/working")

    # Nguồn TabM weights: dùng TEMPLATE "{fold_1}" vì file đặt tên tabm_fold_1.pt (đánh số từ 1).
    # (Chế độ thư mục trần chỉ hoạt động nếu file tên mid_fusion_lite_fold_0.pt — không hợp ở đây.)
    if is_kaggle:
        tabm_weights = "/kaggle/input/datasets/dangnguyen2000/tabular-weight-02-07/tabm_ablation_weights/tabm_fold_{fold_1}.pt"
    else:
        tabm_weights = r"D:\HK1_2025_2026\SEMINAR_CODE\mid_fusion_lite\tabm_weights\tabm_fold_{fold_1}.pt"

    # ========================================================
    # 2. CẤU HÌNH MID-FUSION LITE (STAGE 3)
    # ========================================================
    # Nguồn Image weights: THƯ MỤC chứa các file mid_fusion_lite_fold_0.pt, _fold_1.pt, ...
    # (checkpoint đầy đủ nhánh ảnh đã train, KHÔNG phải file backbone timm thô).
    if is_kaggle:
        image_weights_dir = (
            "/kaggle/input/datasets/khiemdangle/image-backbone-eff-03-07/image_output"
        )
        out_dir = "/kaggle/working/fusion_output"
    else:
        image_weights_dir = (
            r"D:\HK1_2025_2026\SEMINAR_CODE\mid_fusion_lite\image_output"
        )
        out_dir = "fusion_output"

    # CỘT TABULAR PHẢI KHỚP TUYỆT ĐỐI (số lượng + THỨ TỰ) với notebook đã train TabM
    # (only_mode/tabular_only_standard.ipynb, tất cả feature flags = True → 740 ứng viên
    # → 610 cột sau FE). TabM gán trọng số PLE theo VỊ TRÍ cột: nếu Stage 3 dựng cột lệch
    # thứ tự, checkpoint sẽ nạp sai cột một cách âm thầm (không báo lỗi shape). Dùng chung
    # build_ablation_columns để đảm bảo cùng một nguồn sự thật với notebook.
    tabm_numeric_cols, tabm_categorical_cols = build_ablation_columns(
        use_engineered_abcd=True,
        use_patient_norm=True,
        use_ugly_duckling_adv=True,
        use_categorical_context=True,
    )

    fusion_config = MidFusionLiteConfig(
        preset_name="kaggle_top1_fusion",
        n_splits=5,
        fold_indices=None,  # Đổi thành [0, 1] nếu chỉ muốn chạy test 1-2 fold
        # --- A. CẤU HÌNH IMAGE (TOP-1 BACKBONE) ---
        image_size=224,
        backbone_name="efficientnet_b0",
        # Không cần backbone thô: nhánh ảnh đầy đủ được nạp qua pretrained_image_dir bên dưới.
        backbone_weights_path=None,
        train_image_backbone=True,  # Sẽ unfreeze ở Stage 3B
        download_pretrained_backbone=False,  # Không tải từ net vì đã dùng Top-1 local/Kaggle weights
        # --- B. CẤU HÌNH TABULAR (Phải khớp với file tabular-only-*.ipynb của bạn) ---
        tabm_k=32,
        tabm_n_blocks=2,
        tabm_d_block=256,
        n_bins=48,
        d_embedding=32,
        tab_dropout=0.3,
        # Bật đầy đủ các nhóm feature để FE sinh đúng bộ cột như Stage 1 (Tabular notebook)
        use_contextual_group_norm_features=True,
        use_ugly_duckling_features=True,
        use_anatomical_zscore_features=True,
        use_area_aggregation_features=True,
        # Tập cột đưa vào TabM — khớp tuyệt đối notebook (610 cột sau khi runner lọc).
        numeric_cols=tabm_numeric_cols,
        categorical_cols=tabm_categorical_cols,
        # --- C. CẤU HÌNH FUSION ---
        fusion_mode="cross_attention",
        fusion_hidden_dim=256,
        fusion_dropout=0.25,
        cross_attention_heads=4,
        # 85% both, 7.5% image-only, 7.5% tabular-only during fusion training.
        modality_dropout_prob=0.15,
        # --- D. CHỈ CHẠY FUSION TỪ HAI CHECKPOINT ĐỘC LẬP ---
        run_mode="fusion",
        pretrained_tabular_dir=tabm_weights,
        pretrained_image_dir=image_weights_dir,
        # Tắt auxiliary losses để chỉ tính loss duy nhất sau Fusion
        aux_loss_image_weight=0.0,
        aux_loss_tab_weight=0.0,
        stage3a_lr=1e-4,
        stage3b_lr=1e-5,  # lr nền cho backbone ảnh ở Stage 3B
        # Stage 3B dùng lr phân tầng: backbone = stage3b_lr (1e-5), còn
        # tabular_encoder + fusion_head = stage3b_lr * mult (5e-5) để cổng fusion
        # tiếp tục học thay vì đứng yên. Đặt =1.0 để tắt phân tầng.
        stage3b_head_lr_mult=5.0,
        train_batch_size=32,  # Giảm batch size vì ảnh 336 tốn VRAM
        valid_batch_size=64,
        negative_ratio=5,  # Tỷ lệ undersample cho Fusion (nhánh ảnh không cần quá nhiều âm tính)
        output_dir=out_dir,
    )

    # Cập nhật đường dẫn data nếu chạy trên Kaggle
    fusion_config = resolve_kaggle_data_paths(fusion_config)

    # ========================================================
    # 3. CHẠY FUSION PIPELINE
    # ========================================================
    print("\n[+] Bắt đầu quá trình huấn luyện Stage 3 Fusion...")
    results = run_mid_fusion_cv(fusion_config)

    # Kaggle yêu cầu file nộp bài phải có tên đúng là "submission.csv" nằm ở thư mục root /kaggle/working/
    if is_kaggle:
        src_sub = os.path.join(out_dir, "submission_mid_fusion_lite.csv")
        dst_sub = "/kaggle/working/submission.csv"
        if os.path.exists(src_sub):
            shutil.copy(src_sub, dst_sub)
            print(f"\n[+] Đã copy file submission ra thư mục gốc Kaggle: {dst_sub}")
        else:
            print(
                f"\n[!] WARNING: Không tìm thấy file {src_sub} để tạo submission.csv!"
            )

    print("\n[+] Hoàn thành toàn bộ quy trình!")
    print("\n" + "=" * 60)
    print("  KẾT QUẢ HUẤN LUYỆN FUSION")
    print("=" * 60)
    cv_auc = results.get('cv_auc')
    cv_pauc = results.get('cv_pauc')
    print(f"  CV AUC  : {f'{cv_auc:.5f}' if cv_auc is not None else 'N/A'}")
    print(f"  CV pAUC : {f'{cv_pauc:.5f}' if cv_pauc is not None else 'N/A'}")
    print("=" * 60)


if __name__ == "__main__":
    setup_and_run_fusion()
