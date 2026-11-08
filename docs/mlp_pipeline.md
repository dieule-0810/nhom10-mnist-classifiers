# Pipeline MLP: từ EDA tới thí nghiệm

Số liệu EDA lấy từ `eda/notebooks/eda.ipynb` (50.000 ảnh `splits["train"]`). Số liệu thí nghiệm lấy từ `results/mlp/log.txt` và `results/mlp/final_metrics.json` (`python -m src.experiments.mlp`, CPU 14 lõi, giới hạn 8 luồng BLAS). Mọi lựa chọn dựa trên validation; test chỉ dùng một lần ở E8.

## 1. Những gì EDA nói về MLP

### 1.1 Cách xử lý EDA đề xuất

| Phát hiện EDA | Mục | Ảnh hưởng tới MLP | Bản cài đặt |
|---|---|---|---|
| Pixel uint8, cộng trừ bị tràn | 0 | Trừ mean trên uint8 cho kết quả sai | `load_splits` trả float32 trong [0, 1] |
| 72 dead pixel (std = 0) | 1.1 | Chuẩn hoá từng pixel sẽ chia cho 0 | Chuẩn hoá vô hướng: một mean, một std, học trên train: `make_features("none", "scalar")` trong `Pipeline` |
| 116 pixel gần như chết; pixel kề nhau tương quan mạnh (r = 0,708) | 1.1, 2.1 | Trọng số lớp 1 dễ khớp nhiễu | L2 (`weight_decay`) hoặc dropout (`p_drop`), so ở E6 |
| Hiệp phương sai các lớp khác nhau: tổng phương sai trong lớp 22,4–50,8 | 1.2 | Ranh giới tuyến tính có trần, MLP phi tuyến nên vượt được | So với LR (H2) |
| Tâm lớp gần nhau: 4–9, 7–9, 5–8, 3–5, 3–8 | 3.1 | Các cặp dễ nhầm nhất | Khung cam trong `eval_confusion.png`, `share_in_pairs`, `pair_errors` |
| Fisher dồn ở vùng giữa ảnh | 3.3 | Phần học từ dữ liệu của W⁽¹⁾ nên nằm ở vùng Fisher cao | ‖R‖ theo pixel, hoán vị khối 4×4 (H4) |

### 1.2 Giả thuyết liên quan tới MLP (EDA 5.6) và kết quả

Kết luận cuối của từng giả thuyết ở `results/comparison/log.txt` (`compare_models`); bảng dưới chỉ ghi phần MLP đóng góp.

| # | Giả thuyết | Số liệu của MLP (val) |
|---|---|---|
| H2 | RF, MLP vượt LR về accuracy và F1 | acc 0,9831, F1 macro 0,9830 so với LR 0,920 / 0,919 → **ủng hộ** |
| H3 | Mô hình phi tuyến giảm lỗi ở 5 cặp tâm gần nhau mạnh hơn mức giảm chung | So với LR: giảm 85% lỗi ở 5 cặp, 79% tổng lỗi → vế (b) **đạt**. Lỗi còn lại: 25,4% ở 5 cặp (LR 36,7%); top 5 còn 3–5, 3–8, 4–9, thêm 2–7, 2–3 |
| H4 | Vùng mô hình dựa vào trùng vùng Fisher cao | ρ khối 4×4 = 0,88 → **ủng hộ** |

## 2. Thí nghiệm

Baseline: 784 → 256 → 128 → 10, ReLU, khởi tạo He, γ = 0,05, α = 0,9, batch 128. Sau 10 epoch val acc tốt nhất 0,9766 (Lab 04 yêu cầu ≥ 0,97). Mỗi thí nghiệm đổi một yếu tố so với baseline, 5 epoch (E5a: 3 epoch, E6: 15 epoch).

| Thí nghiệm | Kết quả chính | Lý thuyết minh hoạ |
|---|---|---|
| E1: γ × α | Tốt nhất γ = 0,05, α = 0,9 (0,9740), trùng baseline. γ_eff = γ/(1 − α) = 1 sụt còn 0,9669 với α = 0,9 và 0,5771 với α = 0,5; γ_eff = 5 và γ = 0,5 không momentum thì phân kỳ (loss = nan). Không momentum, γ nhỏ thì học chậm (0,9272) | Momentum phóng đại bước học thành γ/(1 − α) (Bài 4 tr. 66–67) |
| E2: batch | 256 tốt nhất (0,9742); 128 được 0,9740. Batch 16 sụp (0,0987): bước học hiệu dụng 0,5 với gradient rất nhiễu. Batch 1024 chỉ có 245 lần cập nhật, chưa kịp hội tụ (0,9659) | Đánh đổi nhiễu gradient và số lần cập nhật |
| E3: rộng / sâu | 784-1024-10 tốt nhất (0,9798). Hẹp nhất 784-64-10 kém nhất (0,9631); mạng 4 lớp ẩn 256 (0,9726) kém mạng 1 lớp 256 (0,9765) sau 5 epoch | Thêm độ rộng giúp; mạng sâu khó tối ưu hơn với cùng số epoch |
| E4: kích hoạt, mạng 6 lớp ẩn | Sigmoid: ‖∂J/∂W‖ ở lớp 1 nhỏ hơn lớp cuối 1355 lần, không học được (0,1124). Tanh + Xavier và ReLU + He giữ gradient đều giữa các lớp (tỉ lệ 0,4 và 1,1), đạt 0,965 và 0,969 | Vanishing gradient, σ′ ≤ 0,25 (Bài 4 tr. 39–41) |
| E5a: khởi tạo hằng số | Sau 3 epoch chỉ 5/64 nơ-ron ẩn khác nhau, val 0,86; Xavier 64/64, val 0,96 | Phá đối xứng (Bài 4 tr. 73) |
| E5b: Xavier hay He cho ReLU | Qua 10 lớp ReLU, std đầu ra Xavier giảm từ 0,72 xuống 0,032; He giữ 0,84 → 0,95 | He bù hệ số 1/2 do ReLU cắt nửa âm (tr. 74) |
| E6: điều chuẩn, 15 epoch | L2 λ = 1e-4 tốt nhất (val acc 0,9818, val loss 0,0817); dropout 0,2 kém hơn gần 0,5 điểm (0,9770, 0,0897); không điều chuẩn val loss đi lên lại (0,1280, nhỏ nhất 0,1042). Dropout 0,5 học chậm (0,9688) | Đúng dự báo EDA 5.4: cần L2 hoặc dropout |

## 3. Chọn mô hình cuối (E7)

Mỗi thí nghiệm chọn một yếu tố khi các yếu tố khác giữ ở baseline, nên ghép mọi lựa chọn lại chưa chắc là tốt nhất. Vì vậy so ba ứng viên, cùng early stopping theo val loss (patience 5, tối đa 50 epoch, khôi phục trọng số tốt nhất):

| Ứng viên | Cấu hình | Epoch tốt nhất | Val acc | Val loss | Thời gian |
|---|---|---|---|---|---|
| A: gộp E1 + E2 + E3 + E6 | 1024, γ 0,05, α 0,9, batch 256, L2 1e-4 | 9 | 0,9802 | 0,0729 | 53 s |
| **B: bộ tối ưu baseline + E3 + E6** | 1024, γ 0,05, α 0,9, batch 128, L2 1e-4 | 42 | **0,9831** | **0,0622** | 248 s |
| C: baseline + E6 | 256-128, γ 0,05, α 0,9, batch 128, L2 1e-4 | 4 | 0,9733 | 0,0931 | 10 s |

**Luật chọn:** val accuracy cao nhất ở epoch được khôi phục. Ghép mọi lựa chọn tốt nhất (A) lại thua B 0,29 điểm, đúng như lý do phải so ứng viên: với batch 256, A tốt nhất ở epoch 9 rồi dừng sớm ở epoch 14. B có cả val accuracy cao nhất lẫn val loss nhỏ nhất.

- Cấu hình cuối: 784 → 1024 → 10, ReLU, He, γ = 0,05, α = 0,9, batch 128, L2 λ = 1e-4, không dropout, 814.090 tham số.
- Early stopping dừng ở epoch 47, khôi phục epoch 42. `fit_time` = 247,7 s (gồm val loss cho early stopping, không gồm loss/acc trên train mỗi epoch).

## 4. Kết quả cuối (E8)

| Tập | Accuracy | F1 macro | Log-loss | Sai mà p > 0,99 | Độ tự tin TB |
|---|---|---|---|---|---|
| train | 1,0000 | 1,0000 | 0,0018 | 0 | 0,998 |
| val | 0,9831 | 0,9830 | 0,0622 | 0,23% | 0,988 |
| test | 0,9836 | 0,9835 | 0,0537 | 0,08% | 0,988 |

Train đạt 100%: mạng đủ lớn để thuộc lòng train; L2 và early stopping giữ khoảng cách train–val ở mức 1,7 điểm. Xác suất sát với accuracy: độ tự tin trung bình 0,988 so với accuracy 0,983 trên val, chỉ hơi quá tự tin.

**Giải thích trọng số lớp 1 (`interp_w1.png`).** W⁽¹⁾ = (khởi tạo, bị L2 co lại) + (phần học từ dữ liệu R). Ở 72 dead pixel mọi ảnh cùng giá trị, nên ΔW = W⁽¹⁾ − W⁽¹⁾_khởi tạo ở đó chỉ phản ánh L2 co phần khởi tạo; hệ số co k ước lượng bằng bình phương tối thiểu trên dead pixel, R = ΔW + k·W⁽¹⁾_khởi tạo. Cấu hình cuối dùng L2 nên phần khởi tạo bị co rõ: tương quan ΔW–W₀ trên dead pixel là −0,998, phần khởi tạo còn 44%; ‖ΔW‖ = 26,7 nhưng phần học từ dữ liệu chỉ ‖R‖ = 9,6. Các nơ-ron mạnh nhất vẫn là bộ dò nét cục bộ (nét chéo, ngang, vòng cung, cặp đỏ–xanh kề nhau) ở vùng giữa ảnh.

**Độ quan trọng theo vùng.** Hoán vị khối 4×4 trên val (số đo chung cho mọi mô hình): Spearman với tổng Fisher của khối 0,88 (`compare_models`).

## 5. Pipeline

```
data/raw/*.idx-ubyte
  └─ load_splits()                                  src/data/preprocess.py   [0,1], 784, 50k/10k/10k
       └─ Pipeline(make_features("none", "scalar"),  src/models/pipeline.py, src/data/features.py
                   MLPClassifierNP)                  src/models/mlp.py
            build_mlp: Linear (W shape (k_out, k_in)), khởi tạo He / Xavier / hằng số
            forward_with_cache (Z = AWᵀ + b) → softmax_cross_entropy (δ = (Ŷ − Y)/N) → backward (BP1–BP4, + λW)
            inverted dropout, SGDMomentum (Δw = αΔw − γ∇J), early stopping theo val loss
       └─ evaluate, probabilistic_scores, block_permutation_importance, save_final_metrics   src/evaluation/
  └─ plot_all() đọc summary.json, final_metrics.json → figures/       src/plots/mlp.py, src/plots/common.py
```

## 6. Vị trí file

| File | Nội dung |
|---|---|
| `src/models/mlp.py` | `MLPClassifierNP(BaseClassifier)`, `build_mlp`, `forward_with_cache`, `softmax_cross_entropy`, `backward`, `SGDMomentum` |
| `tests/check_mlp.py` | 5 kiểm tra: ví dụ softmax trên slide, gradient so với autograd (sai số tương đối < 1e-8), công thức momentum, công thức khởi tạo, forward so với sklearn |
| `src/experiments/mlp.py` | E0–E8: tính và ghi `log.txt`, `summary.json`, `final_metrics.json` |
| `src/plots/mlp.py` | vẽ `figures/` từ các file đó (`python -m src.plots mlp`) |
| `results/mlp/` | `log.txt`, `summary.json`, `final_metrics.json`, `model.pkl`, `figures/` (`hp_e1`–`hp_e6`, `eval_learning_curve`, `eval_confusion`, `eval_confident_errors`, `interp_w1`) |

Lệnh chạy (từ gốc repo): `python -m src.experiments.mlp` (13 phút trên CPU 14 lõi, 8 luồng BLAS).

## 7. Quy ước cài đặt

- Lan truyền thuận Z = AWᵀ + b và lan truyền ngược tự viết bằng NumPy; không dùng autograd khi huấn luyện.
- L2 cộng λW vào gradient của trọng số, không phạt bias. Dropout dạng inverted (chia cho 1 − p khi huấn luyện, không làm gì khi dự đoán).
- Softmax trừ max trước khi lấy exp (`log_softmax` dùng chung); `predict_proba` tính bằng float64.
- OpenBLAS mặc định mở nhiều luồng hơn số lõi, làm phép nhân cỡ mini-batch chậm hàng chục lần; script giới hạn bằng `--blas-threads` (mặc định 8).
- Mỗi thí nghiệm đổi một yếu tố, chọn theo validation, test dùng một lần.
