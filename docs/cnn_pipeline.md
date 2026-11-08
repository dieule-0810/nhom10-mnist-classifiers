# Pipeline CNN (phần nâng cao): từ EDA tới thí nghiệm

Số liệu EDA lấy từ `eda/notebooks/eda.ipynb` (50.000 ảnh `splits["train"]`). Số liệu thí nghiệm lấy từ `results/cnn/log.txt` và `results/cnn/final_metrics.json` (`python -m src.experiments.cnn`, CPU 14 lõi, giới hạn 8 luồng BLAS). Mọi lựa chọn dựa trên validation; test chỉ dùng một lần ở E4.

CNN không thuộc 4 mô hình bắt buộc và không thuộc các giả thuyết H1–H4. Câu hỏi của phần này: cùng là mạng nơ-ron, thêm tích chập có hơn MLP không, và hơn ở đâu.

## 1. Những gì EDA nói về CNN

### 1.1 Lý do làm CNN (EDA 5.4) và cách xử lý

| Phát hiện EDA | Mục | Ảnh hưởng | Bản cài đặt |
|---|---|---|---|
| Thông tin cục bộ: tương quan hai pixel giảm từ r ≈ 0,71 (kề nhau) về ≈ 0 ở khoảng cách 5 | 2.1 | Một bộ lọc nhỏ dùng chung cho mọi vị trí đủ bao tầm tương quan, thay vì học riêng 784 trọng số như lớp 1 của MLP | Conv 3×3, hai khối (28 → 14 → 7) |
| Cặp tâm gần nhau chỉ khác ở vùng nhỏ (24–46 pixel), ví dụ đỉnh số 4 hở hay số 9 khép | 3.2 | Đặc trưng cục bộ mà tích chập phát hiện được ở bất kỳ vị trí nào | So lỗi ở 5 cặp với MLP (mục 3) |
| Vị trí nét lệch vài pixel trong cùng lớp | 1.2 | MLP, LR, RF gắn trọng số / phép chia với một pixel cố định | Max-pool 2×2 chịu được độ lệch nhỏ |
| Pixel uint8, 72 dead pixel | 0, 1.1 | Chuẩn hoá từng pixel chia cho 0 | Chuẩn hoá vô hướng, học trên train: `make_features("none", "scalar")` |
| Cần đủ lưới 28×28 | — | Bỏ pixel (mặt nạ, crop) làm mất cấu trúc lưới | Không dùng mặt nạ pixel |

EDA cũng ghi giới hạn: CNN nhỏ này chỉ chịu được dịch vài pixel, không bất biến với độ nghiêng hay dịch lớn; nhóm không làm tăng cường dữ liệu.

### 1.2 Kết quả so với MLP (val)

Cùng số đo với H2, H3 (`results/comparison/log.txt`, mục "Nâng cao"):

| | MLP | CNN | Thay đổi |
|---|---|---|---|
| Accuracy / F1 macro | 0,9831 / 0,9830 | 0,9855 / 0,9854 | +0,24 điểm |
| Log-loss | 0,0622 | 0,0493 | tốt hơn |
| Sai mà p > 0,99 | 0,23% | 0,12% | |
| Tổng lỗi | 169 | 145 | −14% |
| Lỗi ở 5 cặp EDA 3.1 (4–9, 7–9, 5–8, 3–5, 3–8) | 43 (9, 9, 5, 10, 10) | 42 (12, 6, 8, 6, 10) | −2% |
| Số tham số | 814.090 | 206.922 | ít hơn 4 lần |
| Huấn luyện | 248 s | 273 s | 8 luồng BLAS cả hai |

- **CNN chính xác hơn MLP với ít tham số hơn 4 lần.** Khoảng cách 0,24 điểm (24 ảnh val). CNN cuối (seed 0) lại là seed kém nhất: 5 seed cho val 0,9855–0,9891, trung bình 0,9872 (`results/ensemble/log.txt`, E3), nên khoảng cách điển hình với MLP gần 0,4 điểm, và seed nào cũng hơn MLP.
- **Phần giảm lỗi không nằm ở 5 cặp gần nhau.** Ở 5 cặp lỗi chỉ giảm 2% (43 → 42), trong khi tổng lỗi giảm 14%: tích chập không giúp đặc biệt ở các cặp chỉ khác nhau ở vùng nhỏ như EDA dự đoán. 3–5 giảm từ 10 xuống 6 lỗi, nhưng 4–9 tăng từ 9 lên 12 và 5–8 từ 5 lên 8. Lỗi còn lại có 29,0% ở 5 cặp (rải đều: 11,1%), 3/5 cặp còn trong top 5 nhầm nhiều nhất (4–9, 3–8, 8–9, 5–8, 2–3).
- Test (một lần): CNN 0,9882 so với MLP 0,9836, cùng chiều với val.

## 2. Thí nghiệm

Xuất phát từ ví dụ PyTorch trên slide 12 bài 06: Conv 3×3 (1 → 16) → ReLU → MaxPool → Conv 3×3 (16 → 32) → ReLU → MaxPool → Dense 1568 → 128 → 10, Adam lr = 1e-3, batch 128, 3 epoch. Mỗi thí nghiệm đổi một yếu tố. Mốc phải vượt (E0, đọc `final_metrics.json` của các mô hình): MLP val 0,9831, RF 0,9694.

| Thí nghiệm | Kết quả chính | Lựa chọn |
|---|---|---|
| E1: learning rate của Adam (3 epoch) | 3e-4: 0,9788; 1e-3: 0,9837; 3e-3: 0,9809 (`hp_e1_lr.png`) | lr = 1e-3, đúng giá trị của ví dụ trên slide: 3e-4 hội tụ chậm hơn, 3e-3 bước quá lớn (val loss 0,061 so với 0,056) |
| E2: số bộ lọc mỗi khối (3 epoch, lr 1e-3) | (8, 16): 0,9816, 103.018 tham số, 58 s; (16, 32): 0,9837, 206.922, 101 s; (32, 64): 0,9800, 421.642, 185 s (`hp_e2_width.png`) | (16, 32). Luật: nhanh nhất trong các cấu hình cách tốt nhất ≤ 0,1 điểm (10 ảnh val); (8, 16) kém 0,21 điểm nên bị loại |
| E3: số epoch, tối đa 20, dừng sớm khi val loss không giảm 3 epoch | Val loss nhỏ nhất ở epoch 5 (0,0493); val acc đạt 0,9858 ở epoch 7 nhưng val loss đã tăng lại; dừng ở epoch 8 (`eval_learning_curve.png`) | Khôi phục epoch 5. Huấn luyện 273 s (khoảng 34 s/epoch) |

Ở E2, mạng rộng nhất (32, 64) kém nhất sau 3 epoch: gấp đôi tham số nhưng cùng số bước cập nhật, chưa kịp hội tụ. Thời gian tăng gần theo số bộ lọc (58 s, 101 s, 185 s); (8, 16) nhanh gần gấp đôi (16, 32) nhưng kém 0,21 điểm.

Dừng sớm theo val loss chọn epoch 5 (val acc 0,9855), không chọn epoch 7 (0,9858) có accuracy cao hơn một chút: luật dừng chốt trước theo val loss, như MLP. Chọn theo val acc ở đây là chọn thêm một siêu tham số trên val, dễ lạc quan.

## 3. Kết quả cuối (E4)

Cấu hình: Conv 3×3 (16, 32 bộ lọc) + MaxPool 2×2 hai khối, Dense 128, ReLU, khởi tạo He, Adam lr = 1e-3, batch 128, epoch 5, 206.922 tham số.

| Tập | Accuracy | F1 macro | Log-loss | Sai mà p > 0,99 | Độ tự tin TB | Dự đoán (s) |
|---|---|---|---|---|---|---|
| train | 0,9963 | 0,9963 | 0,0121 | 0,004% | 0,994 | 13,72 |
| val | 0,9855 | 0,9854 | 0,0493 | 0,12% | 0,991 | 2,83 |
| test | 0,9882 | 0,9881 | 0,0391 | 0,13% | 0,990 | 2,88 |

- Khoảng cách train–val 1,1 điểm, nhỏ hơn MLP (1,7 điểm) dù không có dropout hay L2; ngoài dừng sớm (epoch 5), tích chập dùng chung trọng số cho mọi vị trí nên có ít tham số tự do hơn.
- Lớp khó nhất là 9 (val 97,2%, test 97,0%), chủ yếu nhầm sang 4 (10 ảnh val, 11 ảnh test); lớp 8 đạt 98,1% (val) và 98,8% (test).
- Dự đoán chậm hơn MLP khoảng 20 lần (2,9 s so với 0,14 s cho 10.000 ảnh) dù ít tham số hơn: mỗi bộ lọc được áp lên mọi vị trí của ảnh.

**Độ quan trọng theo vùng.** Hoán vị khối 4×4 trên val (số đo chung cho mọi mô hình): Spearman với tổng Fisher của khối 0,930 (`compare_models`); khối quan trọng nhất làm giảm acc 0,049.

## 4. Đối chiếu PyTorch

`python -m src.experiments.torch_reference` huấn luyện cùng kiến trúc, cùng cấu hình cuối bằng `torch.nn` (2.12, CPU) và so (`results/torch_reference/log.txt`):

| | NumPy (tự cài) | PyTorch | Chênh |
|---|---|---|---|
| Val acc | 0,9855 | 0,9880 | −0,25 điểm |
| Test acc | 0,9882 | 0,9888 | −0,06 điểm |
| Epoch giữ lại / dừng | 5 / 8 | 7 / 10 | |
| Huấn luyện | 273 s (8 luồng) | 84 s (8 luồng) | NumPy chậm hơn 3,2 lần |

Hai bản khác khởi tạo (He chuẩn so với Kaiming đều của torch) và khác dãy trộn mini-batch, nên chỉ kỳ vọng accuracy gần nhau, không trùng từng ảnh. Bản torch (0,9880 trên val) nằm trong dải 5 seed của bản NumPy (0,9855–0,9891), nên chênh lệch này không vượt dao động do seed; dù vậy phần đúng của bản tự cài được kiểm bằng `tests/check_cnn.py` (gradient, conv, max-pool, Adam so với torch), không dựa vào con số này.

## 5. Pipeline

```
data/raw/*.idx-ubyte
  └─ load_splits()                                   src/data/preprocess.py   [0,1], 784, 50k/10k/10k
       └─ Pipeline(make_features("none", "scalar"),   src/models/pipeline.py, src/data/features.py
                   CNNClassifierNP)                   src/models/cnn.py
            reshape (N, 784) → (N, 28, 28, 1), NHWC
            Conv2D (im2col, He) → ReLU → MaxPool2x2 → Conv2D → ReLU → MaxPool2x2 → Flatten → Dense → ReLU → Dense
            log_softmax + cross-entropy, backward tự viết, Adam, dừng sớm theo val loss (khôi phục epoch tốt nhất)
       └─ evaluate, probabilistic_scores, block_permutation_importance, save_final_metrics    src/evaluation/
  └─ plot_all() đọc summary.json, final_metrics.json → figures/        src/plots/cnn.py, src/plots/common.py
```

## 6. Vị trí file

| File | Nội dung |
|---|---|
| `src/models/cnn.py` | `Conv2D`, `ReLU`, `MaxPool2x2`, `Flatten`, `Dense`, `Adam`, `build_layers`, `CNNClassifierNP(BaseClassifier)` (`fit`, `predict_proba`) |
| `tests/check_cnn.py` | 5 kiểm tra: gradient dense và cả mạng so với sai phân, conv và max-pool so với torch, Adam so với `torch.optim.Adam` |
| `src/experiments/cnn.py` | E0–E4: tính và ghi `log.txt`, `summary.json`, `final_metrics.json` |
| `src/plots/cnn.py` | vẽ `figures/` từ các file đó (`python -m src.plots cnn`) |
| `src/experiments/torch_reference.py` | đối chiếu PyTorch (mục 4) |
| `results/cnn/` | `log.txt`, `summary.json`, `final_metrics.json`, `model.pkl`, `figures/` (`hp_e1_lr`, `hp_e2_width`, `eval_learning_curve`, `eval_confusion`, `eval_confident_errors`) |
| `results/torch_reference/` | `log.txt`, `summary.json` |

Lệnh chạy (từ gốc repo): `python -m src.experiments.cnn` (21 phút trên CPU 14 lõi, 8 luồng BLAS); `--quick` để thử nhanh.

**Tái lập.** Accuracy tái lập chính xác khi cùng seed **và** cùng số luồng BLAS (`--blas-threads`, mặc định 8). Đổi số luồng làm đổi thứ tự cộng số thực trong phép nhân ma trận, sai khác nhỏ tích luỹ qua hàng nghìn bước Adam, nên accuracy có thể lệch vài phần nghìn, vẫn trong dải dao động do seed.

## 7. Quy ước cài đặt

- Lan truyền thuận và ngược tự viết bằng NumPy; PyTorch chỉ có trong `tests/` và `torch_reference`.
- Bố cục NHWC để im2col chỉ cần reshape; trọng số conv lưu (F, C, kh, kw) như PyTorch để test chép thẳng sang torch.
- Huấn luyện bằng float32; `decision_function` và `predict_proba` trả float64; softmax qua `log_softmax` dùng chung.
- Seed điều khiển cả khởi tạo lẫn thứ tự mini-batch (hai nguồn đa dạng mà seed ensemble dùng, xem `ensemble_pipeline.md`).
- Mỗi thí nghiệm đổi một yếu tố, chọn theo validation, test dùng một lần.
