# Pipeline Naive Bayes: từ EDA tới thí nghiệm

Mọi số liệu EDA lấy từ `eda/notebooks/eda.ipynb` (50.000 ảnh `splits["train"]`). Số liệu thí nghiệm lấy từ `results/naive_bayes/log.txt`, chọn trên validation. Test chỉ dùng một lần ở E4. Số của LR và RF lấy từ `results/logistic_regression/` và `results/random_forest/`.

## 1. Những gì EDA nói về Naive Bayes

### 1.1 Cách xử lý EDA đề xuất

| Phát hiện EDA | Mục | Ảnh hưởng tới NB | Cách xử lý EDA đề xuất | Bản cài đặt |
|---|---|---|---|---|
| Pixel không âm, gần nhị phân (80,86% bằng 0; 69,2% pixel khác 0 ≥ 128) | 1.1 | Multinomial coi pixel là "lượng mực", không giả định hình dạng phân phối; Gaussian NB giả định sai | Multinomial NB trên pixel [0, 1], không chuẩn hoá, không PCA | `MultinomialNBNP`; E3 kiểm chứng (Gaussian 0,796 so với 0,823) |
| 72 dead pixel (std = 0) | 1.1 | x = 0 trên toàn train nên không góp gì; chỉ đổi mẫu số α·d | Bỏ hay giữ đều được | `PixelMask`; E2 |
| 116 pixel gần như chết | 1.1 | Nhiều lớp có tổng mực bằng 0, α = 0 cho log 0 = −∞ | Làm trơn Laplace α > 0, chọn α trên val | `alpha`; E1 |
| 99,76% mực trong 24×24 | 1.3 | Bớt đặc trưng | Ứng viên crop | `PixelMask("crop24" / "crop20")`; E2 |
| Pixel kề nhau tương quan mạnh trong cùng lớp (0,64–0,86) | 2.2 | Một nét bút bị đếm như nhiều bằng chứng độc lập, xác suất bị đẩy về 0 hoặc 1 | Không sửa được trong NB; đo log-loss và tỉ lệ sai mà p > 0,99 | `probabilistic_scores`; H1 |
| 5 cặp tâm gần nhất: 4–9, 7–9, 5–8, 3–5, 3–8 | 3.1 | NB so ảnh với một khuôn mẫu θ cho mỗi lớp, nên nhầm như tâm gần nhất | Theo dõi trong ma trận nhầm lẫn | `pair_errors`, `share_in_pairs`; H3 |
| Fisher dồn vào 129 pixel ở giữa ảnh | 3.3 | θ nên khác nhau giữa các lớp ở đúng vùng đó | Đặt độ quan trọng theo khối 4×4 cạnh Fisher | `block_permutation_importance`; H4 |
| θ_k = ảnh trung bình lớp k chia cho lượng mực của lớp, cộng làm trơn | 5.1 | Hàm quyết định tuyến tính, cùng dạng với LR | So NB → LR chỉ đổi cách tìm trọng số | `weights_ = log θᵀ`, `bias_ = log π` |
| Nhân x với hằng số thì độ tự tin đổi theo | 5.1 | Thang đo phải cố định | Cố định thang [0, 1] | E3 kiểm chứng |

### 1.2 Giả thuyết liên quan tới NB (EDA 5.6) và kết quả

Kết luận cuối của từng giả thuyết ở `results/comparison/log.txt` (`compare_models`); bảng dưới chỉ ghi phần NB đóng góp.

| # | Giả thuyết | Vai trò của NB | Số liệu của NB (val) |
|---|---|---|---|
| H1 | LR chính xác hơn và cho xác suất đáng tin hơn Multinomial NB | Một vế | acc 0,823 so với LR 0,920; log-loss 2,259 so với 0,291; 10,35% mẫu bị đoán sai với p > 0,99 (58,4% số lỗi) so với 0,19% của LR. Độ tự tin trung bình 0,978 trong khi accuracy chỉ 0,823 → **ủng hộ rõ**: đúng hệ quả của giả định độc lập bị vi phạm (EDA 2.2) |
| H2 | RF, MLP vượt LR | Không trực tiếp | Dòng mốc đáng chú ý: NB chỉ hơn tâm gần nhất 1,8 điểm (0,823 so với 0,805), vì θ_k gần như là ảnh trung bình lớp k đã chuẩn hoá (EDA 5.1), tức NB gần một bộ phân loại theo khuôn mẫu |
| H3 | Lỗi của mô hình tuyến tính dồn vào 5 cặp tâm gần nhau | Vế (a) | 5 cặp chiếm 40,9% lỗi (rải đều: 11,1%), đúng 5 cặp nhầm nhiều nhất; riêng 4–9 có 246 lỗi → vế (a) **đạt** |
| H4 | Vùng mô hình dựa vào trùng vùng Fisher cao | Kiểm chứng | ρ khối 4×4 = 0,90 (mục 2.4) |

## 2. Những điểm tinh chỉnh so với EDA (có số liệu)

### 2.1 α hầu như không quan trọng (E1)

Giữ cả 784 pixel:

| α | val acc | val log-loss | sai mà p > 0,99 |
|---|---|---|---|
| **0,0001** | **0,8228** | 2,2544 | 10,33% |
| 0,001 | 0,8228 | 2,2552 | 10,33% |
| 0,01 | 0,8227 | 2,2569 | 10,35% |
| 0,1 | 0,8225 | 2,2589 | 10,36% |
| 1 | 0,8215 | 2,2593 | 10,40% |
| 10 | 0,8205 | 2,2377 | 10,31% |

Từ α = 0,0001 đến 10 (5 bậc độ lớn), val accuracy chỉ chênh 23 ảnh trên 10.000. Luật chọn: val accuracy lớn nhất, bằng nhau thì val log-loss nhỏ hơn, nên chọn α = 0,0001, nằm ở biên dưới của lưới; 0,0001 và 0,001 cho cùng accuracy, nên mở rộng lưới xuống thấp hơn cũng chỉ đổi vài ảnh. α không sửa được độ tự tin quá mức: tỉ lệ sai mà p > 0,99 giữ ở khoảng 10% với mọi α, vì nguyên nhân là giả định độc lập (EDA 2.2), không phải ước lượng θ.

### 2.2 Bỏ pixel hiếm (E2)

| mask | số đặc trưng | val acc | val log-loss |
|---|---|---|---|
| none | 784 | 0,8228 | 2,2544 |
| dead | 712 | 0,8228 | 2,2544 |
| **rare** | **596** | **0,8229** | 2,2585 |
| crop24 | 576 | 0,8218 | 2,2553 |
| crop20 | 400 | 0,8126 | 2,2127 |

NB không có thành phần ngẫu nhiên, nên chênh lệch là chênh lệch thật. Luật chọn: ít đặc trưng nhất trong các mặt nạ có val accuracy cách tốt nhất không quá 0,001 (10 ảnh), cùng tinh thần quy tắc của LR và RF. Theo luật này, mô hình cuối dùng `rare` (596 pixel). Bỏ dead pixel không đổi kết quả, đúng dự báo của EDA (x = 0 thì không góp gì). Crop 20×20 mất 1,0 điểm accuracy: khác với RF, NB cần cả phần mực ở viền khung 20×20.

### 2.3 Kiểm chứng lựa chọn "Multinomial trên lượng mực" (E3)

Mặt nạ rare; mỗi dòng chọn tham số tốt nhất trên val: α ∈ {0,001; 0,01; 0,1; 1} cho Multinomial, var_smoothing ∈ {10⁻⁹; 10⁻³; 0,01; 0,1; 0,3; 1} cho Gaussian:

| biểu diễn / mô hình | tham số | val acc | val log-loss | lỗi có p > 0,99 |
|---|---|---|---|---|
| **Multinomial, x ∈ [0, 1]** (EDA 5.1, mô hình cuối) | α = 0,0001 | **0,8229** | **2,259** | **58,4%** |
| Multinomial, x > 0 | α = 0,01 | 0,8015 | 3,037 | 65,8% |
| Multinomial, x ∈ [0, 255] | α = 0,001 | 0,8229 | 6,105 | 99,8% |
| Gaussian, x ∈ [0, 1] | var_smoothing = 0,1 | 0,7964 | 5,069 | 87,4% |

- **Cường độ tốt hơn nhị phân với Multinomial** (+2,1 điểm accuracy, log-loss thấp hơn rõ): mức xám ở mép nét mang thông tin, đúng lập luận "lượng mực".
- **Thang đo đổi mức tự tin, đúng EDA 5.1**: nhân x với 255 không đổi accuracy (0,8229) nhưng log-loss tăng từ 2,26 lên 6,11 và 99,8% lỗi có p > 0,99, vì phần log-likelihood nhân 255 còn log π giữ nguyên.
- **Gaussian NB kém Multinomial ở mọi độ đo, đúng EDA 5.1** (giả định chuẩn sai với pixel gần nhị phân): kém 2,6 điểm accuracy (265 ảnh val), log-loss gấp 2,2 lần và 87,4% lỗi có p > 0,99. Với ε mặc định của sklearn (var_smoothing = 10⁻⁹) chỉ đạt 0,5849: ở pixel mà một lớp gần như không có mực, σ²_cj ≈ ε rất nhỏ, nên một chút mực lạc ở đó cũng làm log-likelihood của lớp đó rơi rất sâu. Tăng ε đến 0,1 · max Var(x_j) lên 0,7964, rồi giảm lại (0,3: 0,7792; 1: 0,7266) khi mọi pixel gần như cùng phương sai. var_smoothing lớn ở đây thực chất là sửa giả định Gaussian, không phải chỉ làm trơn số.

### 2.4 Độ quan trọng theo vùng (H4)

Số đo chung cho mọi mô hình: hoán vị khối 4×4 trên val. Spearman với tổng Fisher của khối là 0,90; khối quan trọng nhất (giữa ảnh, hàng 12–15, cột 8–11) làm accuracy val giảm 3,6 điểm.

### 2.5 Giải thích bằng θ

θ_c (`interp_theta.png`) là phân phối mực của lớp c, chính là ảnh trung bình ở EDA 1.2 chia cho lượng mực của lớp. Vì θ_c chuẩn hoá về tổng 1, mô hình không dùng chênh lệch tổng lượng mực giữa các lớp (số 0: 135,8; số 1: 59,5), đúng như EDA 5.1 lưu ý.

## 3. Pipeline

```
data/raw/*-ubyte
  └─ load_splits()                          src/data/preprocess.py   [0,1], 784, chia 50k/10k stratify seed 42
       └─ FeaturePipeline (fit trên train)  src/data/features.py     (dùng chung)
            PixelMask("rare")      bỏ 188 pixel có tần suất < 0,1%  → 596
       └─ MultinomialNBNP                   src/models/naive_bayes.py
            α = 0,0001; log θ = log((N_cj + α) / (N_c + α·d)); log-sum-exp cho predict_proba
       └─ Pipeline                          src/models/pipeline.py   (dùng chung)
       └─ evaluate()                        src/evaluation/metrics.py   acc/P/R/F1/CM train, val, test, thời gian
          probabilistic_scores, top_confused_pairs, block_permutation_importance
                                            src/evaluation/analysis.py  (dùng chung cho mọi mô hình)
```

α và mặt nạ được script chọn lại từ validation mỗi lần chạy; giá trị trên là kết quả của lần chạy này.

**Kết quả cuối** (huấn luyện 0,51 s, gồm chọn pixel):

| tập | accuracy | precision | recall | F1 macro | log-loss | độ tự tin TB |
|---|---|---|---|---|---|---|
| train | 0,8247 | 0,8317 | 0,8216 | 0,8233 | 2,187 | 0,977 |
| val | 0,8229 | 0,8317 | 0,8200 | 0,8220 | 2,259 | 0,978 |
| test | 0,8356 | 0,8431 | 0,8325 | 0,8342 | 2,007 | 0,977 |

Train và val gần bằng nhau: NB có ít tham số (10 × 596 xác suất) nên không overfit; giới hạn nằm ở giả định độc lập, không ở lượng dữ liệu. `sklearn.naive_bayes.MultinomialNB` cùng α và mặt nạ cho đúng cùng accuracy (val 0,8229, test 0,8356).

## 4. Vị trí file

| File | Nội dung |
|---|---|
| `src/models/naive_bayes.py` | `MultinomialNBNP` (mô hình chính; có `weights_`, `bias_`), `GaussianNBNP` (đối chứng ở E3; hàm quyết định bậc hai) |
| `src/experiments/naive_bayes.py` | Thí nghiệm E0–E4: tính và ghi số liệu |
| `src/plots/naive_bayes.py` | vẽ `figures/` từ `summary.json`, `final_metrics.json` (`python -m src.plots naive_bayes`) |
| `tests/check_naive_bayes.py` | 4 kiểm tra: đối chiếu sklearn (Multinomial trên số đếm và cường độ, Gaussian, dạng tuyến tính) |
| `results/naive_bayes/` | do script sinh ra: `log.txt`, `summary.json`, `final_metrics.json`, `model.pkl`, `figures/hp_*`, `eval_*`, `interp_*` |

Dùng chung: `src/data/features.py` (`PixelMask`, `Quantizer`, `make_features`), `src/models/pipeline.py`,
`src/models/nearest_centroid.py`, `src/evaluation/analysis.py`, `src/evaluation/results_io.py`,
`src/plots/common.py`. `src/experiments/sklearn_reference.py` có thêm `build_naive_bayes`.

Lệnh chạy (từ gốc repo):

```
python tests/run_all.py
python -m src.experiments.naive_bayes --quick   # lưới nhỏ, chỉ để kiểm tra code chạy được
python -m src.experiments.naive_bayes           # bản đầy đủ, dưới 1 phút
python -m src.experiments.compare_models        # so sánh mọi mô hình đã có final_metrics.json
```

**Số liệu trong tài liệu này lấy từ lần chạy đầy đủ trong repo** (`results/naive_bayes/log.txt`). NB không có
thành phần ngẫu nhiên nên mọi số trừ thời gian tái lập chính xác.

## 5. Quy ước cài đặt

- Lớp `...NP` kế thừa `BaseClassifier`; siêu tham số lưu nguyên tên trong `__init__`; thuộc tính học được có hậu tố `_`.
- Tính trong miền log; `predict_proba` chuẩn hoá bằng `log_softmax` (log-sum-exp) từ `src/models/_numeric.py`.
- α > 0 bắt buộc; Multinomial từ chối đầu vào âm.
- Mỗi thí nghiệm đổi một yếu tố, chọn theo validation; chênh accuracy dưới 10 ảnh coi như bằng nhau và chọn phương án ít đặc trưng hơn. Test dùng một lần.
- sklearn chỉ xuất hiện trong `tests/` và `src/experiments/sklearn_reference.py` để đối chiếu.
