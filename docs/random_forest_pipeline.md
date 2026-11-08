# Pipeline Random Forest: từ EDA tới thí nghiệm

Số liệu EDA lấy từ `eda/notebooks/eda.ipynb` (50.000 ảnh `splits["train"]`). Số liệu thí nghiệm lấy từ `results/random_forest/log.txt`, chọn trên validation; test chỉ dùng một lần ở E6. Rừng là `sklearn.ensemble.RandomForestClassifier` (nhánh ensemble được phép dùng thư viện, như Lab tuần 5 và slide B6b); phần tự viết là tiền xử lý (`TreeFeatures`: mặt nạ pixel, lượng tử), các phép phân tích rừng trong `src/models/random_forest.py` (phiếu từng cây, accuracy theo số cây, sức mạnh và tương quan lỗi) và các độ đo dùng chung cho mọi mô hình.

## 1. Những gì EDA nói về Random Forest

### 1.1 Phát hiện, ảnh hưởng, cách xử lý

| Phát hiện EDA | Mục | Ảnh hưởng tới RF (theo EDA) | EDA đề xuất | Bản cài đặt | Kiểm chứng |
|---|---|---|---|---|---|
| Pixel uint8, cộng trừ bị tràn | 0 | Không ảnh hưởng (cây chỉ so sánh) | Luôn qua `load_splits` | `Quantizer` đọc lại đúng mức 0..255 từ [0, 1] | test |
| Pixel không âm, gần nhị phân (80,86% bằng 0; 69,2% pixel khác 0 ≥ 128) | 1.1 | Chỉ cần ít ngưỡng ứng viên | Không thử cả 256 mức | `Quantizer(n_bins)`, bin 0 luôn là "không mực" | **E1**: 2 bin (pixel > 0) đủ |
| 72 dead pixel (std = 0) | 1.1 | Đặc trưng hằng, không bao giờ được chọn chia | Bỏ hoặc giữ đều được | sklearn bỏ qua đặc trưng hằng trong nút khi tìm phép chia; `PixelMask("dead")` | **E2** |
| 116 pixel gần như chết | 1.1 | Học từ vài chục ảnh, dễ khớp nhiễu | `min_samples_leaf`, giới hạn độ sâu ("bắt buộc", bảng 6.5) | `min_samples_leaf`, `max_depth` | **E5**: không cần |
| 99,76% mực trong 24×24 | 1.3 | "RF lợi nhất", chi phí ∝ số đặc trưng | Ứng viên crop 24×24 | `PixelMask("crop24")`, thêm `crop20` | **E2** |
| Pixel kề nhau tương quan mạnh (r = 0,708) | 2 | Importance loang ra các pixel lân cận; tập con ngẫu nhiên vẫn chứa pixel hữu ích | Đọc importance theo vùng | Độ quan trọng hoán vị **theo khối 4×4** | E6, H4 |
| Hiệp phương sai các lớp khác nhau: tổng phương sai trong lớp 22,4–50,8 | 1.2 | Ranh giới phi tuyến | Cần mô hình phi tuyến | — | H2 |
| 5 cặp tâm gần nhất: 4–9, 7–9, 5–8, 3–5, 3–8 | 3.1 | Cặp có nguy cơ nhầm | Theo dõi trong ma trận nhầm lẫn | `pair_errors`, `share_in_pairs`, `top_confused_pairs` | H3 |
| Khác biệt giữa cặp gần nằm ở 24–46 pixel | 3.2 | Phép chia theo từng pixel dùng trực tiếp được | — | — | H3 |
| Fisher dồn ở 129 pixel giữa ảnh | 3.3 | Importance nên trùng vùng Fisher cao | Đặt cạnh bản đồ Fisher | `fisher_scores`, `spearman` | H4 |
| Tâm lớp tách rõ (2,48–7,56) | 3.1 | Có cơ sở cho baseline | Tâm gần nhất làm mốc | `NearestCentroidNP` | mốc |
| Không cần chuẩn hoá | 5.3 | Ngưỡng bất biến với biến đổi đơn điệu | Pixel [0, 1] | Không chuẩn hoá | — |
| m = √d; "cây mạnh, tương quan thấp" (Breiman 2001) | 5.3 | ≈ 12 trong 28 pixel/nút thuộc nhóm 347 pixel Fisher cao | `max_features` là siêu tham số chính | `max_features`; đo sức mạnh và tương quan cây | **E4** |
| Thêm cây không overfit; có OOB | 5.3 | Đường accuracy theo T bão hoà | Thí nghiệm số cây có biểu đồ | `oob_score_` (warm start, đo ở các mốc T), `staged_accuracy` | **E3** |
| Dự đoán bằng bỏ phiếu | 5.3 | — | — | `tree_votes(rf, X, "soft" / "hard")` | E3 |

### 1.2 Giả thuyết liên quan tới RF (EDA 5.6) và kết quả

Kết luận cuối của từng giả thuyết ở `results/comparison/log.txt` (`compare_models`); bảng dưới chỉ ghi phần RF đóng góp.
Crop/bỏ pixel (E2) và giới hạn cây (E5) là **chọn siêu tham số**, không phải giả thuyết; kết quả ở mục 2.2 và 2.5.

| # | Giả thuyết | Số liệu của RF (val) |
|---|---|---|
| H2 | RF và MLP vượt Logistic Regression | acc 0,9694, F1 macro 0,9693 so với LR 0,920 / 0,919 → **ủng hộ** |
| H3 | Mô hình phi tuyến giảm lỗi ở 5 cặp tâm gần nhau mạnh hơn mức giảm chung | So với LR: giảm 68% lỗi ở 5 cặp, 62% tổng lỗi → vế (b) **đạt**. 5 cặp còn chiếm 30,4% lỗi của RF (LR 36,7%, MLP 25,4%, CNN 29,0%); top 5 cặp nhầm của RF còn 3/5 cặp EDA (7–9, 3–8, 4–9, thêm 2–3 và 2–7) |
| H4 | Vùng mô hình dựa vào trùng vùng Fisher cao | ρ khối 4×4 = 0,88 trên 49 khối → **ủng hộ** |

Riêng về xác suất (để so với H1): RF không quá tự tin, 0% mẫu sai có p > 0,99, p trung bình của mẫu sai là 0,452, log-loss val 0,225.

## 2. Những điểm tinh chỉnh so với EDA (có số liệu)

### 2.1 Một ngưỡng là đủ: nhị phân hoá pixel > 0 (E1)

25 cây, 2 seed, mask none, m = √d:

| n_bins | val acc | OOB | thời gian (s) |
|---|---|---|---|
| **2** (pixel > 0) | **0,9607 ± 0,0015** | **0,9456** | 2,28 |
| 4 | 0,9608 ± 0,0006 | 0,9428 | 2,44 |
| 8 | 0,9599 ± 0,0012 | 0,9424 | 2,65 |
| 16 | 0,9595 ± 0,0001 | 0,9414 | 2,94 |
| 256 (mọi mức) | 0,9588 ± 0,0007 | 0,9400 | 3,07 |

EDA nói "chỉ cần ít ngưỡng". Thực tế chỉ cần **một** ngưỡng, đúng định nghĩa "có mực" của EDA mục 1.1: 4 bin chỉ cao hơn 0,01 điểm, dưới ngưỡng 2 × độ lệch chuẩn do seed (0,18 điểm), và OOB của bản nhị phân cao nhất. Thử cả 256 mức chậm hơn khoảng 1,35 lần và kém 0,19 điểm. Quantizer chia các mức khác 0 theo phân vị trên train, nên cũng dùng được cho số bin khác.

### 2.2 Bỏ pixel: crop 20×20 cũng không mất gì với RF (E2)

| mask | số đặc trưng | m | val acc | thời gian so với none |
|---|---|---|---|---|
| none | 784 | 28 | 0,9607 ± 0,0015 | 1,00 |
| dead | 712 | 26 | 0,9629 ± 0,0008 | 0,93 |
| rare | 596 | 24 | 0,9632 ± 0,0008 | 0,83 |
| crop24 | 576 | 24 | 0,9619 ± 0,0002 | 0,82 |
| **crop20** | **400** | **20** | **0,9628 ± 0,0014** | **0,65** |
| none, m = 24 (đối chứng) | 784 | 24 | 0,9605 ± 0,0002 | 0,96 |

EDA loại crop 20×20 vì nó chỉ giữ 96,66% lượng mực. Với RF, 3,34% mực ở viền không ảnh hưởng accuracy quá dao động do seed (kém rare 0,04 điểm, dưới ngưỡng 2 × 0,0010). Quy tắc chọn: trong các mặt nạ cách tốt nhất không quá 2 × độ lệch chuẩn do seed, chọn mặt nạ ít đặc trưng nhất, nên cấu hình cuối dùng crop20, cũng là mặt nạ huấn luyện nhanh nhất (0,65 lần thời gian của none).

Dòng đối chứng cho thấy lợi ích về thời gian **không phải chủ yếu vì m nhỏ hơn**: giữ 784 pixel nhưng hạ m xuống 24 chỉ còn 0,96 lần thời gian, còn crop24 (cũng m = 24) còn 0,82 lần. Lý do: ở nút sâu, phần lớn pixel viền là hằng trong nút; sklearn phải xét rồi bỏ qua chúng mới tìm đủ m pixel không hằng.

### 2.3 Số cây (E3, thí nghiệm siêu tham số có biểu đồ)

3 seed, crop20, 2 bin. Validation đạt 0,9506 ở T = 10, 0,9673 ở T = 100, 0,9687 ở T = 200; nằm trong 0,1 điểm của giá trị tại T = 200 từ T = 110. Script làm tròn điểm bão hoà lên bội số của 50, nên cấu hình cuối dùng **T = 150** (trồng 200 cây chỉ mất 14,3 s). Bỏ phiếu mềm và cứng cho đúng cùng accuracy ở mọi T: với pixel nhị phân và cây đầy đủ, hầu hết lá thuần nhất, nên phân phối lớp ở lá gần như luôn là one-hot.

OOB ở T nhỏ đánh giá thấp: tại T = 25, OOB thấp hơn val 1,5 điểm, vì mỗi mẫu OOB chỉ được khoảng 0,37·T cây bỏ phiếu (thực chất là một rừng 9 cây). Tới T = 200, OOB và val gần như bằng nhau (chênh 0,02 điểm). OOB đo ở các mốc T bằng `warm_start` (mỗi mốc chỉ trồng thêm cây mới); validation đo ở mọi T bằng `staged_accuracy`. Hình: `hp_e3_n_trees.png`.

### 2.4 Sức mạnh và tương quan giữa các cây (E4)

10 cây mỗi cấu hình, crop20 (d = 400):

| m | val acc | acc trung bình của 1 cây | tương quan lỗi giữa 2 cây |
|---|---|---|---|
| 1 | 0,9254 | 0,734 | 0,215 |
| 4 | 0,9382 | 0,790 | 0,247 |
| 24 (≈ √d) | 0,9499 | 0,838 | 0,293 |
| 96 | 0,9518 | 0,856 | 0,318 |
| 400 (bagging thuần) | 0,9339 | 0,860 | 0,405 |

Đúng lập luận của EDA 5.3: tăng m làm từng cây mạnh lên nhưng các cây giống nhau hơn; với m = d, tương quan lên 0,41 và rừng kém đỉnh 1,8 điểm dù từng cây mạnh hơn. Ứng viên m = 96 được xác nhận lại ở 50 cây × 2 seed: 0,9651 so với 0,9663 của √d, chậm hơn 2,4 lần, nên giữ √d.

### 2.5 Không cần giới hạn lá hay độ sâu (E5)

| cấu hình | train acc | val acc | val log-loss |
|---|---|---|---|
| **min_samples_leaf = 1** (cây đầy đủ) | 0,9999 | **0,9618** | 0,261 |
| min_samples_leaf = 3 | 0,9903 | 0,9575 | 0,279 |
| min_samples_leaf = 10 | 0,9644 | 0,9487 | 0,346 |
| min_samples_leaf = 30 | 0,9387 | 0,9287 | 0,453 |
| max_depth = 20 | 0,9990 | 0,9632 | 0,261 |
| max_depth = 15 | 0,9946 | 0,9593 | 0,255 |
| max_depth = 10 | 0,9573 | 0,9377 | 0,390 |

Giới hạn cây không giúp gì: max_depth = 20 chỉ hơn cây đầy đủ 0,14 điểm (dưới ngưỡng 2 × độ lệch chuẩn do seed), các giới hạn khác làm giảm accuracy. Trung bình nhiều cây đã giảm phương sai, nên cây đầy đủ là đúng cách dùng của Breiman (2001). Lo ngại về 116 pixel gần như chết đo được trực tiếp (đọc từ `est.tree_` của rừng giữ cả 784 pixel): các pixel này chỉ chiếm **0,09% số phép chia và 0,03% tổng độ quan trọng**; khi được dùng, trung vị nút có 33 mẫu và nút con nhỏ hơn có 1 mẫu, tức chỉ tách vài ảnh lẻ ở tầng rất sâu, không ảnh hưởng dự đoán.

### 2.6 Giải thích mô hình (E6, H4)

Bản đồ MDI và bản đồ độ quan trọng hoán vị theo khối 4×4 được đặt cạnh tần suất có mực (`interp_importance.png`); so với Fisher làm chung cho mọi mô hình ở `compare_models` (`cmp_block_importance.png` và mục H4 của `results/comparison/log.txt`). Hoán vị **cả khối** thay vì từng pixel, vì pixel kề bên (r ≈ 0,71) vẫn mang cùng thông tin, nên hoán vị một pixel đánh giá thấp độ quan trọng. Kích thước 4 xấp xỉ tầm tương quan (r ≈ 0,16 ở khoảng cách 3, ≈ 0 ở khoảng cách 5, EDA 2.1).

- **Theo vùng 4×4**: Spearman(hoán vị khối, Fisher khối) = 0,876 trên cả 49 khối (số đo chung, tính ở `compare_models` cho mọi mô hình). Hoán vị khối quan trọng nhất (ở giữa ảnh, quanh hàng 12–15, cột 12–15) làm accuracy val giảm 5,4 điểm.
- Bản đồ MDI cho thấy RF chia chủ yếu ở các pixel giữa ảnh, đúng vùng Fisher cao (EDA 3.3).

### 2.7 Đánh đổi khi chọn RF thay boosting (EDA 5.3)

EDA loại AdaBoost và Gradient Boosting vì boosting học tuần tự qua các vòng, và chấp nhận rằng boosting có thể chính xác hơn. `sklearn_reference` đo đánh đổi đó, cùng đặc trưng (crop20, nhị phân) và cùng 150 cây / vòng với RF cuối, siêu tham số cố định không tinh chỉnh (`results/sklearn_reference/ensemble_alternatives.json`):

| Mô hình | val acc | test acc | fit (s) | luồng |
|---|---|---|---|---|
| **Random Forest (cấu hình cuối)** | **0,9694** | **0,9668** | **7,3** | 14 |
| AdaBoost (SAMME, cây sâu 3, lr 0,5) | 0,9047 | 0,9141 | 480,1 | 1 |
| HistGradientBoosting (lr 0,1) | 0,9752 | 0,9781 | 149,9 | 14 |

- **Gradient Boosting chính xác hơn RF khoảng 1,1 điểm** (test 0,9781 so với 0,9668), đúng điều EDA đã thừa nhận. Đổi lại, cùng 14 luồng, nó huấn luyện lâu gấp hơn 10 lần RF (149,9 s so với 7,3 s ở lần đo này; số giây đổi theo lần đo và theo máy, xem README): mỗi vòng học 10 cây hồi quy và các vòng phải chạy tuần tự, còn các cây của RF độc lập nên trồng song song được. Với các thí nghiệm E1–E5 (khoảng 45 lần huấn luyện rừng), khoảng cách này quyết định thời gian dò siêu tham số; RF còn có sẵn OOB và đường accuracy theo số cây chỉ tăng rồi bão hoà, nên dễ chọn số cây.
- **AdaBoost kém cả hai** (0,9141): cây sâu 3 quá yếu cho 10 lớp trên đặc trưng nhị phân, và thuật toán chỉ chạy một luồng. Không tinh chỉnh nên đây là cận dưới, nhưng khoảng cách hơn 5 điểm cho thấy AdaBoost không phải phương án thay thế hợp lý.

### 2.8 Dùng sklearn mà vẫn phân tích được từng cây

- `make_random_forest` dựng `RandomForestClassifier(bootstrap=True, oob_score=True, random_state=seed)`; với
  `random_state` cố định, rừng giống hệt nhau dù chạy bao nhiêu luồng (`n_jobs`).
- `tree_votes` lấy phiếu của từng cây trong `rf.estimators_` (soft: phân phối lớp ở lá; hard: one-hot của lớp
  thắng); trung bình phiếu soft đúng bằng `rf.predict_proba`.
- `staged_accuracy` cộng dồn phiếu theo thứ tự cây, cho accuracy của rừng 1, 2, …, T cây sau một lần duyệt.
- `strength_correlation` tính accuracy trung bình của một cây và tương quan lỗi trung bình giữa hai cây.
- Độ sâu, số lá, số phép chia và số mẫu ở từng nút đọc từ `est.tree_` (`feature`, `weighted_n_node_samples`,
  `children_left`, `children_right`).
- Bốn hàm trên được đối chiếu với cách tính vét cạn trong `tests/check_random_forest.py`.

## 3. Pipeline

```
data/raw/*-ubyte
  └─ load_splits()                        src/data/preprocess.py     [0,1], 784, chia 50k/10k stratify seed 42
       └─ TreeFeatures (fit trên train)   src/data/features.py        (dùng chung: PixelMask, Quantizer)
            PixelMask("crop20")    giữ khung 20×20 ở giữa             → 400 pixel
            Quantizer(n_bins=2)    mã bin: 0 = không mực, 1 = có mực  → uint8
       └─ make_random_forest              src/models/random_forest.py  (sklearn RandomForestClassifier)
            T = 150, m = √d = 20, Gini, cây đầy đủ, bootstrap, soft voting, OOB
       └─ Pipeline                        src/models/pipeline.py      (dùng chung)
       └─ evaluate()                      src/evaluation/metrics.py   acc/P/R/F1/CM train, val, test, thời gian
          pair_errors, log_loss, fisher_scores, spearman, block_permutation_importance
                                          src/evaluation/analysis.py  (dùng chung cho mọi mô hình)
          tree_votes, staged_accuracy, strength_correlation   src/models/random_forest.py
  └─ plot_all() đọc summary.json, final_metrics.json → figures/   src/plots/random_forest.py, src/plots/common.py
```

Mask, n_bins, m, T, min_samples_leaf, max_depth được script chọn lại từ validation mỗi lần chạy; giá trị trên là kết quả của lần chạy này.

**Kết quả cuối** (huấn luyện 7,3 s với 14 luồng, kể cả tính OOB; OOB 0,9678):

| tập | accuracy | precision | recall | F1 macro | dự đoán (s) |
|---|---|---|---|---|---|
| train | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 0,65 |
| val | 0,9694 | 0,9693 | 0,9693 | 0,9693 | 0,14 |
| test | 0,9668 | 0,9666 | 0,9665 | 0,9666 | 0,15 |

Train = 1,0 là bình thường với cây đầy đủ; nên đọc OOB (0,968) như ước lượng lỗi tổng quát, khớp val và test.

## 4. Vị trí file

Phần dùng chung xem `README.md`, mục Cấu trúc. Các file riêng của RF:

| File | Nội dung |
|---|---|
| `src/models/random_forest.py` | `make_random_forest` (sklearn `RandomForestClassifier`), `tree_stats`, `tree_votes`, `staged_accuracy`, `strength_correlation` |
| `src/experiments/random_forest.py` | Thí nghiệm E0–E6: tính và ghi `log.txt`, `summary.json`, `final_metrics.json` |
| `src/plots/random_forest.py` | vẽ `figures/` từ các file đó (`python -m src.plots random_forest`) |
| `tests/check_random_forest.py` | 5 kiểm tra: phiếu soft trung bình bằng `predict_proba`, phiếu hard là one-hot, `staged_accuracy` khớp rừng từng cỡ, `strength_correlation` khớp vòng lặp theo cặp cây, `voting` sai báo lỗi |
| `results/random_forest/` | do script sinh: `log.txt`, `summary.json`, `final_metrics.json`, `figures/*.png`; `model.pkl` (hơn 100 MB, không đưa lên Git) do `final_models` ghi |

Dùng chung: `src/data/features.py` (`PixelMask`, `Quantizer`, `TreeFeatures`), `src/models/pipeline.py`,
`src/models/nearest_centroid.py`, `src/evaluation/analysis.py`, `src/evaluation/results_io.py`,
`src/plots/common.py`. Phần phân
tích và tâm gần nhất được đối chiếu ở `tests/check_analysis.py`, `check_models_common.py`. Baseline ghi bởi
`src/experiments/baseline.py`.

Lệnh chạy (từ gốc repo):

```
python tests/run_all.py
python -m src.experiments.baseline                       # một lần, nếu chưa có results/nearest_centroid/
python -m src.experiments.random_forest --quick          # lưới nhỏ, kiểm tra code chạy được
python -m src.experiments.random_forest                  # bản đầy đủ: khoảng 3,5 phút trên máy 14 lõi
```

**Số liệu trong tài liệu này lấy từ lần chạy đầy đủ ngày 08/11/2026** (`results/random_forest/log.txt`,
14 luồng). Accuracy, OOB và mọi lựa chọn cấu hình tái lập chính xác giữa các lần chạy (seed cố định, không
phụ thuộc số luồng); thời gian thì dao động giữa các lần chạy và giữa các máy.

## 5. Quy ước cài đặt

- Rừng là estimator của sklearn đặt cuối `Pipeline(TreeFeatures, RandomForestClassifier)`, cùng giao diện `fit` / `predict_proba` / `predict` với các mô hình tự cài; thuộc tính học được đọc từ `estimators_`, `feature_importances_`, `oob_score_`.
- Ngẫu nhiên qua `random_state` (`src/config.py: SEED`, hoặc seed của từng lần lặp); mọi bước tiền xử lý có `fit` / `transform` / `fit_transform`, tham số học từ train.
- Mỗi thí nghiệm đổi một yếu tố, chọn theo validation; khi chênh lệch nhỏ hơn 2 × độ lệch chuẩn do seed, chọn phương án rẻ hơn. Test dùng một lần.

## 6. Tài liệu tham khảo

1. L. Breiman, J. Friedman, R. Olshen, C. Stone, *Classification and Regression Trees*. Wadsworth, 1984.
2. L. Breiman, "Bagging predictors," *Machine Learning*, vol. 24, no. 2, pp. 123–140, 1996.
3. L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.
4. T. Hastie, R. Tibshirani, J. Friedman, *The Elements of Statistical Learning*, 2nd ed. Springer, 2009. Mục 9.2, chương 15.