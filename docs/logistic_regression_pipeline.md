# Pipeline Logistic Regression: từ EDA tới thí nghiệm

Mọi số liệu EDA lấy từ `eda/notebooks/eda.ipynb` (50.000 ảnh `splits["train"]`). Số liệu thí nghiệm lấy từ `results/logistic_regression/log.txt`, chạy trên validation. Test chỉ dùng một lần ở E5.

## 1. Những gì EDA nói về Logistic Regression

### 1.1 Cách xử lý EDA đề xuất

| Phát hiện EDA | Mục | Ảnh hưởng tới LR | Cách xử lý EDA đề xuất | Bản cài đặt |
|---|---|---|---|---|
| Pixel uint8, cộng trừ bị tràn | 0 | Trừ mean trên uint8 cho kết quả sai | Luôn dùng float32 sau `to_unit_range` | `load_splits` |
| 72 dead pixel (std = 0), 71 ở 2 vòng ngoài | 1.1 | Không có thông tin; z-score từng pixel sẽ chia cho 0 | Chuẩn hoá vô hướng (1 mean, 1 std), fit trên train | `ScalarStandardizer`; tuỳ chọn `PixelMask("dead")` |
| 116 pixel gần như chết (có mực ở 1–47 ảnh) | 1.1 | Học từ vài chục ảnh nên dễ khớp nhiễu | L2 | `lam` trong `SoftmaxRegressionNP`; tuỳ chọn `PixelMask("rare")` |
| 99,76% mực nằm trong khung 24×24 | 1.3 | Bớt trọng số | Ứng viên crop 24×24 | `PixelMask("crop24")` |
| Pixel kề nhau tương quan mạnh (r = 0,708), vẫn còn khi đã biết lớp (0,64–0,86) | 2.1, 2.2 | Trọng số không ổn định, ảnh trọng số lốm đốm; điều kiện số xấu nên GD hội tụ chậm | L2; theo dõi loss theo vòng lặp để chọn η và số vòng | `loss_`, `history_`; E2, E3 |
| Hiệp phương sai các lớp khác nhau: tổng phương sai trong lớp 22,4–50,8 | 1.2 | Ranh giới tuyến tính có trần | So với RF, MLP (H2) | `final_metrics.json`, `compare_models` |
| Tâm lớp gần nhau: 4–9, 7–9, 5–8, 3–5, 3–8 | 3.1 | Các cặp có nguy cơ nhầm nhất | Theo dõi riêng trong ma trận nhầm lẫn | `pair_errors`, `share_in_pairs` |
| Fisher dồn vào 129 pixel ở giữa ảnh | 3.3 | Ảnh trọng số nên trùng vùng Fisher cao | Đặt ảnh trọng số cạnh bản đồ Fisher | `fisher_scores`, E5 |
| Tâm lớp tách nhau rõ (2,48 đến 7,56) | 3.1 | Có cơ sở cho mốc tâm gần nhất | Mốc trong mọi bảng | `NearestCentroidNP` |
| LR cùng dạng hàm softmax(Wx + b) với MNB | 5.1, 5.2 | So sánh chỉ đổi một yếu tố | Cùng đầu vào pixel với MNB | `make_features(mask, "scalar")` |
| Chọn LR thay SVM vì cần xác suất | 5.2 | Cần log-loss, mức tự tin | Softmax đa lớp, không dùng one-vs-rest | `SoftmaxRegressionNP`, `probabilistic_scores` |

### 1.2 Giả thuyết liên quan tới LR (EDA 5.6) và kết quả

Kết luận cuối của từng giả thuyết ở `results/comparison/log.txt` (`compare_models`); bảng dưới chỉ ghi phần LR đóng góp.

| # | Giả thuyết | Vai trò của LR | Số liệu của LR (val) |
|---|---|---|---|
| H1 | LR chính xác hơn và cho xác suất đáng tin hơn MNB | Một vế | acc 0,920; log-loss 0,291; 0,19% mẫu sai mà p > 0,99; độ tự tin TB 0,905 (MNB: 0,823; 2,259; 10,35%; 0,978) → **ủng hộ** |
| H2 | RF, MLP vượt LR | Mốc so sánh | acc 0,920, F1 macro 0,919 |
| H3 | Lỗi của mô hình tuyến tính dồn vào 5 cặp tâm gần nhau; mô hình phi tuyến giảm chính các lỗi này | Vế (a) và mốc của vế (b) | 5 cặp chiếm 36,7% lỗi (rải đều: 11,1%), đúng 5 cặp nhầm nhiều nhất → vế (a) **đạt** |
| H4 | Vùng mô hình dựa vào trùng vùng Fisher cao (hoán vị khối 4×4) | Kiểm chứng | ρ khối = 0,91 (cao nhất trong bốn mô hình bắt buộc) → **ủng hộ**, kèm insight: đa cộng tuyến (EDA 2) làm trọng số loang giữa các pixel kề nhau, nên phải đọc theo vùng chứ không theo từng pixel |

## 2. Những điểm tinh chỉnh so với EDA (có số liệu đi kèm)

1. **Chuẩn hoá vô hướng có cơ sở định lượng (E1).** Khi chuẩn hoá từng pixel, max|z| trên val là 1314, so với 2,7 khi chuẩn hoá vô hướng. Accuracy chỉ kém 0,3 điểm (0,918 so với 0,921), nhưng val log-loss tệ hơn (0,298 so với 0,288) và tỉ lệ sai mà p > 0,99 cao hơn (0,24% so với 0,18%). Như vậy cái hại EDA dự báo hiện ra chủ yếu ở mức tự tin của mô hình, không phải ở accuracy.
2. **Ba điểm lập luận cần chính xác** (EDA ghi theo đúng các điểm này): (a) sau chuẩn hoá vô hướng, dead pixel không có "gradient bằng 0" mà thành một hằng số khác 0, tức một cột bias thừa vẫn được cập nhật và bị L2 phạt (EDA 1.1; E4: bỏ hay giữ 72 cột này không đổi accuracy); (b) PCA không whiten chỉ là phép xoay nên không cải thiện điều kiện số, chỉ việc cắt các hướng có trị riêng gần 0 mới cải thiện (vì vậy EDA không dùng PCA); (c) với Multinomial NB, bỏ dead pixel đổi mẫu số làm trơn α·d nên không hoàn toàn "không đổi kết quả", chỉ là ảnh hưởng không đáng kể.
3. **Kích thước batch được chọn trên validation (E2c).** Mỗi batch dò η riêng (batch lớn cần η lớn hơn). Mọi batch từ 32 đến 1024 cho val loss trong khoảng 0,291–0,302, và batch 128 cho val loss nhỏ nhất (0,291); full-batch kém rõ rệt (0,643 sau 20 epoch, vì mỗi epoch chỉ có một bước cập nhật). Quy tắc: giữ batch mặc định 128 trừ khi batch khác tốt hơn vượt ngưỡng 0,002, nên batch 128 được giữ.
4. **"Điều kiện số xấu" thấy được bằng mắt (E2b).** Full-batch GD với η = 0,2 dao động mạnh ở 10 epoch đầu (J lên tới khoảng 1,8). Momentum (α = 0,9) đưa J xuống 0,257 sau 100 epoch, còn mini-batch 128 với α = 0,9, η = 0,01 đạt 0,258 ngay sau 10 epoch.
5. **Hiệu quả của L2 (E3).** Từ λ = 0 tới 1e-3, val acc gần như không đổi (0,922 → 0,920) và val loss nhỏ nhất ở λ = 1e-3, độ lốm đốm của ảnh trọng số giảm 15% và ‖W‖ giảm từ 11,1 xuống 6,7. λ = 1e-2 cho ảnh trọng số mượt rõ rệt nhưng mất thêm 0,5 điểm accuracy. Hình `hp_e3_weights_by_lambda.png` minh hoạ dự báo "trọng số lốm đốm" ở EDA mục 2.
6. **Đo độ quan trọng của pixel đúng cách cho softmax (H4).** Cộng cùng một hằng số vào w_kp của mọi lớp không đổi dự đoán, nên |w_kp| của riêng một lớp không phải số đo hợp lệ. Bản cài đặt dùng std của w_kp theo lớp.
7. **Quy tắc chọn cấu hình.** Khi val loss của các cấu hình chênh nhau dưới 0,002 (cỡ dao động giữa các seed), chọn cấu hình ít đặc trưng nhất. Theo quy tắc này, cấu hình cuối dùng mask `crop24` (576 pixel, bỏ hai vòng ngoài cùng, nơi có 71 trong 72 dead pixel; EDA 1.3).
8. **Đối chiếu sklearn cùng điều kiện dừng (`sklearn_reference`).** `LogisticRegression` của sklearn không có mini-batch, momentum hay dừng sớm (lbfgs giải tới hội tụ), nên bản đối chiếu dùng `MLPClassifier(hidden_layer_sizes=())`: mạng không lớp ẩn, đầu ra softmax, tức đúng softmax regression. Cấu hình trùng bản tự cài: SGD, batch 128, η = 0,01, momentum 0,9 (không Nesterov), L2 `alpha = λ·batch` (sklearn chia phạt cho cỡ batch), và huấn luyện đúng 13 epoch (epoch mà bản tự cài giữ lại khi dừng sớm). Kết quả: val 0,9182 so với 0,9197, test 0,9167 so với 0,9238 (chênh 0,71 điểm, tức 71 ảnh). Khoảng cách trên test lớn hơn trên val vì với momentum 0,9 nghiệm sau từng epoch còn dao động quanh cực tiểu, và hai bản khác khởi tạo (Glorot so với W = 0) cùng dãy trộn mini-batch. Khi cho sklearn giải tới hội tụ (`LogisticRegression`, lbfgs, C = 1/(λN) = 0,02, cùng hàm mục tiêu, trên cùng đặc trưng crop24; dòng `lbfgs` của `sklearn_reference`, khoá `converged`) thì val 0,9216, test 0,9238, đúng bằng bản tự cài: dừng sớm theo val loss không làm mất accuracy test so với điểm cực tiểu của J. Nghiệm tối ưu khi chạy đủ lâu được đối chiếu với `LogisticRegression` trong `tests/check_logistic_regression.py`.

## 3. Pipeline

```
data/raw/*.idx-ubyte
  └─ load_splits()                          src/data/preprocess.py   [0,1], 784, chia 50k/10k stratify seed 42
       └─ FeaturePipeline (fit trên train)  src/data/features.py
            PixelMask("crop24")    bỏ 2 vòng pixel ngoài cùng      → 576
            ScalarStandardizer     (x − μ)/σ, μ, σ vô hướng
       └─ SoftmaxRegressionNP               src/models/logistic_regression.py
            mini-batch 128, η = 0,01, α = 0,9, λ = 1e-3, early stopping (patience 8) trên val
       └─ Pipeline                          src/models/pipeline.py          (dùng chung)
       └─ evaluate()                        src/evaluation/metrics.py       acc/P/R/F1/CM, thời gian
          probabilistic_scores, top_confused_pairs, fisher_scores, spearman, block_permutation_importance
                                            src/evaluation/analysis.py      (dùng chung cho mọi mô hình)
```

Các tham số η, α, λ, mask được script chọn lại từ validation mỗi lần chạy; giá trị ở trên là kết quả của lần chạy này.

## 4. Vị trí file

Phần dùng chung xem `README.md`, mục Cấu trúc. Các file riêng của LR:

| File | Nội dung |
|---|---|
| `src/models/logistic_regression.py` | `SoftmaxRegressionNP` (kế thừa `BaseClassifier`; `add_bias`, `log_softmax`, ... lấy từ `src/models/_numeric.py`) |
| `src/experiments/logistic_regression.py` | Thí nghiệm E0–E5 (E2a, E2c, E2b, E3, …): tính và ghi số liệu |
| `src/plots/logistic_regression.py` | vẽ `figures/` từ `summary.json`, `final_metrics.json` (`python -m src.plots logistic_regression`) |
| `tests/check_logistic_regression.py` | 2 kiểm tra: gradient so với sai phân, nghiệm tối ưu so với sklearn |
| `results/logistic_regression/` | do script sinh ra: `log.txt`, `summary.json`, `final_metrics.json`, `model.pkl`, `figures/hp_*`, `eval_*`, `interp_*` |

Dùng chung: `src/data/features.py` (`PixelMask`, `ScalarStandardizer`, `PixelStandardizer`,
`make_features`), `src/models/pipeline.py`, `src/models/nearest_centroid.py`, `src/evaluation/analysis.py`,
`src/evaluation/results_io.py`, `src/plots/common.py`. Phần phân tích và tâm gần nhất được đối chiếu ở `tests/check_analysis.py`,
`check_models_common.py`. Baseline ghi bởi `src/experiments/baseline.py`.

Lệnh chạy (từ gốc repo):

```
python tests/run_all.py
python -m src.experiments.baseline                      # một lần, nếu chưa có results/nearest_centroid/
python -m src.experiments.logistic_regression --quick   # khoảng 40 giây, kiểm tra code chạy được
python -m src.experiments.logistic_regression           # bản đầy đủ
python -m src.experiments.compare_models                # so sánh mọi mô hình đã có final_metrics.json
```

**Số liệu trong tài liệu này lấy từ lần chạy đầy đủ trong repo** (`results/logistic_regression/log.txt`,
khoảng 4 phút). Accuracy, loss và mọi lựa chọn cấu hình tái lập chính xác giữa các lần chạy (seed cố định); thời
gian thì dao động giữa các lần chạy và giữa các máy.

## 5. Quy ước cài đặt

- Bias là thành phần cuối của W (`add_bias`, x̃ = [x; 1]); L2 không phạt bias (mặt nạ `m = [1, …, 1, 0]`).
- Mỗi mô hình có `fit`, `decision_function`, `predict_proba`, `predict`; lưu lịch sử loss trong `loss_`; khởi tạo W = 0 (vì J lồi).
- Có `tol` để dừng khi |J(t) − J(t−1)| < tol; momentum Δw = αΔw − η∇J; `batch_size=None` cho đúng GD full-batch.
- Mỗi bước tiền xử lý có `fit` / `transform` / `fit_transform`. Mọi tham số học từ train.
- Log-softmax trừ max trước khi lấy exp; gradient được kiểm chứng bằng sai phân trung tâm.
- Mỗi thí nghiệm đổi một yếu tố, chọn theo validation, test dùng một lần.
- sklearn chỉ xuất hiện trong `tests/` và `src/experiments/sklearn_reference.py` để đối chiếu.