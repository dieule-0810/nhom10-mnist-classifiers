# Pipeline gộp mô hình (soft voting): từ các mô hình đơn tới phương án cuối

Số liệu lấy từ `results/ensemble/log.txt` và `results/ensemble/final_metrics.json` (`python -m src.experiments.ensemble`, CPU 14 lõi). Mọi lựa chọn (tổ hợp, trọng số) làm trên validation; test chỉ dùng một lần ở E5.

Phần này khác Random Forest (phần ensemble bắt buộc, `random_forest_pipeline.md`): Random Forest gộp nhiều cây cùng loại bên trong một mô hình. Ở đây gộp **các mô hình cuối đã có** của nhóm, theo bước 4–5 của slide khuyến nghị bài 06: đo độ đa dạng, rồi gộp xác suất. Không có mô hình mới; chỉ CNN được huấn luyện lại với nhiều seed.

## 1. Thành viên và cách gộp

Thành viên dựng lại từ `config` trong `results/<key>/final_metrics.json`, đúng lớp trong `src/models/` (NB, LR, MLP, CNN tự cài; RF là `RandomForestClassifier` của sklearn). Bốn mô hình cổ điển cho lại đúng val acc đã ghi:

| Thành viên | Val acc | Huấn luyện lại |
|---|---|---|
| Random Forest | 0,9694 | 7 s (14 luồng) |
| Logistic Regression | 0,9197 | 3 s |
| Multinomial Naive Bayes | 0,8229 | dưới 1 s |
| MLP | 0,9831 | 276 s |
| CNN, seed 0–4 | 0,9855; 0,9864; 0,9891; 0,9879; 0,9870 (trung bình 0,9872) | 205–374 s mỗi mạng |

CNN seed 0 trùng với CNN cuối (`cnn_pipeline.md`). Seed chỉ đổi khởi tạo trọng số và thứ tự mini-batch; cấu hình giữ nguyên.

**Soft voting có trọng số:** P = Σ wᵢ Pᵢ / Σ wᵢ, dự đoán argmax P (`soft_vote`). Hard voting (đếm phiếu, `hard_vote`) chỉ để so ở E3.

## 2. Thí nghiệm

### 2.1 Độ đa dạng (E1, `div_pairs.png`)

Với mỗi cặp, "% cùng sai" = số ảnh val cả hai cùng sai / số ảnh có ít nhất một mô hình sai (Kuncheva & Whitaker, 2003).

| Cặp | % cùng sai |
|---|---|
| NB – CNN | 4,7–5,9% |
| LR – CNN | 10,0–13,3% |
| RF – CNN | 20,1–26,6% |
| MLP – CNN | 31,7–39,4% |
| CNN – CNN (hai seed) | 33,5–39,1% |
| RF – MLP | 34,6% |
| LR – NB | 32,0% |

Mọi cặp đều dưới ngưỡng 60% của slide (trên ngưỡng thì gộp gần như không thêm thông tin). Ba CNN seed 0–2: 232 ảnh có ít nhất một mạng sai, 56 ảnh (24,1%) cả ba cùng sai, gần con số 20,0% trên slide 11. Hai CNN khác seed sai trùng nhau nhiều hơn CNN với mô hình cổ điển (ngang MLP – CNN), nhưng vẫn chỉ khoảng một phần ba.

### 2.2 Gộp các mô hình cổ điển (E2)

Soft voting cùng trọng số, so với mô hình tốt nhất trong tổ hợp (val):

| Tổ hợp | Val acc | So với mô hình tốt nhất |
|---|---|---|
| **RF + MLP** | **0,9826** | −0,05 (MLP 0,9831) |
| LR + MLP | 0,9763 | −0,68 |
| RF + LR + MLP | 0,9773 | −0,58 |
| RF + LR | 0,9460 | −2,34 (RF 0,9694) |
| RF + LR + NB + MLP | 0,9572 | −2,59 |
| NB + MLP | 0,9120 | −7,11 |

(11 tổ hợp, đủ ở `log.txt`.) Không tổ hợp nào hơn thành viên tốt nhất; gần nhất là RF + MLP, kém MLP 5 ảnh. **Đa dạng không đủ:** NB sai rất khác CNN và MLP (% cùng sai thấp nhất), nhưng accuracy thấp hơn 16 điểm và quá tự tin (10% mẫu sai có p > 0,99, log-loss 2,26), nên kéo trung bình xác suất về phía sai. Cùng trọng số thì thành viên yếu làm hỏng thành viên mạnh.

### 2.3 Seed ensemble CNN (E3, `ens_seeds.png`)

| Số mạng k | Soft (k mạng đầu) | Soft, trung bình mọi tổ hợp k mạng (min–max) | Hard (k mạng đầu) |
|---|---|---|---|
| 1 | 0,9855 | 0,9872 (0,9855–0,9891) | 0,9855 |
| 2 | 0,9887 | 0,9897 (0,9883–0,9908) | 0,9886 |
| 3 | 0,9902 | 0,9904 (0,9893–0,9911) | 0,9902 |
| 4 | 0,9911 | 0,9908 (0,9900–0,9911) | 0,9912 |
| 5 | 0,9912 | 0,9912 | 0,9916 |

- 5 mạng hơn mạng tốt nhất 0,21 điểm và hơn trung bình từng mạng 0,40 điểm (40 ảnh val). Đây là phần lợi lớn nhất và chắc nhất của cả phần gộp.
- Lợi ích giảm dần: thêm mạng thứ 2 được +0,32 điểm, mạng thứ 3 +0,15, thứ 4 +0,09, thứ 5 +0,01.
- Soft và hard gần như ngang nhau, chênh không quá 4 ảnh ở mọi k; ở k = 4 và 5 hard còn nhỉnh hơn (1 và 4 ảnh). Về lý thuyết hard voting bỏ mất độ tự tin (một mạng rất chắc đúng thua hai mạng hơi nghiêng về nhãn sai), nhưng với các CNN đã khá đồng thuận, khác biệt này không thấy được trên accuracy. Soft vẫn được dùng vì cho xác suất để đo log-loss.

### 2.4 Trộn CNN ensemble với mô hình cổ điển (E4, `ens_weights.png`)

CNN ensemble (5 mạng, trung bình xác suất) nhận trọng số w, thành viên cổ điển trọng số 1; w chọn trên val trong {1, 2, 3, 5, 8}:

| Phương án | w = 1 | 2 | 3 | 5 | 8 | Chọn |
|---|---|---|---|---|---|---|
| CNN ens (một mình) | | | | | | 0,9912 |
| **+ RF** | 0,9901 | 0,9909 | 0,9911 | 0,9912 | **0,9913** | **w = 8** |
| + LR | 0,9837 | 0,9893 | 0,9907 | 0,9909 | 0,9912 | w = 8 |
| + NB | 0,9102 | 0,9878 | 0,9900 | 0,9907 | 0,9909 | w = 8 |
| + MLP | 0,9886 | 0,9906 | 0,9911 | 0,9908 | 0,9910 | w = 3 |
| + RF + MLP | 0,9881 | 0,9899 | 0,9905 | 0,9907 | 0,9909 | w = 8 |

Cùng trọng số (w = 1) thì mọi thành viên cổ điển làm CNN ensemble kém đi; chỉ khi CNN chiếm phần lớn trọng số mới ngang hoặc hơn, và lợi tối đa 1 ảnh. Phương án tốt nhất trên val: **CNN ensemble × 8 + RF**, val 0,9913; w = 8 nằm ở biên lưới, tức thành viên cổ điển càng ít trọng số càng tốt.

## 3. Kết quả cuối (E5)

Bảng tổng kết trên test, mọi cấu hình đã chốt trên val trước:

| Phương án | Val acc | Test acc | Ảnh sai (test) | Log-loss (test) |
|---|---|---|---|---|
| Naive Bayes | 0,8229 | 0,8356 | 1644 | 2,0074 |
| Logistic Regression | 0,9197 | 0,9238 | 762 | 0,2761 |
| Random Forest | 0,9694 | 0,9668 | 332 | 0,2180 |
| MLP | 0,9831 | 0,9836 | 164 | 0,0537 |
| CNN đơn (seed 0) | 0,9855 | 0,9882 | 118 | 0,0391 |
| Soft voting RF + MLP | 0,9826 | 0,9841 | 159 | 0,1170 |
| CNN ensemble 5 seed | 0,9912 | 0,9915 | 85 | 0,0249 |
| RF + CNN ens, cùng trọng số | 0,9901 | 0,9907 | 93 | 0,1053 |
| **RF + CNN ens, CNN × 8 (chọn)** | **0,9913** | **0,9915** | **85** | 0,0410 |

Độ đo đầy đủ của phương án chọn (`final_metrics.json`, `compare_models`):

| Tập | Accuracy | F1 macro | Log-loss | Sai mà p > 0,99 | Độ tự tin TB |
|---|---|---|---|---|---|
| train | 0,9993 | 0,9993 | 0,0156 | 0 | 0,986 |
| val | 0,9913 | 0,9913 | 0,0472 | 0,01% | 0,971 |
| test | 0,9915 | 0,9915 | 0,0410 | 0 | 0,971 |

**Đọc kết quả, kể cả thiên lệch khi chọn trên val.**
- Phương án cuối được chọn là cấu hình có val acc cao nhất trong khoảng 40 cấu hình thử ở E2–E4, nên val acc 0,9913 của nó lạc quan hơn một ước lượng không thiên lệch. Test (dùng một lần) cho 0,9915, không thấp hơn val, nên không thấy dấu hiệu chọn quá khớp val.
- Nhưng các phương án đứng đầu cách nhau quá ít để xếp hạng: CNN ensemble một mình kém phương án chọn 1 ảnh trên val và bằng đúng trên test (cùng 85 ảnh sai). Kết luận chắc chắn là **seed ensemble CNN** (118 → 85 ảnh sai trên test, −28%); thêm RF với trọng số nhỏ không có lợi phân biệt được với dao động, và còn làm xác suất kém đi: log-loss test 0,0410 so với 0,0249, độ tự tin trung bình 0,971 trong khi accuracy 0,992, vì RF thiếu tự tin kéo xác suất về giữa.
- So với CNN đơn trên val: 145 → 87 lỗi (−40%), 5 cặp EDA 3.1: 42 → 26 lỗi. Lỗi còn lại vẫn có 29,9% ở 5 cặp; top 5 cặp nhầm: 4–9, 8–9, 7–9, 2–7, 5–8. `eval_residual_errors.png` vẽ 20 ảnh test bị đoán sai với xác suất cao nhất: phần lớn mơ hồ cả với người (số 6 viết như số 1, số 5 như số 3, số 6 nét hở như số 0); riêng hai số 7 có gạch ngang giữa thân đều bị đoán là 2.

**Chi phí.** Huấn luyện cộng dồn 6 thành viên 1452 s (RF 7 s + 5 CNN), so với 273 s của CNN đơn; dự đoán 10.000 ảnh 14,5 s so với 2,9 s (gấp 5,0 lần). Không có bản đồ độ quan trọng: hoán vị khối 4×4 cho 5 CNN quá đắt, và trung bình 6 mô hình khó giải thích hơn từng mô hình.

**Trong demo.** Giao diện demo dùng 5 CNN không kèm RF (lựa chọn "CNN × 5", `demo/models.py`): trên test bằng đúng phương án chọn (85 ảnh sai), log-loss tốt hơn (0,0249 so với 0,0410), và không cần file RF (~220 MB, không có trong Git). Trên 300 ảnh test vẽ lại to và lệch (`python -m demo.robustness`): acc ảnh gốc 0,990, qua `to_mnist` 0,987, giữ 0,993 dự đoán, thu nhỏ thẳng 0,593 (một CNN: 0,983 / 0,987 / 0,993 / 0,540).

## 4. Pipeline

```
results/<key>/final_metrics.json ("config" của RF, LR, NB, MLP, CNN)
  └─ BUILDERS[key](cfg) / make_cnn(cfg, seed)        src/experiments/ensemble.py, src/experiments/cnn.py
       └─ fit_proba: huấn luyện trên train, predict_proba trên train / val / test
  └─ E1 co_error (% cùng sai)  E2–E4 soft_vote(Ps, weights) trên val   E5 test một lần
  └─ save_final_metrics (schema chung, để compare_models xếp cạnh mô hình đơn)   src/evaluation/
  └─ save_model: 5 CNN → results/ensemble/cnn_s<seed>.pkl                        src/evaluation/results_io.py
  └─ plot_all() đọc summary.json, final_metrics.json → figures/   src/plots/ensemble.py, src/plots/common.py
```

`src/experiments/final_models.py` (`ensemble_proba`) dùng lại đúng cách gộp này để nạp / huấn luyện lại phương án cuối cho `notebooks/results.ipynb`.

## 5. Vị trí file

| File | Nội dung |
|---|---|
| `src/experiments/ensemble.py` | `BUILDERS`, `fit_proba`, `soft_vote`, `hard_vote`, `co_error`, E1–E5: tính và ghi số liệu |
| `src/plots/ensemble.py` | vẽ `figures/` từ `summary.json`, `final_metrics.json` (`python -m src.plots ensemble`) |
| `results/ensemble/` | `log.txt`, `summary.json`, `final_metrics.json`, `cnn_s0.pkl` … `cnn_s4.pkl`, `figures/` (`div_pairs`, `ens_seeds`, `ens_weights`, `eval_confusion`, `eval_residual_errors`) |

Lệnh chạy (từ gốc repo, sau khi đã có kết quả các mô hình): `python -m src.experiments.ensemble` (30 phút trên CPU 14 lõi); `--quick` để thử nhanh.

## 6. Quy ước

- Chỉ dùng mô hình cuối của các phần trước (cùng lớp, cùng cấu hình); không có mô hình nào chỉ có trong ensemble.
- Mọi lựa chọn trên val; test chỉ ở E5 (bẫy chọn trên test, slide 20 bài 06).
- `fit_time` của ensemble là tổng thời gian huấn luyện các thành viên chạy lần lượt; so thời gian với mô hình đơn phải kèm điều kiện này.

## 7. Tài liệu tham khảo

- L. I. Kuncheva, C. J. Whitaker, "Measures of diversity in classifier ensembles," *Machine Learning*, vol. 51, pp. 181–207, 2003.
- B. Lakshminarayanan, A. Pritzel, C. Blundell, "Simple and scalable predictive uncertainty estimation using deep ensembles," *NeurIPS*, 2017.
