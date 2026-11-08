# Quy ước thư mục `results/`

Mọi file trong `results/` do script sinh ra; không sửa tay. Chạy lại script là ghi đè.
Script thí nghiệm (`src/experiments/`) chỉ tính và ghi số liệu; hình trong `figures/` do `src/plots/<mô hình>.py` vẽ
từ `summary.json` và `final_metrics.json` (gọi ở cuối mỗi script; vẽ lại riêng: `python -m src.plots [mô hình ...]`).
Chỉ commit kết quả của lần chạy **đầy đủ** (không `--quick`); `compare_models` cảnh báo nếu gặp kết quả quick.

```
results/
├── README.md                     file này
├── nearest_centroid/             baseline (mốc trong mọi bảng): kết quả do `python -m src.experiments.baseline` ghi,
│                                 model.pkl do `final_models` ghi
├── naive_bayes/                  Naive Bayes
├── logistic_regression/          Logistic Regression (softmax)
├── mlp/                          MLP
├── random_forest/                Random Forest
├── cnn/                          CNN (tự cài NumPy)
│   ├── log.txt                   mọi bảng script in ra, theo thứ tự thí nghiệm E0, E1, ...
│   ├── summary.json              toàn bộ số liệu thí nghiệm (kể cả số liệu cho hình), cấu trúc tuỳ mô hình
│   ├── final_metrics.json        CHỈ mô hình cuối, cấu trúc CHUNG cho mọi mô hình (bên dưới)
│   ├── model.pkl                 mô hình cuối đã huấn luyện (src/experiments/final_models.py; RF không đưa lên Git)
│   └── figures/
│       ├── hp_<thí nghiệm>_<nội dung>.png       ảnh hưởng của siêu tham số   (rubric mục 3)
│       ├── eval_confusion.png                    ma trận nhầm lẫn train | val | test (bắt buộc)
│       ├── eval_confident_errors.png             20 lỗi tự tin nhất (mô hình có predict_proba)
│       ├── eval_learning_curve.png               đường học (mô hình huấn luyện theo epoch)
│       └── interp_<nội dung>.png                 khả năng giải thích        (rubric mục 4)
├── ensemble/                     do `python -m src.experiments.ensemble` sinh: diversity, soft voting, seed ensemble CNN
│                                 final_metrics.json (schema chung) là phương án chọn trên val ở E5; không có
│                                 block_importance; fit_time = tổng thời gian huấn luyện các thành viên;
│                                 runtime.n_jobs = luồng BLAS của các CNN, extra.rf_n_jobs = luồng của RF;
│                                 cnn_s0.pkl … cnn_s4.pkl: 5 CNN của seed ensemble (không có model.pkl)
├── torch_reference/              do `python -m src.experiments.torch_reference` sinh: CNN NumPy so với PyTorch
│                                 (torch chạy cùng số luồng với CNN NumPy)
├── sklearn_reference/            do `python -m src.experiments.sklearn_reference` sinh (rubric mục 2)
│   ├── log.txt
│   ├── summary.json              bản tự cài so với sklearn, cùng cấu hình (chạy một phần thì chỉ cập nhật các dòng đó);
│   │                             dòng LR có thêm "converged": LogisticRegression (lbfgs) giải tới hội tụ
│   └── ensemble_alternatives.json  RF so với AdaBoost, Gradient Boosting của sklearn (phương án bị loại ở EDA 5.3)
├── demo/                         do `python -m demo.robustness` sinh: độ bền của bước chuẩn hoá ảnh vẽ
│   ├── log.txt, summary.json     bảng độ bền trên 300 ảnh test vẽ lại to và lệch, ví dụ cho hình
│   └── figures/demo_pipeline.png các bước của demo/drawing.py và dự đoán của từng mô hình
└── comparison/                   do `python -m src.experiments.compare_models` sinh
    ├── log.txt                   bảng so sánh, trạng thái giả thuyết H1–H4, bảng sklearn
    ├── comparison.csv            một hàng cho mỗi mô hình × tập
    ├── summary.json              số liệu cho các hình cmp_*
    └── figures/
        ├── cmp_metrics.png       accuracy, F1 macro trên val và test
        ├── cmp_time.png          thời gian huấn luyện và dự đoán (thang log, kèm số luồng)
        ├── cmp_pair_errors.png   lỗi ở 5 cặp EDA 3.1 (H3)
        ├── cmp_per_class_f1.png
        └── cmp_block_importance.png  độ quan trọng theo khối 4×4 cạnh Fisher, CÙNG cách đo cho mọi mô hình (H4)
```

Tên thư mục mô hình là cố định (`src/config.py: MODEL_KEYS`); script so sánh tìm theo các tên này.

## Tên hình

| Tiền tố | Dùng cho | Ví dụ |
|---|---|---|
| `hp_` | thí nghiệm siêu tham số, có số thí nghiệm để tra lại trong `log.txt` | `hp_e2a_lr_momentum.png`, `hp_e3_n_trees.png` |
| `eval_` | đánh giá mô hình cuối | `eval_confusion.png`, `eval_confident_errors.png`, `eval_learning_curve.png` |
| `interp_` | giải thích mô hình | `interp_theta.png`, `interp_weights.png`, `interp_w1.png` |

Hai tên **bắt buộc**, vẽ bằng hàm chung trong `src/plots/common.py`: `eval_confusion.png` (`plot_confusion_splits`) và,
nếu mô hình có `predict_proba`, `eval_confident_errors.png` (`plot_confident_errors`).

## Định nghĩa chung (để so sánh các mô hình có cơ sở)

- **`fit_time`**: thời gian `pipeline.fit` trên `splits["train"]` với cấu hình cuối, GỒM tiền xử lý và những
  gì thuật toán cần (ví dụ tính val loss cho early stopping), KHÔNG GỒM phần chẩn đoán chỉ để vẽ/ghi log
  (loss và accuracy trên train mỗi epoch, đường OOB theo số cây). Thời gian có chẩn đoán ghi ở `extra` nếu cần.
  Với Random Forest, `fit_time` gồm cả accuracy OOB cuối (`oob_score=True` tính nó ngay trong `fit`).
- **`runtime.n_jobs`**: số luồng khi huấn luyện: luồng trồng cây của Random Forest (`--n-jobs`, mặc định mọi lõi)
  hoặc luồng BLAS (LR, MLP, CNN: `--blas-threads`, mặc định 8). Mô hình nào chạy song song phải ghi đúng con số
  này; `compare_models` in cảnh báo khi các mô hình khác nhau.
- **`predict_time`**: thời gian `pipeline.predict` trên cả tập (gồm tiền xử lý).
- **Độ đo xác suất**: `analysis.probabilistic_scores` (log-loss, tỉ lệ sai mà p > 0,99, độ tự tin trung bình).
- **Giải thích**: `block_importance` là số đo CHUNG cho mọi mô hình: `analysis.block_permutation_importance` trên val, khối 4×4,
  `PERM_REPEATS` lần, `SEED` (src/config.py). So với Fisher score (H4) chỉ làm ở `compare_models`, cho mọi mô
  hình cùng lúc; script của từng mô hình không tính Fisher. Mọi hệ số Spearman tính bằng `analysis.spearman`.
- **Chọn cấu hình**: chỉ trên val; test dùng đúng một lần cho mô hình cuối.

## `final_metrics.json`: cấu trúc chung

Ghi bằng `src.evaluation.results_io.save_final_metrics`. Mọi mô hình phải có các khoá dưới đây; khoá không áp
dụng thì để `null`.

```
{
  "model":        "logistic_regression",                 tên thư mục
  "display_name": "Logistic Regression",
  "quick":        false,                                 true nếu chạy --quick (bảng so sánh sẽ cảnh báo)
  "created_at":   "2026-11-01T12:30:00",
  "config":       {...},                                 siêu tham số của mô hình cuối (sklearn_reference đọc)
  "fit_time":     3.8,                                   giây, theo định nghĩa chung ở trên
  "runtime":      {"n_jobs": 1, "cpu_count": 8},
  "splits": {
    "train" | "val" | "test": {
      "n", "accuracy", "precision_macro", "recall_macro", "f1_macro", "predict_time",
      "per_class": {"precision": [10], "recall": [10], "f1": [10]},
      "confusion_matrix": [[10 × 10]],
      "probabilistic": {"log_loss", "overconfident_errors", "overconfident_share_of_errors",
                        "mean_confidence", "mean_confidence_errors"} | null
    }
  },
  "block_importance": [49 số] | null,
  "block_importance_kind": "accuracy giảm khi hoán vị khối 4×4 trên val",
  "extra": {...}                                        số riêng của mô hình (OOB, best_epoch, ...)
}
```

## Thứ tự chạy

```
python -m src.experiments.baseline                    # một lần
python -m src.experiments.naive_bayes
python -m src.experiments.logistic_regression
python -m src.experiments.random_forest
python -m src.experiments.mlp
python -m src.experiments.cnn
python -m src.experiments.torch_reference             # sau cnn
python -m src.experiments.sklearn_reference           # sau khi các mô hình đã chạy
python -m src.experiments.ensemble                    # sau khi các mô hình đã chạy (5 seed CNN), trước compare_models
python -m src.experiments.final_models                # lưu model.pkl và kiểm tái lập (khớp final_metrics.json)
python -m src.experiments.compare_models              # chạy lại mỗi khi có mô hình mới
python -m demo.robustness                             # sau final_models (cần model.pkl)
python -m src.plots                                   # (tuỳ chọn) vẽ lại mọi hình từ số liệu đã ghi
```
