# MNIST Classifiers — Đồ án 1: Nhận dạng chữ số viết tay

So sánh các bộ phân lớp trên tập MNIST: Naïve Bayes, bộ phân lớp tuyến tính (Gradient Descent), MLP (backpropagation) và CNN tự cài đặt bằng NumPy, cùng phương pháp Ensemble Random Forest (scikit-learn).

Các lệnh khác nhau giữa hệ điều hành được ghi riêng cho **Windows (PowerShell)** và **macOS/Linux (Terminal)**.
Các lệnh `python ...` còn lại giống nhau trên mọi hệ điều hành, chạy sau khi đã kích hoạt môi trường ảo.

## Cài đặt

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```
> Nếu PowerShell báo lỗi không được chạy script, chạy một lần
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` rồi kích hoạt lại.
> Dùng Command Prompt (cmd) thì kích hoạt bằng `.venv\Scripts\activate.bat`.

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Dữ liệu

Dự án sử dụng tập dữ liệu MNIST (LeCun et al., 1998), lấy từ Kaggle:
https://www.kaggle.com/datasets/hojjatk/mnist-dataset

4 file cần dùng đã có sẵn trong repo, clone về là chạy được, không cần tải:
```
data/raw/
├── train-images.idx3-ubyte   # 60.000 ảnh huấn luyện
├── train-labels.idx1-ubyte   # 60.000 nhãn huấn luyện
├── t10k-images.idx3-ubyte    # 10.000 ảnh kiểm thử
└── t10k-labels.idx1-ubyte    # 10.000 nhãn kiểm thử
```

Muốn tải lại từ Kaggle (ví dụ khi file bị hỏng), chạy từ thư mục gốc repo:
```bash
python -m src.data.download --force
```
Script tải bằng gói `kaggle` (đã có trong `requirements.txt`), chỉ giữ 4 file trên và đọc thử để kiểm tra. Bộ dữ
liệu là công khai nên không cần tài khoản Kaggle: dòng "Authentication required..." mà `kaggle` in ra có thể bỏ
qua. Chỉ khi tải thất bại vì lỗi xác thực mới cần chạy `kaggle auth login` rồi thử lại.

## Chạy thí nghiệm

Chạy từ thư mục gốc repo (thứ tự và ý nghĩa từng file kết quả: [`results/README.md`](results/README.md)):

```bash
python -m src.experiments.baseline                    # mốc tâm gần nhất, chạy một lần
python -m src.experiments.naive_bayes                 # thêm --quick để thử nhanh
python -m src.experiments.logistic_regression         # thêm --quick để thử nhanh
python -m src.experiments.random_forest               # thêm --quick để thử nhanh
python -m src.experiments.mlp                         # thêm --quick để thử nhanh
python -m src.experiments.cnn                         # thêm --quick để thử nhanh
python -m src.experiments.torch_reference             # đối chiếu CNN với PyTorch
python -m src.experiments.sklearn_reference           # đối chiếu sklearn trên MNIST
python -m src.experiments.ensemble                    # sau các mô hình; --quick để thử nhanh
python -m src.experiments.final_models                # huấn luyện mô hình cuối, ghi results/<mô hình>/model.pkl
python -m src.experiments.compare_models              # bảng và hình so sánh mọi mô hình đã chạy
python -m demo.robustness                             # độ bền của bước chuẩn hoá ảnh vẽ (cần model.pkl)
```

LR, MLP, CNN và ensemble giới hạn số luồng BLAS bằng `--blas-threads` (mặc định 8); Random Forest dùng mọi lõi
(`--n-jobs`). Con số này được ghi vào `final_metrics.json`, để bảng thời gian so được giữa các mô hình. Chạy lần lượt từng lệnh (không chạy song song),
nếu không thời gian huấn luyện đo được sẽ sai. Toàn bộ mất khoảng 2 giờ trên CPU.

## Vẽ lại hình

Mỗi script thí nghiệm ghi mọi số liệu (kể cả số liệu cho hình) vào `results/<mô hình>/summary.json` và
`final_metrics.json`, rồi gọi `src/plots/<mô hình>.py` để vẽ `figures/`. Vẽ lại hình mà không chạy lại thí nghiệm:

```bash
python -m src.plots                      # mọi mô hình đã có kết quả
python -m src.plots random_forest cnn    # chỉ vài mô hình (tên thư mục trong results/)
```

## Tái lập kết quả

Kết quả của lần chạy đầy đủ (`results/`) và mô hình cuối đã huấn luyện (`results/<mô hình>/model.pkl`) có sẵn
trong Git.
Notebook tái lập nạp các mô hình này, dự đoán lại trên train / val / test và **tính lại** các con số của báo cáo,
kèm cột "khớp" so với số đã ghi:

```bash
jupyter lab notebooks/results.ipynb      # rồi Run All
```

| Cách chạy | Làm gì | Thời gian |
|---|---|---|
| `RETRAIN = False` (mặc định, ô đầu notebook) | nạp mô hình đã lưu, dự đoán lại, tính lại độ đo, ma trận nhầm lẫn, H1–H3 | vài phút |
| `RETRAIN = True` | huấn luyện lại mọi mô hình cuối theo cấu hình đã chốt, lưu lại, rồi như trên | ~35–40 phút |
| `python -m src.experiments.final_models` | giống `RETRAIN = True`, không cần Jupyter; in "khớp / LỆCH" cho từng mô hình | ~35–40 phút |
| các lệnh ở mục Chạy thí nghiệm | chạy lại cả phần dò siêu tham số, sinh lại toàn bộ `results/` | ~2 giờ |

- Random Forest đã lưu nặng khoảng 220 MB (quá giới hạn 100 MB của GitHub) nên không có trong Git: lần chạy notebook đầu tiên tự
  huấn luyện lại nó (vài giây), cho đúng cùng một rừng.
- Tái lập đúng từng ô ma trận nhầm lẫn cần cùng seed (`src/config.py`) **và** cùng số luồng BLAS (8): đổi số luồng
  làm đổi thứ tự cộng số thực, MLP và CNN sẽ huấn luyện ra trọng số hơi khác (accuracy lệch ở chữ số thứ tư).
- Kết quả trong repo chạy trên CPU Intel 14 lõi, Windows 11, Python 3.14, numpy 2.x, scikit-learn 1.9, torch 2.12
  (CPU). Thời gian huấn luyện phụ thuộc máy; accuracy thì không.
- Thời gian ghi trong `results/` và báo cáo chỉ để tham khảo. Chạy lại trên cùng máy (lúc rảnh) lệch vài phần trăm; máy
  khác, số lõi khác hoặc máy đang bận thì có thể lệch nhiều lần, nhất là tỉ lệ giữa mô hình một luồng (MNB, AdaBoost) và
  mô hình chạy mọi lõi (RF, HistGradientBoosting). Nên so thứ tự và độ lớn tương đối, không so từng giây.

Phân tích khám phá dữ liệu (EDA) nằm ở `eda/notebooks/eda.ipynb`.

## Demo

Vẽ một chữ số và xem mô hình cuối đoán (Gradio; cần `results/<mô hình>/model.pkl`, có sẵn trong Git trừ Random
Forest):

```bash
python -m demo.app                       # mở http://127.0.0.1:7860
```

Ảnh vẽ được đưa về dạng MNIST trước khi đoán (`demo/drawing.py`): cắt theo khung chữ, thu nhỏ giữ tỉ lệ vào
20×20, đặt vào 28×28 sao cho tâm khối lượng ở giữa. Trang hiện cả ảnh 28×28 mà mô hình nhận được.
Danh sách mô hình (`demo/models.py`) gồm 5 mô hình đơn và **CNN × 5**: trung bình xác suất của 5 CNN khác seed
(`results/ensemble/cnn_s0..4.pkl`, có sẵn trong Git), test 99,15% so với 98,82% của một CNN. Tổ hợp cuối còn
cộng RF, nhưng RF không làm tăng accuracy test và làm xác suất kém đi nên demo bỏ.
Random Forest chỉ có trong danh sách khi đã có `results/random_forest/model.pkl`: tạo bằng
`python -m src.experiments.final_models --models random_forest` (vài giây) hoặc chạy `notebooks/results.ipynb` một lần.
`python -m demo.robustness` đo độ bền của bước chuẩn hoá trên 300 ảnh test vẽ lại to và lệch, ghi
`results/demo/` (báo cáo chỉ dẫn con số chính ở phụ lục B; bảng và hình đầy đủ nằm trong `results/demo/`).
Bấm vào biểu tượng bút trước khi vẽ. Gradio tải ảnh vẽ lên thành file; bấm Dự đoán ngay sau khi vẽ thì file có thể
chưa ghi xong, nên `demo/app.py` đọc lại tối đa ~3 giây thay vì báo "Error".

## Kiểm thử

Không cần pytest:

```bash
python tests/run_all.py                 # mọi file tests/check_*.py
python tests/run_all.py cnn             # chỉ các file có tên chứa "cnn"
python tests/check_analysis.py          # chạy một file
```

## Cấu trúc

```
nhom10-mnist-classifiers/
├── README.md                 hướng dẫn cài đặt, chạy, tái lập
├── requirements.txt          danh sách gói, có chú thích gói nào dùng ở đâu
├── REPORT.pdf                báo cáo nộp
├── SLIDE.pdf                 slide nộp
├── data/raw/                 4 file MNIST dạng IDX (có sẵn trong repo)
├── src/                      toàn bộ mã nguồn chính (chạy bằng python -m src....)
│   ├── config.py             hằng số dùng chung: seed, ngưỡng EDA, tên và màu mô hình
│   ├── data/                 đọc IDX, chia tập, biến đổi đặc trưng
│   ├── models/               các bộ phân lớp (tự cài; Random Forest bọc sklearn) + Pipeline + giao diện chung
│   ├── evaluation/           độ đo, phân tích (H1–H4), ghi/đọc kết quả
│   ├── experiments/          mỗi mô hình một script thí nghiệm + các script tổng hợp
│   └── plots/                mỗi mô hình một module vẽ hình từ results/ (python -m src.plots)
├── tests/                    check_*.py: kiểm tra tự cài, đối chiếu sklearn/scipy/torch
├── eda/notebooks/eda.ipynb   phân tích khám phá dữ liệu: nguồn gốc mọi quyết định
├── notebooks/results.ipynb   notebook tái lập: nạp mô hình cuối, tính lại mọi con số
├── demo/                     giao diện Gradio vẽ chữ số (app.py) và chuẩn hoá ảnh vẽ (drawing.py)
├── report/                   hình dùng trong report
├── docs/                     tài liệu chi tiết từng mô hình (<mô hình>_pipeline.md)
└── results/                  kết quả do script sinh ra (lần chạy đầy đủ, có trong Git)
    ├── <mô hình>/            log.txt, summary.json, final_metrics.json, model.pkl, figures/
    ├── ensemble/             như trên, cộng cnn_s0..4.pkl (5 CNN của tổ hợp)
    ├── comparison/           bảng và hình so sánh mọi mô hình
    └── sklearn_reference/, torch_reference/   đối chiếu thư viện
```

## Báo cáo

Báo cáo nộp là `REPORT.pdf` và slide là `SLIDE.pdf`, cùng ở thư mục gốc repo. 

`docs/<mô hình>_pipeline.md` là tài liệu chi tiết cho từng mô hình: EDA dẫn tới cách xử lý nào,
bảng đầy đủ của từng thí nghiệm, luật chọn cấu hình, vị trí mã nguồn và lệnh chạy.

Quy tắc: hàm nào dùng được cho hơn một mô hình thì đặt ở `src/config.py`, `src/data/features.py`,
`src/models/pipeline.py`, `src/evaluation/analysis.py`, `results_io.py` hoặc `src/plots/common.py`, không chép vào
file của từng mô hình.

## Lưu ý

- `results/` chứa kết quả của lần chạy đầy đủ và nằm trong Git; không sửa tay, chạy lại script là ghi đè. Không
  commit kết quả `--quick` (`compare_models` và `notebooks/results.ipynb` đều cảnh báo).
- Mọi biến đổi có tham số (mặt nạ pixel, chuẩn hoá, lượng tử) chỉ học từ `splits["train"]`.
