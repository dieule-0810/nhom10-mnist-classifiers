"""Hằng số dùng chung cho mọi mô hình. Mọi file khác import từ đây, không tự khai báo lại.

Đổi một giá trị ở đây là đổi cho cả nhóm, nên chỉ sửa khi cả nhóm đã thống nhất.
"""

# --- Dữ liệu ---
N_CLASSES = 10
IMAGE_SHAPE = (28, 28)
N_PIXELS = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]

# Chia validation (src/data/preprocess.py): 10.000 ảnh, stratify, seed 42 (EDA 0.3, 0.5)
VAL_SIZE = 10_000
SPLIT_SEED = 42

# Seed của mọi thành phần ngẫu nhiên trong mô hình và thí nghiệm
SEED = 0

# --- Số liệu chốt từ EDA (eda/notebooks/eda.ipynb, 50.000 ảnh splits["train"]) ---
RARE_FREQ = 0.001                                         # pixel "gần như chết": có mực ở < 0,1% số ảnh (EDA 1.1)
EDA_CLOSE_PAIRS = ((4, 9), (7, 9), (5, 8), (3, 5), (3, 8))  # 5 cặp có tâm gần nhau nhất (EDA 3.1), giả thuyết H3
BLOCK = 4                                                 # khối 4×4 cho độ quan trọng hoán vị (7 × 7 = 49 khối)
PERM_REPEATS = 2                                          # số lần hoán vị mỗi khối; như nhau cho mọi mô hình

# --- Kết quả ---
MODEL_KEYS = ("naive_bayes", "logistic_regression", "mlp", "random_forest", "cnn")
BASELINE_KEY = "nearest_centroid"
ENSEMBLE_KEY = "ensemble"                                 # gộp mô hình (bước 4–5), chạy sau mọi MODEL_KEYS
SPLITS = ("train", "val", "test")
DISPLAY_NAMES = {
    "nearest_centroid": "Tâm gần nhất (baseline)",
    "naive_bayes": "Naive Bayes",
    "logistic_regression": "Logistic Regression",
    "mlp": "MLP",
    "random_forest": "Random Forest",
    "cnn": "CNN",
    "ensemble": "Gộp mô hình (soft voting)",
    "cnn_ensemble": "CNN × 5 (gộp 5 seed)",       # chỉ dùng trong demo (demo/models.py)
}
# Một màu cố định cho mỗi mô hình, dùng ở MỌI hình (hình riêng của mô hình và hình so sánh)
MODEL_COLORS = {
    "nearest_centroid": "#9e9e9e",
    "naive_bayes": "#2a78d6",
    "logistic_regression": "#eb6834",
    "mlp": "#8e5bd0",
    "random_forest": "#1baf7a",
    "cnn": "#c43c5c",
    "ensemble": "#3d3d3d",
    "cnn_ensemble": "#7a1f38",
}
