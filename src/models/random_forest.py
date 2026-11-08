"""Random Forest cho bài toán phân lớp: RandomForestClassifier của scikit-learn và các hàm phân tích rừng.

Nhánh ensemble dùng thư viện scikit-learn (giảng viên cho phép, như bài Lab tuần 5 và slide B6b "Ensemble cho
ảnh MNIST"); các mô hình còn lại của nhóm tự cài trong src/models/.

Tham khảo:
    [1] L. Breiman, "Bagging predictors," Machine Learning, vol. 24, no. 2, pp. 123–140, 1996.
    [2] L. Breiman, "Random forests," Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.
    [3] T. Hastie, R. Tibshirani, J. Friedman, The Elements of Statistical Learning, 2nd ed.
        Springer, 2009. Chương 15.

Thuật toán [3, Algorithm 15.1]. Với t = 1..T:
    1. Rút một mẫu bootstrap: N ảnh có hoàn lại từ tập train [1].
    2. Trồng một cây CART (Gini) trên mẫu đó; ở mỗi nút chỉ xét max_features pixel chọn ngẫu nhiên [2].
Dự đoán: trung bình phân phối lớp ở lá của T cây rồi lấy argmax (soft voting, cách của sklearn).
Ước lượng out-of-bag (OOB) [2, mục 3.1]: mỗi ảnh train không được rút vào khoảng (1 − 1/N)^N ≈ e⁻¹ ≈ 37% số
cây; dự đoán ảnh đó chỉ bằng các cây này cho một ước lượng lỗi không tốn thêm dữ liệu (rf.oob_score_).

Các hàm phân tích chỉ dùng API công khai của sklearn: rf.estimators_ là danh sách T cây
(DecisionTreeClassifier), mỗi cây có predict_proba, get_depth, get_n_leaves và cấu trúc nút est.tree_.
Nhãn là 0..9 nên cột k của predict_proba ứng với chữ số k.
Kiểm tra trong tests/check_random_forest.py.
"""

import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.config import SEED

# Rừng ít cây có vài ảnh chưa là OOB của cây nào; sklearn cảnh báo rồi vẫn tính oob_score_ trên mọi ảnh.
warnings.filterwarnings("ignore", message="Some inputs do not have OOB scores")


def make_random_forest(n_estimators=100, criterion="gini", max_features="sqrt", max_depth=None,
                       min_samples_leaf=1, n_jobs=-1, seed=SEED):
    """RandomForestClassifier có bootstrap và ước lượng OOB; n_jobs là số luồng trồng cây song song."""
    return RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_features=max_features,
                                  max_depth=max_depth, min_samples_leaf=min_samples_leaf, bootstrap=True,
                                  oob_score=True, n_jobs=n_jobs, random_state=seed)


def tree_stats(rf):
    """Độ sâu và số lá trung bình của các cây trong rừng."""
    return (float(np.mean([t.get_depth() for t in rf.estimators_])),
            float(np.mean([t.get_n_leaves() for t in rf.estimators_])))


def tree_votes(rf, X, voting="soft"):
    """Đóng góp của từng cây cho các ảnh X, shape (T, N, K).

    soft: phân phối lớp ở lá của cây (trung bình theo cây chính là rf.predict_proba);
    hard: one-hot của lớp cây dự đoán, tức mỗi cây bỏ một phiếu (cách của Breiman [2]).
    """
    P = np.stack([t.predict_proba(X) for t in rf.estimators_])
    if voting == "soft":
        return P
    if voting != "hard":
        raise ValueError(f"voting phải là 'soft' hoặc 'hard', nhận {voting!r}.")
    return np.eye(P.shape[2])[P.argmax(axis=2)]


def staged_accuracy(rf, X, y, voting="soft"):
    """Accuracy của rừng gồm 1, 2, ..., T cây đầu tiên: cộng dồn phiếu theo cây rồi lấy argmax."""
    cum = np.cumsum(tree_votes(rf, X, voting), axis=0)
    return [float((C.argmax(axis=1) == y).mean()) for C in cum]


def strength_correlation(rf, X, y):
    """Hai đại lượng trong cận sai số của Breiman [2]: cây mạnh và tương quan giữa các cây thấp.

    strength    = accuracy trung bình của từng cây trên (X, y)
    correlation = hệ số tương quan Pearson trung bình giữa các vectơ lỗi 0/1 của hai cây khác nhau
    """
    E = (tree_votes(rf, X).argmax(axis=2) != y).astype(np.float64)
    strength = float(1.0 - E.mean())
    C = np.corrcoef(E)
    iu = np.triu_indices(len(E), k=1)
    return strength, float(np.nanmean(C[iu]))
