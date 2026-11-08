"""Đưa ảnh người dùng vẽ về đúng dạng ảnh MNIST trước khi đưa vào mô hình.

MNIST (LeCun et al., 1998) chuẩn hoá mỗi chữ số như sau, và mô hình chỉ quen với dạng này:
    1. Chữ số trắng trên nền đen (mực = giá trị lớn).
    2. Thu nhỏ, GIỮ tỉ lệ, sao cho cạnh dài vừa khung 20×20 (khử răng cưa nên có mức xám ở mép nét).
    3. Đặt vào khung 28×28 sao cho TÂM KHỐI LƯỢNG của mực nằm ở giữa (14, 14).
Bỏ qua các bước này thì một chữ số vẽ lệch góc hoặc quá to sẽ bị đoán sai dù mô hình tốt: RF, LR, MLP
gắn trọng số với từng pixel cố định (EDA 1.2), còn CNN chỉ chịu được dịch vài pixel.
"""

import numpy as np
from PIL import Image
from scipy import ndimage

BOX = 20            # cạnh dài của chữ số sau khi thu nhỏ (MNIST)
SIZE = 28
CANVAS = 280        # cạnh khung vẽ của demo/app.py
INK_THRESHOLD = 0.05


def to_ink(img) -> np.ndarray:
    """Ảnh vẽ (H, W), (H, W, 3) hoặc (H, W, 4), nét tối trên nền sáng hoặc nền trong suốt → mực trong [0, 1]."""
    a = np.asarray(img, dtype=np.float64)
    if a.max() > 1.0:
        a = a / 255.0
    if a.ndim == 2:
        return 1.0 - a
    rgb = a[..., :3]
    gray = rgb @ np.array([0.299, 0.587, 0.114])
    if a.shape[-1] == 4:                                   # nền trong suốt: đặt lên nền trắng trước
        alpha = a[..., 3]
        gray = alpha * gray + (1.0 - alpha)
    return 1.0 - gray


def drawn_canvas(x, shift, scale=7) -> np.ndarray:
    """Mô phỏng một ảnh vẽ từ ảnh MNIST x (784 giá trị mực trong [0, 1]): phóng to scale lần, đặt vào khung
    CANVAS×CANVAS nền trắng, tâm lệch khỏi giữa khung shift = (dọc, ngang) pixel. Trả về độ sáng trong [0, 1]
    (nét tối trên nền sáng, như ảnh từ khung vẽ), shape (CANVAS, CANVAS)."""
    big = SIZE * scale
    ink = Image.fromarray((255 * np.asarray(x).reshape(SIZE, SIZE)).astype(np.uint8)).resize((big, big), Image.BILINEAR)
    canvas = np.zeros((CANVAS, CANVAS))
    top, left = (CANVAS - big) // 2 + int(shift[0]), (CANVAS - big) // 2 + int(shift[1])
    canvas[top:top + big, left:left + big] = np.asarray(ink, dtype=np.float64) / 255.0
    return 1.0 - canvas


def mnist_steps(img, drop_halo=True) -> dict:
    """Các bước của to_mnist, giữ lại từng ảnh trung gian (dùng cho hình minh hoạ và đo độ bền).

    Trả về dict: "ink" (mực của cả khung vẽ), "crop" (khung chữ), "small" (sau khi thu nhỏ), "out" (28×28, float32
    trong [0, 1]). Chưa có nét nào thì "crop", "small" là None và "out" toàn 0.
    drop_halo=True: đặt về 0 các pixel mực < INK_THRESHOLD (quầng mờ do đổi kích thước).
    """
    ink = to_ink(img)
    rows, cols = np.nonzero(ink > INK_THRESHOLD)
    if len(rows) == 0:
        return dict(ink=ink, crop=None, small=None, out=np.zeros((SIZE, SIZE), dtype=np.float32))
    crop = ink[rows.min():rows.max() + 1, cols.min():cols.max() + 1]

    h, w = crop.shape
    scale = BOX / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    small = Image.fromarray((crop * 255).astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)
    small = np.asarray(small, dtype=np.float64) / 255.0

    # Đặt vào khung lớn rồi dịch sao cho tâm khối lượng ở giữa; khung đệm đủ rộng để dịch không mất mực
    canvas = np.zeros((SIZE * 2, SIZE * 2))
    top, left = (2 * SIZE - new_h) // 2, (2 * SIZE - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = small
    cy, cx = ndimage.center_of_mass(canvas)
    canvas = np.roll(canvas, (round(SIZE - cy), round(SIZE - cx)), axis=(0, 1))
    out = np.clip(canvas[SIZE // 2:SIZE // 2 + SIZE, SIZE // 2:SIZE // 2 + SIZE], 0.0, 1.0)
    if drop_halo:
        # Bỏ quầng mờ do đổi kích thước (Lanczos): RF nhị phân hoá theo pixel > 0 nên coi quầng là mực, nét dày ra
        out[out < INK_THRESHOLD] = 0.0
    return dict(ink=ink, crop=crop, small=small, out=out.astype(np.float32))


def to_mnist(img, drop_halo=True) -> np.ndarray:
    """Ảnh vẽ → mảng (784,) float32 trong [0, 1], cùng quy ước với load_splits (trước mọi bước của Pipeline).

    Trả về toàn 0 nếu chưa có nét nào. Các bước: xem mnist_steps.
    """
    return mnist_steps(img, drop_halo)["out"].reshape(-1)
