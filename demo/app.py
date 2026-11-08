"""Demo: vẽ một chữ số, mô hình cuối của nhóm đoán nhãn.

Chạy từ gốc repo (cần results/<mô hình>/model.pkl; "CNN × 5" cần results/ensemble/cnn_s0..4.pkl; file của Random
Forest không có trong Git, tạo bằng python -m src.experiments.final_models --models random_forest, hoặc chạy
notebooks/results.ipynb một lần; danh sách mô hình và cách nạp ở demo/models.py):
    python -m demo.app            # mở http://127.0.0.1:7860

Ảnh vẽ được đưa về dạng MNIST (demo/drawing.py: cắt, thu nhỏ vào 20×20, căn tâm khối lượng trong 28×28) rồi
đưa vào đúng Pipeline đã lưu, nên mọi bước tiền xử lý của mô hình (mặt nạ pixel, chuẩn hoá, lượng tử) giữ nguyên.
"""

import time

import gradio as gr
import numpy as np
import PIL.Image

from demo.drawing import CANVAS, SIZE, to_mnist
from demo.models import DEMO_MODELS, is_available, load_demo_model
from src.config import DISPLAY_NAMES

MODELS = {k: load_demo_model(k) for k in DEMO_MODELS if is_available(k)}
LABELS = {DISPLAY_NAMES[k]: k for k in MODELS}
if not MODELS:
    raise SystemExit("Chưa có results/<mô hình>/model.pkl nào: chạy python -m src.experiments.final_models trước.")


def retry_image_reads(pad):
    """Cho gr.Sketchpad đọc lại ảnh vẽ khi file tải lên chưa ghi xong; trả về chính `pad`.

    Gradio 6 tải ảnh vẽ lên thành file PNG; bấm Dự đoán ngay sau khi vẽ thì máy chủ có thể mở file lúc còn ghi dở
    (UnidentifiedImageError) và trang báo "Error". Thử lại tối đa ~3 giây; các lớp vẽ riêng (layers) không dùng tới
    nên lỗi ở chúng thì bỏ qua, chỉ cần ảnh gộp (composite).

    Gắn vào đối tượng thay vì viết lớp con: Gradio coi lớp con của một component là custom component và đi tải
    giao diện JS riêng của nó; không có file đó nên khung vẽ kẹt ở "Loading...".
    """
    original = pad.convert_and_format_image

    def convert_and_format_image(file, flatten_on_white=False):
        for _ in range(30):
            try:
                return original(file, flatten_on_white)
            except (PIL.UnidentifiedImageError, OSError):
                time.sleep(0.1)
        if flatten_on_white:                          # background / composite: lần cuối, để lỗi thật hiện ra
            return original(file, flatten_on_white)
        return None

    pad.convert_and_format_image = convert_and_format_image
    return pad


def predict(drawing, model_name):
    """drawing: giá trị của gr.Sketchpad (dict có 'composite', RGBA). Trả về (top-3 xác suất, ảnh 28×28 mô hình thấy)."""
    if drawing is None or drawing.get("composite") is None:
        return {}, None
    x = to_mnist(drawing["composite"])
    if not x.any():
        return {}, None
    proba = MODELS[LABELS[model_name]].predict_proba(x[None, :])[0]
    seen = np.kron(x.reshape(SIZE, SIZE), np.ones((8, 8)))          # phóng to 8 lần để dễ nhìn
    return {str(d): float(p) for d, p in enumerate(proba)}, (255 * (1 - seen)).astype(np.uint8)


with gr.Blocks(title="Nhận dạng chữ số viết tay") as demo:
    gr.Markdown("## Nhận dạng chữ số viết tay (MNIST)\nChọn bút (biểu tượng bên trái khung), vẽ **một** chữ số từ 0 đến 9, chọn mô hình rồi bấm Dự đoán.")
    with gr.Row():
        with gr.Column():
            pad = retry_image_reads(gr.Sketchpad(
                type="numpy", image_mode="RGBA", canvas_size=(CANVAS, CANVAS),
                brush=gr.Brush(default_size=18, colors=["#000000"], color_mode="fixed"),
                layers=False, label="Vẽ chữ số"))
            model = gr.Dropdown(list(LABELS), value=next(iter(LABELS)), label="Mô hình")
            button = gr.Button("Dự đoán", variant="primary")
        with gr.Column():
            label = gr.Label(num_top_classes=3, label="Dự đoán (3 lớp có xác suất cao nhất)")
            seen = gr.Image(label="Ảnh 28×28 mô hình nhận được", interactive=False)
    button.click(predict, [pad, model], [label, seen])

if __name__ == "__main__":
    demo.launch()
