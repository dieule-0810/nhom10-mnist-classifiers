"""Vẽ hình từ kết quả đã lưu trong results/<key>/ (summary.json, final_metrics.json), không huấn luyện lại.

Mỗi mô hình một module src/plots/<key>.py có plot_all(); script thí nghiệm gọi plot_all() ở cuối, và có thể
vẽ lại bất cứ lúc nào: python -m src.plots [key ...].
"""
