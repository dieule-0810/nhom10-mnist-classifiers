"""Hình của mốc tâm gần nhất, vẽ từ results/nearest_centroid/final_metrics.json.

Chạy: python -m src.plots.baseline   (hoặc python -m src.plots để vẽ mọi mô hình)
"""

from src.config import BASELINE_KEY, DISPLAY_NAMES
from src.evaluation.results_io import clear_figures, fig_dir, load_final_metrics
from src.plots.common import plot_confusion_splits

KEY = BASELINE_KEY


def plot_all():
    fm = load_final_metrics(KEY)
    clear_figures(KEY)
    return [plot_confusion_splits(fm["splits"], fig_dir(KEY) / "eval_confusion.png", DISPLAY_NAMES[KEY])]


if __name__ == "__main__":
    for path in plot_all():
        print(path)
