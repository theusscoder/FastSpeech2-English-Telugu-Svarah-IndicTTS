"""Plot training and validation losses from the CSV logs written by train.py.

Only logged values are drawn. Each loss gets its own panel and y-scale, because the
terms differ in size and sharing an axis would flatten the smaller ones.

    python English/plot_losses.py
"""
import argparse
import csv
from pathlib import Path

import common  # noqa: F401
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from common import load_configs  # noqa: E402

PANELS = [
    ("total", "Total"),
    ("mel", "Mel (MAE)"),
    ("mel_postnet", "Mel after PostNet (MAE)"),
    ("pitch", "Pitch (MSE, normalized)"),
    ("energy", "Energy (MSE, normalized)"),
    ("duration", "Log duration (MSE)"),
]
LINE = "#2a78b8"
RAW = "#a9cbe6"
INK = "#333333"
MUTED = "#777777"
GRID = "#e6e6e6"


def read_log(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} has no logged values.")
    return {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}


def moving_average(values, window):
    if len(values) < window:
        return None
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot(log, title, subtitle, path, smooth_window=0):
    figure, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharex=True)
    figure.patch.set_facecolor("white")
    steps = log["step"]
    single_point = len(steps) == 1

    for axis, (key, label) in zip(axes.flat, PANELS):
        values = log[key]
        smoothed = moving_average(values, smooth_window) if smooth_window else None
        if smoothed is not None:
            axis.plot(steps, values, color=RAW, linewidth=0.8)
            axis.plot(steps[smooth_window - 1 :], smoothed, color=LINE, linewidth=2)
        else:
            axis.plot(steps, values, color=LINE, linewidth=2,
                      marker="o" if single_point or len(steps) <= 40 else None, markersize=4)
        axis.set_title(label, fontsize=10.5, color=INK, loc="left")
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color(MUTED)
        axis.tick_params(colors=MUTED, labelsize=8.5, length=0)
        axis.ticklabel_format(axis="x", style="plain")
        axis.annotate(
            f"last {values[-1]:.3f}", xy=(1, 1), xycoords="axes fraction",
            ha="right", va="bottom", fontsize=8.5, color=INK,
        )
    for axis in axes[1]:
        axis.set_xlabel("Training step", fontsize=9, color=MUTED)

    figure.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
    figure.text(0.01, 0.925, subtitle, ha="left", fontsize=9.5, color=MUTED)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(path, dpi=170, facecolor="white")
    plt.close(figure)
    print("Wrote", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(common.DEFAULT_CONFIG))
    args = parser.parse_args()
    preprocess_config, _, train_config, _ = load_configs(args.config)
    language = {"en": "English", "te": "Telugu"}[preprocess_config["preprocessing"]["text"]["language"]]
    name = f"{language} FastSpeech 2"
    log_dir = Path(train_config["path"]["log_path"])
    results = Path(train_config["path"]["result_path"])
    results.mkdir(parents=True, exist_ok=True)

    train_log = read_log(log_dir / "train_loss.csv")
    window = 20
    plot(
        train_log,
        f"{name}: training loss",
        f"{len(train_log['step'])} logged batches to step {int(train_log['step'][-1])}. "
        f"Pale: logged values. Dark: moving average of {window} logged values.",
        results / "training_loss.png",
        smooth_window=window,
    )
    val_log = read_log(log_dir / "val_loss.csv")
    plot(
        val_log,
        f"{name}: validation loss",
        f"{len(val_log['step'])} evaluations to step {int(val_log['step'][-1])}, "
        "averaged over the validation set using ground-truth durations.",
        results / "validation_loss.png",
    )


if __name__ == "__main__":
    main()
