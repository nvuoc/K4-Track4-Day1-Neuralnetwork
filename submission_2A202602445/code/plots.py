"""plots.py — Vẽ đồ thị huấn luyện và so sánh thí nghiệm.

Mỗi thí nghiệm một ảnh: figures/<exp_id>.png gồm 3 ô:
  (1) train_loss và val_loss theo epoch
  (2) val_acc và val_macro_f1 theo epoch
  (3) grad_norm theo epoch (đo trước khi clip)
Và các ảnh so sánh nhóm: figures/compare_<nhóm>.png.
"""
from __future__ import annotations

import os
import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc và val_macro_f1 theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    """
    history = result.get("history", {})
    cfg = result.get("cfg", {})
    summary = result.get("summary", {})

    epochs = history.get("epoch", [])
    if not epochs:
        return

    exp_id = cfg.get("exp_id", "exp")
    opt = cfg.get("optimizer", "")
    lr = cfg.get("lr", "")
    batch = cfg.get("batch", "")
    best_epoch = summary.get("best_epoch", None)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Ô 1: Loss
    ax1 = axes[0]
    ax1.plot(epochs, history.get("train_loss", []), label="Train Loss (eval mode)", color="#1f77b4", lw=1.8)
    ax1.plot(epochs, history.get("val_loss", []), label="Val Loss", color="#d62728", lw=1.8)
    if best_epoch and best_epoch in epochs:
        ax1.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best ep ({best_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train & Val Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Ô 2: Accuracy & Macro-F1
    ax2 = axes[1]
    ax2.plot(epochs, history.get("val_acc", []), label="Val Accuracy", color="#2ca02c", lw=1.8)
    if "val_macro_f1" in history:
        ax2.plot(epochs, history.get("val_macro_f1", []), label="Val Macro-F1", color="#ff7f0e", lw=1.8)
    if best_epoch and best_epoch in epochs:
        ax2.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best ep ({best_epoch})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Val Accuracy & Macro-F1")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    # Ô 3: Gradient Norm
    ax3 = axes[2]
    ax3.plot(epochs, history.get("grad_norm", []), label="Grad Norm (pre-clip)", color="#9467bd", lw=1.8)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("L2 Norm")
    ax3.set_title("Gradient Norm (trước khi clip)")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend()

    fig.suptitle(f"{exp_id} | Opt: {opt} | lr: {lr} | Batch: {batch}", fontsize=13, fontweight="bold", y=1.02)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số của nhiều thí nghiệm trên cùng một trục."""
    if not results:
        return

    metric_names = {
        "val_loss": "Val Loss",
        "val_macro_f1": "Val Macro-F1",
        "val_acc": "Val Accuracy",
        "train_loss": "Train Loss",
        "grad_norm": "Grad Norm (pre-clip)",
    }
    y_label = metric_names.get(metric, metric)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for r in results:
        cfg = r.get("cfg", {})
        hist = r.get("history", {})
        epochs = hist.get("epoch", [])
        vals = hist.get(metric, [])
        if epochs and vals:
            label = cfg.get("exp_id", "exp")
            ax.plot(epochs, vals, marker="o", markersize=3, label=label, lw=1.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.set_title(title or f"So sánh {y_label}", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
