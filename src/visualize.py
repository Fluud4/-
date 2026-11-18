"""Визуализация результатов обучения."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .metrics import roc_curve


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_loss_curve(loss_history, out_path: Path) -> None:
    _ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(loss_history, color="#1f4e79", linewidth=2)
    ax.set_title("Кривая функции потерь (log-loss)")
    ax.set_xlabel("Итерация")
    ax.set_ylabel("Log-loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, out_path: Path, title: str) -> None:
    _ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc(
    y_true: np.ndarray,
    scores: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    _ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    colors = ["#1f4e79", "#c45c26"]
    for (name, score), color in zip(scores.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, score)
        auc = float(np.trapezoid(tpr, fpr))
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-кривые")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_weights(
    feature_names,
    weights: np.ndarray,
    out_path: Path,
    top_k: int = 12,
    title: str = "Веса признаков",
) -> None:
    _ensure_dir(out_path.parent)
    idx = np.argsort(np.abs(weights))[::-1][:top_k]
    names = [feature_names[i] for i in idx][::-1]
    vals = weights[idx][::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#c45c26" if v < 0 else "#1f4e79" for v in vals]
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Вес")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
