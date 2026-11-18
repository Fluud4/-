"""Метрики качества бинарной классификации."""

from __future__ import annotations

from typing import Dict

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(y_true == y_pred))


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def roc_curve(y_true: np.ndarray, y_score: np.ndarray):
    """Простая ROC-кривая: возвращает fpr, tpr, thresholds."""
    y_true = np.asarray(y_true).ravel().astype(int)
    y_score = np.asarray(y_score).ravel().astype(float)
    thresholds = np.unique(y_score)[::-1]
    thresholds = np.concatenate([[thresholds[0] + 1e-9], thresholds, [-np.inf]])

    tpr_list, fpr_list = [], []
    p = np.sum(y_true == 1)
    n = np.sum(y_true == 0)
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tpr_list.append(tp / p if p else 0.0)
        fpr_list.append(fp / n if n else 0.0)
    return np.array(fpr_list), np.array(tpr_list), thresholds


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.trapezoid(tpr, fpr))


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    base = precision_recall_f1(y_true, y_pred)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": base["precision"],
        "recall": base["recall"],
        "f1": base["f1"],
        "roc_auc": roc_auc_score(y_true, y_score),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
