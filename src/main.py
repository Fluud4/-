"""
Точка входа курсового проекта.

Запуск из корня репозитория:
    python -m src.main
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogReg

from .data import load_dataset, prepare_splits
from .logistic_regression import LogisticRegressionScratch
from .metrics import classification_report
from .visualize import (
    plot_confusion_matrix,
    plot_feature_weights,
    plot_loss_curve,
    plot_roc,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _print_report(title: str, report: dict) -> None:
    print(f"\n=== {title} ===")
    print(f"Accuracy : {report['accuracy']:.4f}")
    print(f"Precision: {report['precision']:.4f}")
    print(f"Recall   : {report['recall']:.4f}")
    print(f"F1-score : {report['f1']:.4f}")
    print(f"ROC-AUC  : {report['roc_auc']:.4f}")
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(report["confusion_matrix"])


def run() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)

    X, y, description = load_dataset(save_csv=True)
    print(description)
    print(f"Баланс классов: 0 → {(y == 0).sum()}, 1 → {(y == 1).sum()}")

    X_train, X_test, y_train, y_test, _scaler = prepare_splits(X, y)

    scratch = LogisticRegressionScratch(
        learning_rate=0.3,
        n_iterations=2000,
        l2=0.5,
        random_state=42,
    )
    scratch.fit(X_train, y_train)

    y_prob_scratch = scratch.predict_proba(X_test)[:, 1]
    y_pred_scratch = scratch.predict(X_test)
    report_scratch = classification_report(y_test, y_pred_scratch, y_prob_scratch)
    _print_report("Собственная реализация", report_scratch)
    print(f"Итераций до сходимости: {scratch.n_iter_}")

    sklearn_model = SklearnLogReg(max_iter=2000, C=1.0, solver="lbfgs")
    sklearn_model.fit(X_train, y_train)
    y_prob_sk = sklearn_model.predict_proba(X_test)[:, 1]
    y_pred_sk = sklearn_model.predict(X_test)
    report_sk = classification_report(y_test, y_pred_sk, y_prob_sk)
    _print_report("sklearn.LogisticRegression", report_sk)

    plot_loss_curve(scratch.loss_history_, RESULTS / "loss_curve.png")
    plot_confusion_matrix(
        report_scratch["confusion_matrix"],
        RESULTS / "cm_scratch.png",
        "Матрица ошибок — собственная модель",
    )
    plot_confusion_matrix(
        report_sk["confusion_matrix"],
        RESULTS / "cm_sklearn.png",
        "Матрица ошибок — sklearn",
    )
    plot_roc(
        y_test,
        {
            "Scratch": y_prob_scratch,
            "sklearn": y_prob_sk,
        },
        RESULTS / "roc_curves.png",
    )
    plot_feature_weights(
        list(X.columns),
        scratch.weights_,
        RESULTS / "feature_weights.png",
        title="Топ весов собственной модели",
    )

    summary = {
        "dataset": description,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "scratch": {
            "accuracy": report_scratch["accuracy"],
            "precision": report_scratch["precision"],
            "recall": report_scratch["recall"],
            "f1": report_scratch["f1"],
            "roc_auc": report_scratch["roc_auc"],
            "n_iter": scratch.n_iter_,
            "final_loss": scratch.loss_history_[-1] if scratch.loss_history_ else None,
        },
        "sklearn": {
            "accuracy": report_sk["accuracy"],
            "precision": report_sk["precision"],
            "recall": report_sk["recall"],
            "f1": report_sk["f1"],
            "roc_auc": report_sk["roc_auc"],
        },
    }

    with open(RESULTS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nАртефакты сохранены в: {RESULTS}")
    return summary


if __name__ == "__main__":
    run()
