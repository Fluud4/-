"""
Приложение Б.1 — базовый скрипт подготовки данных и метрик.

Датасет: Spector & Mazzeo (statsmodels.datasets.spector).
Обучение: statsmodels.Logit.
Метрики на тесте: scikit-learn (порог 0.5).

Зависимости: numpy, pandas, statsmodels, scikit-learn
"""

import numpy as np
import statsmodels.api as sm
from statsmodels.datasets import spector
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split


def main() -> None:
    data = spector.load_pandas().data
    print("Столбцы:", list(data.columns))
    print(data.head())
    print("Баланс GRADE:\n", data["GRADE"].value_counts())

    y = data["GRADE"].astype(int)
    X = data[["GPA", "TUCE", "PSI"]]
    X_const = sm.add_constant(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_const, y, test_size=0.3, random_state=42, stratify=y
    )

    model = sm.Logit(y_train, X_train)
    result = model.fit(disp=False)
    print(result.summary())

    or_table = np.exp(result.params)
    print("\nOdds ratios:")
    print(or_table)

    proba = result.predict(X_test)
    y_pred = (proba >= 0.5).astype(int)

    print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, proba):.4f}")
    print("Confusion matrix [[TN FP],[FN TP]]:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))


if __name__ == "__main__":
    main()
