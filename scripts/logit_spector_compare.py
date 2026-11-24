"""
Приложение Б.2 — сравнение statsmodels.Logit и sklearn.LogisticRegression.

Один и тот же train/test split (70/30, stratify, random_state=42).
"""

import numpy as np
import statsmodels.api as sm
from statsmodels.datasets import spector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


def report(name, y_true, y_pred, y_score) -> None:
    print(f"\n=== {name} ===")
    print("Accuracy :", round(accuracy_score(y_true, y_pred), 4))
    print("Precision:", round(precision_score(y_true, y_pred, zero_division=0), 4))
    print("Recall   :", round(recall_score(y_true, y_pred, zero_division=0), 4))
    print("F1-score :", round(f1_score(y_true, y_pred, zero_division=0), 4))
    print("ROC-AUC  :", round(roc_auc_score(y_true, y_score), 4))
    print("CM [[TN FP],[FN TP]]:\n", confusion_matrix(y_true, y_pred))


def main() -> None:
    data = spector.load_pandas().data
    y = data["GRADE"].astype(int)
    X = data[["GPA", "TUCE", "PSI"]]
    X_sm = sm.add_constant(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sm, y, test_size=0.3, random_state=42, stratify=y
    )

    result = sm.Logit(y_train, X_train).fit(disp=False)
    print(result.summary())
    print("SM odds ratios:\n", np.exp(result.params))

    proba_sm = result.predict(X_test)
    pred_sm = (proba_sm >= 0.5).astype(int)

    X_train_sk = X_train.drop(columns=["const"])
    X_test_sk = X_test.drop(columns=["const"])

    sk = LogisticRegression(max_iter=1000)
    sk.fit(X_train_sk, y_train)

    print("SK intercept:", sk.intercept_[0])
    print("SK coefs:", dict(zip(X_train_sk.columns, sk.coef_[0])))
    print(
        "SK odds ratios:",
        dict(zip(X_train_sk.columns, np.exp(sk.coef_[0]))),
    )

    proba_sk = sk.predict_proba(X_test_sk)[:, 1]
    pred_sk = sk.predict(X_test_sk)

    report("statsmodels.Logit", y_test, pred_sm, proba_sm)
    report("sklearn.LogisticRegression", y_test, pred_sk, proba_sk)


if __name__ == "__main__":
    main()
