"""
Приложение Б.3 — единый скрипт подготовки блоков «ДАННЫЕ» для промптов.
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


def main() -> None:
    data = spector.load_pandas().data
    y = data["GRADE"].astype(int)
    X = data[["GPA", "TUCE", "PSI"]]
    X_sm = sm.add_constant(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sm, y, test_size=0.3, random_state=42, stratify=y
    )

    result = sm.Logit(y_train, X_train).fit(disp=False)
    proba_sm = result.predict(X_test)
    pred_sm = (proba_sm >= 0.5).astype(int)
    cm_sm = confusion_matrix(y_test, pred_sm)

    X_train_sk = X_train.drop(columns=["const"])
    X_test_sk = X_test.drop(columns=["const"])
    sk = LogisticRegression(max_iter=1000).fit(X_train_sk, y_train)
    proba_sk = sk.predict_proba(X_test_sk)[:, 1]
    pred_sk = sk.predict(X_test_sk)
    cm_sk = confusion_matrix(y_test, pred_sk)

    tn, fp, fn, tp = cm_sm.ravel()

    print("=" * 60)
    print("БЛОК ДАННЫХ ДЛЯ ПРОМПТОВ: интерпретация / академический / рецензия")
    print("=" * 60)
    print(
        f"Датасет: Spector & Mazzeo (statsmodels.datasets.spector), n={len(data)}.\n"
        f"Признаки: GPA, TUCE, PSI. Цель: GRADE (0/1).\n"
        f"Разбиение: train/test 70/30, stratify, random_state=42 "
        f"(train={len(y_train)}, test={len(y_test)}).\n"
    )
    print("statsmodels.Logit (train):")
    print(
        f"Pseudo R²={result.prsquared:.4f}, "
        f"converged={result.mle_retvals.get('converged', True)}, "
        f"LLR p-value={result.llr_pvalue:.5f}"
    )
    for name in result.params.index:
        print(
            f"{name}={result.params[name]:.4f} (p={result.pvalues[name]:.3f}), "
            f"OR={np.exp(result.params[name]):.3f}"
        )
    print(
        f"\nТест (порог 0.5):\n"
        f"Accuracy={accuracy_score(y_test, pred_sm):.2f}, "
        f"Precision={precision_score(y_test, pred_sm, zero_division=0):.2f}, "
        f"Recall={recall_score(y_test, pred_sm, zero_division=0):.2f}, "
        f"F1={f1_score(y_test, pred_sm, zero_division=0):.2f}, "
        f"ROC-AUC={roc_auc_score(y_test, proba_sm):.2f}\n"
        f"Confusion matrix [[TN FP],[FN TP]] = [[{tn} {fp}],[{fn} {tp}]]"
    )

    print("\n" + "=" * 60)
    print("БЛОК ДАННЫХ ДЛЯ ПРОМПТА: бизнес / простой язык")
    print("=" * 60)
    print(
        f"Датасет Spector: n={len(data)} студентов. Цель GRADE=1 — студент улучшил оценку.\n"
        f"Признаки: GPA (средний балл), TUCE (тест), PSI (участвовал в программе: 1/0).\n\n"
        f"Коэффициенты Logit: "
        f"const={result.params['const']:.2f} (p={result.pvalues['const']:.3f}), "
        f"GPA={result.params['GPA']:+.2f} (p={result.pvalues['GPA']:.3f}), "
        f"TUCE={result.params['TUCE']:+.2f} (p={result.pvalues['TUCE']:.3f}), "
        f"PSI={result.params['PSI']:+.2f} (p={result.pvalues['PSI']:.3f}).\n"
        f"Тест ({len(y_test)} человек): "
        f"Accuracy={accuracy_score(y_test, pred_sm):.2f}, "
        f"Precision={precision_score(y_test, pred_sm, zero_division=0):.2f}, "
        f"Recall={recall_score(y_test, pred_sm, zero_division=0):.2f}, "
        f"F1={f1_score(y_test, pred_sm, zero_division=0):.2f}, "
        f"ROC-AUC={roc_auc_score(y_test, proba_sm):.2f}.\n"
        f"Матрица: TN={tn}, FP={fp}, FN={fn}, TP={tp}."
    )

    print("\n" + "=" * 60)
    print("БЛОК ДАННЫХ ДЛЯ ПРОМПТА: сравнение statsmodels vs sklearn")
    print("=" * 60)
    print(
        f"Датасет Spector, n={len(data)}, train={len(y_train)}, test={len(y_test)}, "
        f"один split (random_state=42, stratify).\n"
        f"Признаки GPA, TUCE, PSI; цель GRADE.\n\n"
        f"statsmodels.Logit:\n"
        f"coefs: const={result.params['const']:.3f}, GPA={result.params['GPA']:.3f}, "
        f"TUCE={result.params['TUCE']:.3f}, PSI={result.params['PSI']:.3f}\n"
        f"p-values: const={result.pvalues['const']:.3f}, GPA={result.pvalues['GPA']:.3f}, "
        f"TUCE={result.pvalues['TUCE']:.3f}, PSI={result.pvalues['PSI']:.3f}\n"
        f"Test: Acc={accuracy_score(y_test, pred_sm):.2f}, "
        f"Prec={precision_score(y_test, pred_sm, zero_division=0):.2f}, "
        f"Rec={recall_score(y_test, pred_sm, zero_division=0):.2f}, "
        f"F1={f1_score(y_test, pred_sm, zero_division=0):.2f}, "
        f"AUC={roc_auc_score(y_test, proba_sm):.4f}, CM={cm_sm.tolist()}\n\n"
        f"sklearn.LogisticRegression(max_iter=1000):\n"
        f"intercept={sk.intercept_[0]:.3f}, GPA={sk.coef_[0][0]:.3f}, "
        f"TUCE={sk.coef_[0][1]:.3f}, PSI={sk.coef_[0][2]:.3f}\n"
        f"Test: Acc={accuracy_score(y_test, pred_sk):.2f}, "
        f"Prec={precision_score(y_test, pred_sk, zero_division=0):.2f}, "
        f"Rec={recall_score(y_test, pred_sk, zero_division=0):.2f}, "
        f"F1={f1_score(y_test, pred_sk, zero_division=0):.2f}, "
        f"AUC={roc_auc_score(y_test, proba_sk):.4f}, CM={cm_sk.tolist()}"
    )

    print("\n" + "=" * 60)
    print("БЛОК ДАННЫХ ДЛЯ ПРОМПТА: порог и ошибки")
    print("=" * 60)
    sig = [name for name in ["GPA", "TUCE", "PSI"] if result.pvalues[name] < 0.05]
    print(
        f"Задача: предсказать GRADE=1 (студент улучшил оценку), Spector n={len(data)}, "
        f"test={len(y_test)}.\n"
        f"Порог 0.5. Метрики: Acc={accuracy_score(y_test, pred_sm):.2f}, "
        f"Prec={precision_score(y_test, pred_sm, zero_division=0):.2f}, "
        f"Rec={recall_score(y_test, pred_sm, zero_division=0):.2f}, "
        f"F1={f1_score(y_test, pred_sm, zero_division=0):.2f}, "
        f"AUC={roc_auc_score(y_test, proba_sm):.2f}.\n"
        f"CM: TN={tn}, FP={fp}, FN={fn}, TP={tp}.\n"
        f"Значимые на 5%: {', '.join(sig) if sig else 'нет'}; "
        f"незначимы: {', '.join([n for n in ['GPA', 'TUCE', 'PSI'] if n not in sig])}."
    )


if __name__ == "__main__":
    main()
