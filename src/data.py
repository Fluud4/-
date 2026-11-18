"""Загрузка и подготовка данных для курсового проекта."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def load_dataset(save_csv: bool = True) -> Tuple[pd.DataFrame, pd.Series, str]:
    """
    Загружает датасет Breast Cancer Wisconsin (бинарная классификация).

    Целевая переменная:
      0 — злокачественная опухоль (malignant)
      1 — доброкачественная опухоль (benign)
    """
    bundle = load_breast_cancer()
    X = pd.DataFrame(bundle.data, columns=bundle.feature_names)
    # в sklearn: 0=malignant, 1=benign — оставляем как есть
    y = pd.Series(bundle.target, name="target")
    description = (
        "Breast Cancer Wisconsin Diagnostic Dataset: "
        f"{X.shape[0]} объектов, {X.shape[1]} признаков, "
        "задача бинарной классификации (malignant / benign)."
    )

    if save_csv:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        frame = X.copy()
        frame["target"] = y
        frame.to_csv(DATA_DIR / "breast_cancer.csv", index=False)

    return X, y, description


def prepare_splits(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    random_state: int = 42,
):
    """Делит данные на train/test и стандартизирует признаки."""
    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y.values,
        test_size=test_size,
        random_state=random_state,
        stratify=y.values,
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test, scaler
