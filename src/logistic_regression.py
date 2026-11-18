"""
Собственная реализация логистической регрессии (бинарная классификация).

Оптимизация: градиентный спуск по логистической функции потерь (log-loss).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Сигмоида σ(z) = 1 / (1 + e^{-z}) с защитой от переполнения."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class LogisticRegressionScratch:
    """
    Логистическая регрессия, обученная градиентным спуском.

    Модель: P(y=1|x) = σ(w·x + b)
    """

    learning_rate: float = 0.1
    n_iterations: int = 1000
    l2: float = 0.0
    fit_intercept: bool = True
    tol: float = 1e-6
    random_state: Optional[int] = 42

    weights_: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    bias_: float = field(default=0.0, init=False, repr=False)
    loss_history_: List[float] = field(default_factory=list, init=False, repr=False)
    n_iter_: int = field(default=0, init=False, repr=False)

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return X
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.hstack([ones, X])

    def _loss(self, y: np.ndarray, y_prob: np.ndarray, theta: np.ndarray) -> float:
        eps = 1e-12
        y_prob = np.clip(y_prob, eps, 1.0 - eps)
        n = y.shape[0]
        log_loss = -np.mean(y * np.log(y_prob) + (1.0 - y) * np.log(1.0 - y_prob))
        if self.l2 > 0:
            # не штрафуем свободный член
            start = 1 if self.fit_intercept else 0
            log_loss += (self.l2 / (2.0 * n)) * np.sum(theta[start:] ** 2)
        return float(log_loss)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        """Обучает модель на матрице признаков X и векторе меток y ∈ {0, 1}."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError("X должен быть двумерным массивом")
        if set(np.unique(y)) - {0.0, 1.0}:
            raise ValueError("y должен содержать только метки 0 и 1")

        X_design = self._add_intercept(X)
        n_samples, n_features = X_design.shape

        rng = np.random.default_rng(self.random_state)
        theta = rng.normal(scale=0.01, size=n_features)

        self.loss_history_ = []
        prev_loss = np.inf

        for iteration in range(1, self.n_iterations + 1):
            logits = X_design @ theta
            y_prob = sigmoid(logits)
            error = y_prob - y

            gradient = (X_design.T @ error) / n_samples
            if self.l2 > 0:
                start = 1 if self.fit_intercept else 0
                reg = np.zeros_like(theta)
                reg[start:] = (self.l2 / n_samples) * theta[start:]
                gradient = gradient + reg

            theta = theta - self.learning_rate * gradient
            loss = self._loss(y, y_prob, theta)
            self.loss_history_.append(loss)

            if abs(prev_loss - loss) < self.tol:
                self.n_iter_ = iteration
                break
            prev_loss = loss
        else:
            self.n_iter_ = self.n_iterations

        if self.fit_intercept:
            self.bias_ = float(theta[0])
            self.weights_ = theta[1:].copy()
        else:
            self.bias_ = 0.0
            self.weights_ = theta.copy()
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите fit().")
        X = np.asarray(X, dtype=float)
        return X @ self.weights_ + self.bias_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Возвращает вероятности классов [P(y=0), P(y=1)]."""
        p1 = sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
