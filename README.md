# Курсовой проект: логистическая регрессия на Python

Репозиторий содержит реализацию **бинарной логистической регрессии** с нуля (градиентный спуск) и сравнение с `sklearn.linear_model.LogisticRegression` на датасете **Breast Cancer Wisconsin**.

## Цель работы

1. Изучить математическую модель логистической регрессии.
2. Реализовать обучение методом градиентного спуска.
3. Оценить качество классификации (Accuracy, Precision, Recall, F1, ROC-AUC).
4. Сравнить собственную реализацию с библиотечной.

## Краткая теория

Для объекта с признаками \(x \in \mathbb{R}^d\) модель оценивает вероятность положительного класса:

\[
P(y=1 \mid x) = \sigma(w^\top x + b), \quad
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

Функция потерь — бинарная кросс-энтропия (log-loss):

\[
L(w, b) = -\frac{1}{n}\sum_{i=1}^{n}
\left[
y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)
\right]
+ \frac{\lambda}{2n}\|w\|^2
\]

Параметры обновляются градиентным спуском:

\[
\theta \leftarrow \theta - \eta \nabla_\theta L
\]

Решение о классе принимается по порогу \(0.5\): \(\hat{y} = [\hat{p} \ge 0.5]\).

## Структура проекта

```
├── README.md
├── requirements.txt
├── data/                 # CSV датасета (создаётся при запуске)
├── results/              # графики и metrics.json
├── notebooks/
│   └── logistic_regression.ipynb
└── src/
    ├── logistic_regression.py  # собственная модель
    ├── metrics.py              # метрики
    ├── data.py                 # загрузка и препроцессинг
    ├── visualize.py            # графики
    └── main.py                 # полный пайплайн
```

## Установка и запуск

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# полный эксперимент
python -m src.main
```

После запуска в `results/` появятся:

- `loss_curve.png` — сходимость log-loss
- `cm_scratch.png` / `cm_sklearn.png` — матрицы ошибок
- `roc_curves.png` — ROC-кривые обеих моделей
- `feature_weights.png` — наиболее влиятельные признаки
- `metrics.json` — числовые метрики

## Датасет

**Breast Cancer Wisconsin Diagnostic** (из `sklearn.datasets`):

- 569 объектов, 30 числовых признаков
- классы: `0` — malignant, `1` — benign
- признаки стандартизируются (`StandardScaler`)
- разбиение train/test = 75% / 25%, со стратификацией

## Ожидаемый результат

Собственная модель и `sklearn` дают близкое качество (Accuracy и ROC-AUC обычно > 0.95 на этом датасете), что подтверждает корректность реализации градиентного спуска.

## Автор

Курсовой проект по теме логистической регрессии на Python.
