# Приложение Б. Код подготовки данных для апробации промптов

В приложении приведён код на Python, которым готовились данные и метрики
для промптов интерпретации логистической регрессии (раздел 2.3 курсовой работы).

## Датасет

Spector & Mazzeo (`statsmodels.datasets.spector`): n = 32,
признаки GPA, TUCE, PSI, цель GRADE (0/1).

Разбиение: train/test 70/30, `stratify=y`, `random_state=42`
(train = 22, test = 10).

## Зависимости

```bash
pip install -r requirements.txt
```

## Скрипты

| Файл | Назначение |
|------|------------|
| [`scripts/logit_spector_basic.py`](../scripts/logit_spector_basic.py) | Б.1 — `statsmodels.Logit` + метрики sklearn |
| [`scripts/logit_spector_compare.py`](../scripts/logit_spector_compare.py) | Б.2 — сравнение Logit и `sklearn.LogisticRegression` |
| [`scripts/prepare_prompt_data_blocks.py`](../scripts/prepare_prompt_data_blocks.py) | Б.3 — печать блоков «ДАННЫЕ» для всех промптов |

### Запуск

```bash
python scripts/logit_spector_basic.py
python scripts/logit_spector_compare.py
python scripts/prepare_prompt_data_blocks.py
```

После запуска скрипта Б.3 скопируйте нужный блок в конец соответствующего
промпта из [Приложения А](prilozhenie-a-polnye-prompty.md).

## Соответствие кода и промптов

| Промпт | Скрипт | Что брать из вывода |
|--------|--------|---------------------|
| Интерпретация результатов | Б.1 или Б.3 | summary, OR, метрики, CM |
| Генерация кода | Б.2 (образец) | весь скрипт |
| Сравнение statsmodels и sklearn | Б.2 или Б.3 | оба набора коэффициентов и метрик |
| Академический стиль | Б.1 или Б.3 | summary + метрики + CM |
| Рецензирование модели | Б.1 или Б.3 | Pseudo R², p-values, метрики |
| Порог и качество решений | Б.1 или Б.3 | CM, Prec/Rec, AUC, порог 0.5 |
| Бизнес / простой язык | Б.3 | «человеческий» блок данных |
