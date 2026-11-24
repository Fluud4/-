# Курсовой проект: логистическая регрессия на Python

Репозиторий содержит материалы по разработке и апробации промптов для интерпретации логистической регрессии (`statsmodels` / `scikit-learn`).

## Документы

- [`docs/prilozhenie-a-polnye-prompty.md`](docs/prilozhenie-a-polnye-prompty.md) — **Приложение А**: полные тексты всех промптов (без сокращений)
- [`docs/prilozhenie-a-polnye-prompty.txt`](docs/prilozhenie-a-polnye-prompty.txt) — тот же текст в `.txt`
- [`docs/prompts/`](docs/prompts/) — отдельные файлы шаблонов промптов
- [`docs/prilozhenie-b-podgotovka-dannyh.md`](docs/prilozhenie-b-podgotovka-dannyh.md) — **Приложение Б**: описание кода подготовки данных

## Код (Приложение Б)

```bash
pip install -r requirements.txt
python scripts/logit_spector_basic.py
python scripts/logit_spector_compare.py
python scripts/prepare_prompt_data_blocks.py
```

| Скрипт | Назначение |
|--------|------------|
| `scripts/logit_spector_basic.py` | Logit + метрики sklearn |
| `scripts/logit_spector_compare.py` | сравнение statsmodels и sklearn |
| `scripts/prepare_prompt_data_blocks.py` | блоки «ДАННЫЕ» для промптов |

Датасет апробации: Spector & Mazzeo (`statsmodels.datasets.spector`).
