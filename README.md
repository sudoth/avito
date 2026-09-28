# Avito OCR Text Orientation Classification

Решение тестового задания Avito Data Science Bootcamp: для каждого OCR-кропа необходимо предсказать вероятность того, что текст повёрнут на 180°.

Метрика:

$$
\text{Score} = 1 - \frac{1}{N}\sum_i (p_i-y_i)^2
$$

Финальное решение — взвешенное объединение трёх моделей:

- ConvNeXt DINOv3 Base, E1 — `0.45`;
- ConvNeXt DINOv3 Base, E2 с синтетическими кириллическими данными — `0.20`;
- ConvNeXtV2 Base, E1 — `0.35`.

Дополнительно используются поворот на 180° и три фрагмента для строк с `aspect ratio >= 20`.

## Что смотреть

Основной отчёт с описанием данных, валидации, экспериментов и финального решения:

[`notebooks/solution.ipynb`](notebooks/solution.ipynb)

Финальный inference:

[`src/avito_orientation/predict.py`](src/avito_orientation/predict.py)

Финальный файл с предсказаниями:

[`submissions/submission.csv`](submissions/submission.csv)

## Подход

Размеченной обучающей выборки в задании нет, поэтому используются открытые OCR-датасеты:

- `MiXaiLL76/TextOCR_OCR` — обучение;
- `MiXaiLL76/ICDAR2015_OCR` — независимая validation.

Для каждого исходного изображения ориентация создаётся автоматически:
исходное изображение имеет класс `0`, изображение после поворота на 180° — класс `1`.

Финальное решение состоит из двух моделей:

- ConvNeXt Base с DINOv3 pretraining — вес `0.65`;
- ConvNeXtV2 Base — вес `0.35`.

Для каждой модели используется 180° Test Time Augmentation:

$$
p_{\mathrm{TTA}}(x)
=
\frac{p(x) + 1 - p(R_{180}(x))}{2}.
$$

Большинство изображений масштабируется до высоты 64 px с максимальной шириной 768 px.

Для очень длинных изображений (`aspect ratio >= 20`) используются три фрагмента — левый, центральный и правый. Предсказания по ним усредняются. Это позволяет избежать сильного горизонтального сжатия длинных строк.

## Validation

Validation полностью отделена от training data.

Для каждого изображения ICDAR2015 используются обе ориентации, поэтому validation:

- детерминирована;
- сбалансирована по классам;
- не использует разметку тестовой выборки.

Основные результаты:

| Модель | Validation score с TTA |
|---|---:|
| ConvNeXtV2 Tiny | 0.961414 |
| ConvNeXtV2 Base | 0.964727 |
| ConvNeXt DINOv3 Small | 0.965725 |
| ConvNeXt DINOv3 Base | 0.968861 |
| DINOv3 Base + ConvNeXtV2 Base | 0.970590 |

Трёхмодельный ансамбль давал `0.970846`, но выигрыш `0.000256` не оправдывал дополнительную модель на inference, поэтому использован двухмодельный вариант.

## Воспроизведение

Использовался Python 3.12 и `uv`.

Установка зависимостей:

```bash
uv sync --frozen
```

Тестовые изображения из задания необходимо расположить в:

```text
data/test/
```

Финальные checkpoints должны находиться в:

```text
models/convnext_dinov3_base_e1_w768_seed42.pt
models/convnextv2_base_e1_w768_seed42.pt
```

Запуск inference:

```bash
uv run python -m avito_orientation.predict
```

Результат:

```text
submissions/submission.csv
```

## Структура проекта

```text
notebooks/solution.ipynb
    основной отчёт

src/avito_orientation/
    dataset.py
    model.py
    train.py
    eval.py
    eval_ensemble.py
    eval_wide.py
    predict.py
    synthetic.py

submissions/submission.csv
    финальный submission
```

## Использованные библиотеки и модели

- PyTorch / torchvision
- timm
- Albumentations
- Hugging Face Datasets
- pretrained ConvNeXt DINOv3
- pretrained ConvNeXtV2

Внешние inference API не используются.