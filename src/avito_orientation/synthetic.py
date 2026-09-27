import random
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset


WORDS = [
    "автомобиль", "доставка", "бесплатно", "гарантия",
    "производитель", "оригинальный", "комплект",
    "размер", "модель", "товар", "новый", "цена",
    "скидка", "заказ", "магазин", "качество",
    "мощность", "двигатель", "продажа", "наличие",
    "упаковка", "материал", "цвет", "серия",
    "Москва", "Петербург", "Россия",
]

PHRASES = [
    "В наличии",
    "Бесплатная доставка",
    "Российский производитель",
    "Гарантия 12 месяцев",
    "Оригинальный товар",
    "Новая модель",
    "Высокое качество",
    "Комплект поставки",
    "Сделано в России",
    "Цена за комплект",
    "Для автомобилей",
    "Официальный магазин",
    "Быстрая доставка",
    "Срок службы 5 лет",
    "Не является офертой",
]

def random_color(rng, low, high):
    return tuple(
        rng.randint(low, high)
        for _ in range(3)
    )


def get_cyrillic_fonts():
    result = subprocess.run(
        ["fc-list", ":lang=ru", "file"],
        capture_output=True,
        text=True,
        check=True,
    )

    fonts = []

    for line in result.stdout.splitlines():
        path = line.split(":")[0]

        if path.endswith((".ttf", ".otf")):
            fonts.append(path)

    return sorted(set(fonts))


class SyntheticCyrillicDataset(Dataset):
    def __init__(
        self,
        size=50_000,
        fonts=None,
        seed=42,
    ):
        self.size = size
        self.fonts = fonts or get_cyrillic_fonts()
        self.seed = seed

    def __len__(self):
        return self.size

    def make_text(self, rng):
        kind = rng.random()

        if kind < 0.30:
            text = rng.choice(WORDS)

        elif kind < 0.60:
            text = rng.choice(PHRASES)

        elif kind < 0.75:
            text = f"Модель {rng.randint(100, 9999)}"

        elif kind < 0.85:
            text = (
                f"{rng.randint(50, 500)}x"
                f"{rng.randint(20, 300)}x"
                f"{rng.randint(10, 200)} мм"
            )

        elif kind < 0.93:
            text = f"{rng.randint(100, 99999):,} ₽".replace(",", " ")

        else:
            text = (
                f"{rng.choice(['А', 'Б', 'В', 'М', 'К'])}"
                f"{rng.randint(10, 999)}-"
                f"{rng.randint(1, 99)}"
            )

        # Иногда uppercase — в Avito такого много.
        if rng.random() < 0.25:
            text = text.upper()

        return text

    def __getitem__(self, idx):
        # Воспроизводимо для каждого idx, но выборка большая.
        rng = random.Random(self.seed + idx)

        text = self.make_text(rng)

        font_path = rng.choice(self.fonts)
        font_size = rng.randint(18, 72)

        try:
            font = ImageFont.truetype(
                font_path,
                font_size,
            )
        except OSError:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                font_size,
            )

        # Не только белый фон.
        if rng.random() < 0.65:
            # Светлый фон, тёмный текст
            bg = random_color(rng, 180, 255)
            fg = random_color(rng, 0, 90)

        elif rng.random() < 0.75:
            # Тёмный фон, светлый текст
            bg = random_color(rng, 0, 80)
            fg = random_color(rng, 175, 255)

        else:
            # Более насыщенные цветовые комбинации
            bg = random_color(rng, 20, 235)
            fg = random_color(rng, 20, 235)

            # Не допускаем почти одинаковый foreground/background
            while sum(
                abs(a - b)
                for a, b in zip(bg, fg)
            ) < 180:
                fg = random_color(rng, 20, 235)

        dummy = Image.new("RGB", (10, 10))
        draw = ImageDraw.Draw(dummy)

        bbox = draw.textbbox(
            (0, 0),
            text,
            font=font,
        )

        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        pad_x = rng.randint(3, 20)
        pad_y = rng.randint(2, 10)

        image = Image.new(
            "RGB",
            (
                text_w + 2 * pad_x,
                text_h + 2 * pad_y,
            ),
            bg,
        )

        draw = ImageDraw.Draw(image)

        draw.text(
            (
                pad_x - bbox[0],
                pad_y - bbox[1],
            ),
            text,
            font=font,
            fill=fg,
        )

        return {
            "image": image,
            "text": text,
        }


class MixedDataset(Dataset):
    def __init__(
        self,
        real_dataset,
        synthetic_dataset,
        synthetic_fraction=0.25,
    ):
        self.real_dataset = real_dataset
        self.synthetic_dataset = synthetic_dataset
        self.synthetic_fraction = synthetic_fraction

    def __len__(self):
        return len(self.real_dataset)

    def __getitem__(self, idx):
        if random.random() < self.synthetic_fraction:
            synthetic_idx = random.randrange(
                len(self.synthetic_dataset)
            )
            return self.synthetic_dataset[synthetic_idx]

        real_idx = random.randrange(
            len(self.real_dataset)
        )
        return self.real_dataset[real_idx]