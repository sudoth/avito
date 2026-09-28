import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

import albumentations as A
import cv2


class OrientationDataset(Dataset):
    def __init__(
        self,
        dataset,
        height=64,
        max_width=768,
        train=True,
    ):
        self.dataset = dataset
        self.height = height
        self.max_width = max_width
        self.train = train
        self.augment = None

        if self.train:
            self.augment = A.Compose([
                # геометрические искажения
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(-0.04, 0.04),
                    rotate=(-10, 10),
                    shear=(-5, 5),
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=255,
                    p=0.6,
                ),

                A.Perspective(
                    scale=(0.02, 0.06),
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=255,
                    p=0.25,
                ),

                # проблемы качества
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.MotionBlur(blur_limit=(3, 5)),
                ], p=0.25),

                A.OneOf([
                    A.GaussNoise(std_range=(0.01, 0.05)),
                    A.ISONoise(),
                ], p=0.20),

                A.RandomBrightnessContrast(
                    brightness_limit=0.20,
                    contrast_limit=0.20,
                    p=0.30,
                ),

                A.ImageCompression(
                    quality_range=(35, 95),
                    p=0.25,
                ),

                # уменьшение качества через понижение разрешения
                A.Downscale(
                    scale_range=(0.35, 0.8),
                    p=0.25,
                ),
            ])

    def __len__(self):
        if self.train:
            return len(self.dataset)

        # Для проверки используем обе ориентации каждого изображения.
        # Так выборка получается детерминированной и сбалансированной.
        return 2 * len(self.dataset)

    def get_image_and_label(self, idx):
        if self.train:
            image = self.dataset[idx]["image"].convert("RGB")

            # На обучении класс создаётся автоматически:
            # исходное изображение — 0, повёрнутое на 180° — 1.
            # Поэтому для каждого обращения ориентация выбирается случайно.
            label = random.randint(0, 1)

            if label == 1:
                image = image.rotate(180)

        else:
            source_idx = idx // 2
            label = idx % 2

            image = self.dataset[source_idx]["image"].convert("RGB")

            if label == 1:
                image = image.rotate(180)

        return image, label

    def resize_and_pad(self, image):
        w, h = image.size

        scale = self.height / h
        new_w = round(w * scale)

        # Сохраняем пропорции, пока ширина не превышает max_width.
        # Более длинные строки приходится горизонтально сжимать;
        # поэтому при предсказании для них отдельно используются три фрагмента.
        new_w = min(new_w, self.max_width)

        image = image.resize(
            (new_w, self.height),
            Image.Resampling.BILINEAR,
        )

        # Нормализуем как в imagenet
        tensor = TF.to_tensor(image)

        # Дополняем изображение справа белым до фиксированной ширины.
        output = torch.ones(
            3,
            self.height,
            self.max_width,
            dtype=tensor.dtype,
        )

        output[:, :, :new_w] = tensor

        return output

    def __getitem__(self, idx):
        image, label = self.get_image_and_label(idx)

        if self.augment is not None:
            image_np = np.array(image)
            image_np = self.augment(image=image_np)["image"]
            image = Image.fromarray(image_np)

        image = self.resize_and_pad(image)

        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        return image, torch.tensor(label, dtype=torch.float32)
