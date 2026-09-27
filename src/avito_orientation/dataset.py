import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


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

    def __len__(self):
        if self.train:
            return len(self.dataset)

        # в валидационном датасете присутствует и прямое и перевернутое изображение
        return 2 * len(self.dataset)

    def get_image_and_label(self, idx):
        if self.train:
            image = self.dataset[idx]["image"].convert("RGB")

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

        # сжимаем слишком широкие изображения (!)
        new_w = min(new_w, self.max_width)

        image = image.resize(
            (new_w, self.height),
            Image.Resampling.BILINEAR,
        )

        # Нормализуем как в imagenet
        tensor = TF.to_tensor(image)

        # пока что просто паддинг белым
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

        image = self.resize_and_pad(image)

        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        return image, torch.tensor(label, dtype=torch.float32)