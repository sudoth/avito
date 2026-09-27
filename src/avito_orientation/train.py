import random
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from avito_orientation.dataset import OrientationDataset
from avito_orientation.model import create_model
from avito_orientation.synthetic import (
    MixedDataset,
    SyntheticCyrillicDataset,
    get_cyrillic_fonts,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_BATCH_SIZE = 256
VAL_BATCH_SIZE = 256
NUM_WORKERS = 8
EPOCHS = 8
LR = 1e-4
WEIGHT_DECAY = 1e-4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    confidence_sum = 0.0
    sum_squared_error = 0.0
    correct = 0
    n = 0

    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            logits1 = model(images).squeeze(1)
            probs1 = torch.sigmoid(logits1)

            rotated = torch.rot90(
                images,
                k=2,
                dims=(-2, -1),
            )

            logits2 = model(rotated).squeeze(1)
            probs2 = torch.sigmoid(logits2)

            probs = (probs1 + (1.0 - probs2)) / 2.0

        confidence = torch.maximum(probs, 1 - probs)
        confidence_sum += confidence.sum().item()
        sum_squared_error += ((probs - labels) ** 2).sum().item()
        correct += ((probs >= 0.5) == labels.bool()).sum().item()
        n += labels.numel()

    brier = sum_squared_error / n
    accuracy = correct / n
    mean_confidence = confidence_sum / n

    return {
        "brier": brier,
        "score": 1.0 - brier,
        "accuracy": accuracy,
        "mean_confidence": mean_confidence,
    }


def main():
    torch.set_float32_matmul_precision("high")
    MODEL_DIR.mkdir(exist_ok=True)

    print(f"Device: {DEVICE}")

    train_hf = load_dataset(
        "MiXaiLL76/TextOCR_OCR",
        split="train",
    )

    val_hf = load_dataset(
        "MiXaiLL76/ICDAR2015_OCR",
        split="train",
    )

    all_fonts = get_cyrillic_fonts()

    rng = random.Random(42)
    rng.shuffle(all_fonts)

    split = int(len(all_fonts) * 0.8)

    train_fonts = all_fonts[:split]
    val_fonts = all_fonts[split:]

    print(
        f"Cyrillic fonts: "
        f"{len(train_fonts)} train / "
        f"{len(val_fonts)} val"
    )

    synthetic_train = SyntheticCyrillicDataset(
        size=100_000,
        fonts=train_fonts,
        seed=42,
    )

    mixed_train = MixedDataset(
        real_dataset=train_hf,
        synthetic_dataset=synthetic_train,
        synthetic_fraction=0.25,
    )

    train_dataset = OrientationDataset(
        mixed_train,
        train=True,
    )

    val_dataset = OrientationDataset(
        val_hf,
        train=False,
    )

    synthetic_val_raw = SyntheticCyrillicDataset(
        size=5_000,
        fonts=val_fonts,
        seed=1_000_000,
    )

    cyrillic_val_dataset = OrientationDataset(
        synthetic_val_raw,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    cyrillic_val_loader = DataLoader(
        cyrillic_val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    model = create_model().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6,
)

    best_brier = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        running_loss = 0.0
        n = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{EPOCHS}",
        )

        for images, labels in progress:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                logits = model(images).squeeze(1)
                loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)

            running_loss += loss.item() * batch_size
            n += batch_size

            progress.set_postfix(
                loss=f"{running_loss / n:.4f}"
            )

        icdar_metrics = evaluate(
            model,
            val_loader,
        )

        cyr_metrics = evaluate(
            model,
            cyrillic_val_loader,
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch {epoch}: "
            f"train_loss={running_loss / n:.5f} | "
            f"ICDAR score={icdar_metrics['score']:.6f} "
            f"accuracy={icdar_metrics['accuracy']:.2%} | "
            f"CYR score={cyr_metrics['score']:.6f} "
            f"accuracy={cyr_metrics['accuracy']:.2%} | "
            f"lr={current_lr:.2e}"
        )

        if icdar_metrics["brier"] < best_brier:
            best_brier = icdar_metrics["brier"]

            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "icdar_metrics": icdar_metrics,
                    "cyrillic_metrics": cyr_metrics,
                },
                MODEL_DIR / "efficientnet_v2_s_e2.pt",
            )

            print("Saved new best E2 checkpoint")


if __name__ == "__main__":
    main()