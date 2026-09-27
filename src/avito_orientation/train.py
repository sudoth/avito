from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from avito_orientation.dataset import OrientationDataset
from avito_orientation.model import create_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_BATCH_SIZE = 256
VAL_BATCH_SIZE = 256
NUM_WORKERS = 8
EPOCHS = 5
LR = 3e-4
WEIGHT_DECAY = 1e-4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    sum_squared_error = 0.0
    correct = 0
    n = 0

    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits)

        sum_squared_error += ((probs - labels) ** 2).sum().item()
        correct += ((probs >= 0.5) == labels.bool()).sum().item()
        n += labels.numel()

    brier = sum_squared_error / n
    accuracy = correct / n

    return {
        "brier": brier,
        "score": 1.0 - brier,
        "accuracy": accuracy,
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

    train_dataset = OrientationDataset(
        train_hf,
        train=True,
    )

    val_dataset = OrientationDataset(
        val_hf,
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

    model = create_model().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
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

        metrics = evaluate(model, val_loader)

        print(
            f"\nEpoch {epoch}: "
            f"train_loss={running_loss / n:.5f} | "
            f"val_brier={metrics['brier']:.6f} | "
            f"score={metrics['score']:.6f} | "
            f"accuracy={metrics['accuracy']:.4%}"
        )

        if metrics["brier"] < best_brier:
            best_brier = metrics["brier"]

            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                },
                MODEL_DIR / "efficientnet_v2_s_e0.pt",
            )

            print("Saved new best checkpoint")


if __name__ == "__main__":
    main()