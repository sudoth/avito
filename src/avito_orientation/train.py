import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
import numpy as np
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

TRAIN_BATCH_SIZE = 128
VAL_BATCH_SIZE = 128
NUM_WORKERS = 8
EPOCHS = 8
LR = 1e-4
WEIGHT_DECAY = 1e-4
MODEL_NAME = "convnext_tiny"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
    choices=[
        "efficientnet_v2_s",
        "convnext_tiny",
        "convnextv2_tiny",
        "mobilenetv4",
    ],
        default="efficientnet_v2_s",
    )

    parser.add_argument(
        "--protocol",
        choices=["e1", "e2"],
        default="e1",
    )

    parser.add_argument(
        "--max-width",
        type=int,
        default=768,
    )

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


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

        # with torch.autocast(
        #     device_type="cuda",
        #     dtype=torch.bfloat16,
        # ):
        logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits)

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
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    MODEL_NAME = args.model
    PROTOCOL = args.protocol
    MAX_WIDTH = args.max_width

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

    if PROTOCOL == "e1":
        train_source = train_hf

    elif PROTOCOL == "e2":
        all_fonts = get_cyrillic_fonts()

        rng = random.Random(42)
        rng.shuffle(all_fonts)

        split = int(len(all_fonts) * 0.8)
        train_fonts = all_fonts[:split]
        val_fonts = all_fonts[split:]

        synthetic_train = SyntheticCyrillicDataset(
            size=100_000,
            fonts=train_fonts,
            seed=42,
        )

        train_source = MixedDataset(
            real_dataset=train_hf,
            synthetic_dataset=synthetic_train,
            synthetic_fraction=0.25,
        )

    train_dataset = OrientationDataset(
        train_source,
        height=64,
        max_width=MAX_WIDTH,
        train=True,
    )

    val_dataset = OrientationDataset(
        val_hf,
        height=64,
        max_width=MAX_WIDTH,
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

    model = create_model(MODEL_NAME).to(DEVICE)
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

            # with torch.autocast(
            #     device_type="cuda",
            #     dtype=torch.bfloat16,
            # ):
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)

            running_loss += loss.item() * batch_size
            n += batch_size

            progress.set_postfix(loss=f"{running_loss / n:.4f}")

        metrics = evaluate(
            model,
            val_loader,
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch {epoch}: "
            f"train_loss={running_loss / n:.5f} | "
            f"score={metrics['score']:.6f} "
            f"accuracy={metrics['accuracy']:.2%} | "
            f"lr={current_lr:.2e}"
        )

        if metrics["brier"] < best_brier:
            best_brier = metrics["brier"]

            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                    "model_name": MODEL_NAME,
                    "protocol": PROTOCOL,
                    "max_width": MAX_WIDTH,
                    "seed": args.seed,
                },
                MODEL_DIR / f"{MODEL_NAME}_{PROTOCOL}_w{MAX_WIDTH}_seed{args.seed}.pt.pt",
            )


if __name__ == "__main__":
    main()
