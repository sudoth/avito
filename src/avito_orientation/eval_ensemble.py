from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from avito_orientation.dataset import OrientationDataset
from avito_orientation.model import create_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = [
    (
        "convnext_dinov3_base",
        "models/convnext_dinov3_base_e1_w768_seed42.pt",
    ),
    (
        "convnext_dinov3_small",
        "models/convnext_dinov3_small_e1_w768_seed42.pt",
    ),
    (
        "convnextv2_base",
        "models/convnextv2_base_e1_w768_seed42.pt",
    ),
]


@torch.no_grad()
def predict_tta(model, loader):
    model.eval()
    all_probs = []
    all_labels = []

    for images, labels in tqdm(loader, leave=False):
        images = images.to(DEVICE, non_blocking=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(DEVICE == "cuda"),
        ):
            probs = torch.sigmoid(model(images).squeeze(1))

        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)

    p0 = probs[0::2]
    p180 = probs[1::2]

    # proper 180° TTA
    p_tta = (p0 + (1.0 - p180)) / 2.0

    y = labels[0::2]

    return p_tta, y


def main():
    val_hf = load_dataset(
        "MiXaiLL76/ICDAR2015_OCR",
        split="train",
    )

    val_dataset = OrientationDataset(
        val_hf,
        height=64,
        max_width=768,
        train=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    predictions = []
    labels = None

    for model_name, checkpoint_path in MODELS:
        print(f"\nLoading {model_name}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        model = create_model(model_name).to(DEVICE)
        model.load_state_dict(checkpoint["model"])

        probs, y = predict_tta(model, val_loader)

        brier = ((probs - y) ** 2).mean().item()

        print(
            f"{model_name}: "
            f"brier={brier:.6f} | "
            f"score={1-brier:.6f}"
        )

        predictions.append(probs)
        labels = y

        del model
        torch.cuda.empty_cache()

    p_dino_base, p_dino_small, p_v2_base = predictions

    def score_probs(probs):
        brier = ((probs - labels) ** 2).mean().item()
        return brier, 1.0 - brier

    # ---------------------------------------------------------
    # DINO Base + DINO Small
    # ---------------------------------------------------------
    best_bs = None

    for i in range(101):
        w_base = i / 100
        w_small = 1.0 - w_base

        probs = (
            w_base * p_dino_base
            + w_small * p_dino_small
        )

        brier, score = score_probs(probs)

        if best_bs is None or score > best_bs[0]:
            best_bs = (score, brier, w_base, w_small)

    score, brier, w_base, w_small = best_bs

    print("\nBEST: DINO Base + DINO Small")
    print(f"DINO Base:  {w_base:.2f}")
    print(f"DINO Small: {w_small:.2f}")
    print(f"Brier: {brier:.6f}")
    print(f"Score: {score:.6f}")

    # ---------------------------------------------------------
    # DINO Base + ConvNeXtV2 Base
    # ---------------------------------------------------------
    best_bv2 = None

    for i in range(101):
        w_dino = i / 100
        w_v2 = 1.0 - w_dino

        probs = (
            w_dino * p_dino_base
            + w_v2 * p_v2_base
        )

        brier, score = score_probs(probs)

        if best_bv2 is None or score > best_bv2[0]:
            best_bv2 = (score, brier, w_dino, w_v2)

    score, brier, w_dino, w_v2 = best_bv2

    print("\nBEST: DINO Base + ConvNeXtV2 Base")
    print(f"DINO Base:       {w_dino:.2f}")
    print(f"ConvNeXtV2 Base: {w_v2:.2f}")
    print(f"Brier: {brier:.6f}")
    print(f"Score: {score:.6f}")

    # ---------------------------------------------------------
    # Three-model ensemble
    # weights with step 0.05
    # ---------------------------------------------------------
    best_three = None

    for i in range(21):
        w_base = i / 20

        for j in range(21 - i):
            w_small = j / 20
            w_v2 = 1.0 - w_base - w_small

            probs = (
                w_base * p_dino_base
                + w_small * p_dino_small
                + w_v2 * p_v2_base
            )

            brier, score = score_probs(probs)

            if best_three is None or score > best_three[0]:
                best_three = (
                    score,
                    brier,
                    w_base,
                    w_small,
                    w_v2,
                )

    score, brier, w_base, w_small, w_v2 = best_three

    print("\nBEST: THREE MODELS")
    print(f"DINO Base:       {w_base:.2f}")
    print(f"DINO Small:      {w_small:.2f}")
    print(f"ConvNeXtV2 Base: {w_v2:.2f}")
    print(f"Brier: {brier:.6f}")
    print(f"Score: {score:.6f}")

    # ---------------------------------------------------------
    # Simple equal-weight ensembles for reference
    # ---------------------------------------------------------
    probs = 0.5 * p_dino_base + 0.5 * p_dino_small
    brier, score = score_probs(probs)

    print("\n50/50: DINO Base + DINO Small")
    print(f"Brier: {brier:.6f}")
    print(f"Score: {score:.6f}")

    probs = (
        p_dino_base
        + p_dino_small
        + p_v2_base
    ) / 3.0

    brier, score = score_probs(probs)

    print("\n1/3 EACH")
    print(f"Brier: {brier:.6f}")
    print(f"Score: {score:.6f}")


if __name__ == "__main__":
    main()