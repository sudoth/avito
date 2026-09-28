import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from avito_orientation.dataset import OrientationDataset
from avito_orientation.model import create_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint")

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "efficientnet_v2_s",
            "convnext_tiny",
            "convnextv2_tiny",
            "convnextv2_base",
            "mobilenetv4",
            "caformer_s18",
            "convnext_dinov3_base",
            "convnext_dinov3_small",
        ],
    )

    parser.add_argument(
        "--max-width",
        type=int,
        default=768,
    )

    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    all_probs = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Validation"):
        images = images.to(DEVICE)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(DEVICE == "cuda"),
        ):
            probs = torch.sigmoid(
                model(images).squeeze(1)
            )

        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)


    # В выборке подряд идут две версии одного изображения:
    # исходная с классом 0 и повёрнутая на 180° с классом 1.
    p0 = probs[0::2]
    p180 = probs[1::2]

    y0 = labels[0::2]

    # Вероятность для повёрнутой версии переводим обратно
    # к вероятности класса исходного изображения и усредняем.
    p_tta = (p0 + (1.0 - p180)) / 2.0

    normal_brier = ((p0 - y0) ** 2).mean().item()
    tta_brier = ((p_tta - y0) ** 2).mean().item()
    both_brier = ((probs - labels) ** 2).mean().item()

    normal_acc = (
        (p0 >= 0.5) == y0.bool()
    ).float().mean().item()

    tta_acc = (
        (p_tta >= 0.5) == y0.bool()
    ).float().mean().item()

    # Насколько сильно нарушается ожидаемая симметрия
    # p(x) = 1 - p(R180(x)).
    consistency = (
        p0 - (1.0 - p180)
    ).abs().mean().item()

    return {
        "both_brier": both_brier,
        "normal_brier": normal_brier,
        "normal_acc": normal_acc,
        "tta_brier": tta_brier,
        "tta_acc": tta_acc,
        "consistency": consistency,
    }


def main():
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model_name = checkpoint.get(
        "model_name",
        args.model,
    )

    max_width = checkpoint.get(
        "max_width",
        args.max_width,
    )

    print(
        f"{model_name} | "
        f"epoch={checkpoint['epoch']} | "
        f"width={max_width}"
    )

    model = create_model(model_name).to(DEVICE)
    model.load_state_dict(checkpoint["model"])

    val_hf = load_dataset(
        "MiXaiLL76/ICDAR2015_OCR",
        split="train",
    )

    val_dataset = OrientationDataset(
        val_hf,
        height=64,
        max_width=max_width,
        train=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    metrics = evaluate(model, val_loader)

    print(
        f"normal: "
        f"brier={metrics['normal_brier']:.6f} | "
        f"score={1 - metrics['normal_brier']:.6f} | "
        f"acc={metrics['normal_acc']:.2%}"
    )

    print(
        f"TTA:    "
        f"brier={metrics['tta_brier']:.6f} | "
        f"score={1 - metrics['tta_brier']:.6f} | "
        f"acc={metrics['tta_acc']:.2%}"
    )

    print(
        f"consistency={metrics['consistency']:.6f}"
    )


if __name__ == "__main__":
    main()