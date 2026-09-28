import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from avito_orientation.dataset import OrientationDataset
from avito_orientation.model import create_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINTS = [
    "models/convnextv2_tiny_e1_w768.pt",
    "models/convnextv2_tiny_e2_w768.pt",
    "models/convnextv2_tiny_e1_w768_seed1337.pt",
]

HEIGHT = 64
MAX_WIDTH = 768

CHUNK_AR = 12.0
OVERLAP = 0.25

N_PER_GROUP = 500

AR_GROUPS = [
    (12, 15),
    (15, 20),
    (20, 30),
    (30, 45),
]

SEED = 42

preprocessor = OrientationDataset(
    dataset=[],
    height=HEIGHT,
    max_width=MAX_WIDTH,
    train=False,
)


def normalize(x):
    return TF.normalize(
        x,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )


def preprocess_squash(image):
    x = preprocessor.resize_and_pad(
        image.convert("RGB")
    )
    return normalize(x)


def preprocess_crop(image):
    x = preprocessor.resize_and_pad(
        image.convert("RGB")
    )
    return normalize(x)

def crop_width_for_image(image):
    _, h = image.size
    return round(CHUNK_AR * h)


def center_crop(image):
    image = image.convert("RGB")

    w, h = image.size
    crop_w = min(
        crop_width_for_image(image),
        w,
    )

    left = (w - crop_w) // 2

    return image.crop(
        (
            left,
            0,
            left + crop_w,
            h,
        )
    )


def three_crops(image):
    """
    Left + center + right.
    """
    image = image.convert("RGB")

    w, h = image.size
    crop_w = min(
        crop_width_for_image(image),
        w,
    )

    if crop_w >= w:
        return [image]

    starts = [
        0,
        (w - crop_w) // 2,
        w - crop_w,
    ]

    starts = list(dict.fromkeys(starts))

    return [
        image.crop(
            (
                start,
                0,
                start + crop_w,
                h,
            )
        )
        for start in starts
    ]


def make_chunks(image):
    image = image.convert("RGB")

    w, h = image.size

    chunk_w = round(
        CHUNK_AR * h
    )

    if w <= chunk_w:
        return [image]

    stride = round(
        chunk_w * (1 - OVERLAP)
    )

    starts = list(
        range(
            0,
            w - chunk_w + 1,
            stride,
        )
    )

    last_start = w - chunk_w

    if not starts or starts[-1] != last_start:
        starts.append(last_start)

    return [
        image.crop(
            (
                start,
                0,
                start + chunk_w,
                h,
            )
        )
        for start in starts
    ]

def concat_images(images, target_ar):

    prepared = []

    target_h = 64

    for image in images:
        image = image.convert("RGB")

        w, h = image.size

        new_w = max(
            1,
            round(w * target_h / h),
        )

        image = image.resize(
            (new_w, target_h),
            Image.Resampling.BILINEAR,
        )

        prepared.append(image)

    target_w = round(
        target_ar * target_h
    )

    canvas = Image.new(
        "RGB",
        (target_w, target_h),
        "white",
    )

    x = 0

    for image in prepared:
        if x >= target_w:
            break

        remaining = target_w - x

        if image.width <= remaining:
            canvas.paste(
                image,
                (x, 0),
            )
            x += image.width

        else:
            part = image.crop(
                (
                    0,
                    0,
                    remaining,
                    target_h,
                )
            )

            canvas.paste(
                part,
                (x, 0),
            )

            x = target_w

    return canvas


def make_long_dataset(source):
    rng = random.Random(SEED)

    examples = []

    for ar_low, ar_high in AR_GROUPS:

        print(
            f"Generating AR {ar_low}-{ar_high}..."
        )

        for _ in range(N_PER_GROUP):

            target_ar = rng.uniform(
                ar_low,
                ar_high,
            )

            pieces = []

            total_ar = 0.0

            while total_ar < target_ar + 3:
                idx = rng.randrange(
                    len(source)
                )

                image = source[idx]["image"].convert(
                    "RGB"
                )

                pieces.append(image)

                w, h = image.size
                total_ar += w / h

            image = concat_images(
                pieces,
                target_ar,
            )

            label = rng.randint(0, 1)

            if label == 1:
                image = image.rotate(180)

            examples.append(
                {
                    "image": image,
                    "label": label,
                    "ar": target_ar,
                    "group": f"{ar_low}-{ar_high}",
                }
            )

    return examples

def load_models():
    models = []

    for path in CHECKPOINTS:

        checkpoint = torch.load(
            PROJECT_ROOT / path,
            map_location="cpu",
            weights_only=True,
        )

        model = create_model(
            "convnextv2_tiny"
        ).to(DEVICE)

        model.load_state_dict(
            checkpoint["model"]
        )

        model.eval()

        print(
            f"Loaded {path} | "
            f"epoch={checkpoint['epoch']}"
        )

        models.append(model)

    return models

@torch.inference_mode()
def model_tta_on_views(model, views):

    inputs = []

    for view in views:
        inputs.append(
            preprocess_crop(view)
        )

        inputs.append(
            preprocess_crop(
                view.rotate(180)
            )
        )

    x = torch.stack(inputs).to(
        DEVICE
    )

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=(DEVICE == "cuda"),
    ):
        logits = model(x).squeeze(1)

    probs = torch.sigmoid(
        logits.float()
    )

    p_original = probs[0::2]
    p_rotated = probs[1::2]

    p_tta = (
        p_original
        + (1.0 - p_rotated)
    ) / 2.0

    return p_tta.mean().item()


@torch.inference_mode()
def model_squash_tta(model, image):

    x0 = preprocess_squash(
        image
    )

    x180 = preprocess_squash(
        image.rotate(180)
    )

    x = torch.stack(
        [x0, x180]
    ).to(DEVICE)

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=(DEVICE == "cuda"),
    ):
        logits = model(x).squeeze(1)

    probs = torch.sigmoid(
        logits.float()
    )

    return (
        probs[0].item()
        + 1.0
        - probs[1].item()
    ) / 2.0


def ensemble_predict(
    models,
    image,
    strategy,
):

    predictions = []

    if strategy == "squash":
        for model in models:
            p = model_squash_tta(
                model,
                image,
            )
            predictions.append(p)

    else:

        if strategy == "center":
            views = [
                center_crop(image)
            ]

        elif strategy == "3crop":
            views = three_crops(
                image
            )

        elif strategy == "chunks":
            views = make_chunks(
                image
            )

        else:
            raise ValueError(
                strategy
            )

        for model in models:
            p = model_tta_on_views(
                model,
                views,
            )

            predictions.append(p)

    return float(
        np.mean(predictions)
    )


def brier(y, p):
    y = np.asarray(
        y,
        dtype=np.float64,
    )

    p = np.asarray(
        p,
        dtype=np.float64,
    )

    return np.mean(
        (p - y) ** 2
    )


def accuracy(y, p):
    y = np.asarray(y)

    p = np.asarray(p)

    return np.mean(
        (p >= 0.5) == y
    )


def print_result(
    name,
    y,
    p,
):
    bs = brier(y, p)

    acc = accuracy(y, p)

    print(
        f"{name:10s} | "
        f"Brier={bs:.6f} | "
        f"Score={1-bs:.6f} | "
        f"Acc={acc*100:.2f}%"
    )



def main():

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    torch.set_float32_matmul_precision(
        "high"
    )

    print("Loading ICDAR...")

    source = load_dataset(
        "MiXaiLL76/ICDAR2015_OCR",
        split="train",
    )

    examples = make_long_dataset(
        source
    )

    print(
        f"\nGenerated {len(examples)} "
        f"long images."
    )

    models = load_models()

    strategies = [
        "squash",
        "center",
        "3crop",
        "chunks",
    ]

    results = {
        strategy: []
        for strategy in strategies
    }

    labels = []
    groups = []
    ars = []

    for example in tqdm(
        examples,
        desc="Evaluating",
    ):

        image = example["image"]

        labels.append(
            example["label"]
        )

        groups.append(
            example["group"]
        )

        ars.append(
            example["ar"]
        )

        for strategy in strategies:

            p = ensemble_predict(
                models,
                image,
                strategy,
            )

            results[
                strategy
            ].append(p)

    labels_np = np.asarray(labels)
    groups_np = np.asarray(groups)
    ars_np = np.asarray(ars)

    print("\n")
    print("=" * 70)
    print("RESULTS BY AR")
    print("=" * 70)

    for group in [
        "12-15",
        "15-20",
        "20-30",
        "30-45",
    ]:

        mask = (
            groups_np == group
        )

        print(
            f"\nAR {group} "
            f"(n={mask.sum()})"
        )

        for strategy in strategies:
            p = np.asarray(
                results[strategy]
            )[mask]

            print_result(
                strategy,
                labels_np[mask],
                p,
            )


    mask20 = ars_np >= 20

    print("\n")
    print("=" * 70)
    print(
        f"AR >= 20 "
        f"(n={mask20.sum()})"
    )
    print("=" * 70)

    for strategy in strategies:

        p = np.asarray(
            results[strategy]
        )[mask20]

        print_result(
            strategy,
            labels_np[mask20],
            p,
        )


    print("\n")
    print("=" * 70)
    print(
        f"ALL WIDE "
        f"(n={len(labels_np)})"
    )
    print("=" * 70)

    for strategy in strategies:

        print_result(
            strategy,
            labels_np,
            results[strategy],
        )


    print("\n")
    print("=" * 70)
    print("AR >= 20: COMPARISON WITH SQUASH")
    print("=" * 70)

    squash = np.asarray(
        results["squash"]
    )[mask20]

    y = labels_np[mask20]

    squash_error = (
        squash - y
    ) ** 2

    for strategy in [
        "center",
        "3crop",
        "chunks",
    ]:

        p = np.asarray(
            results[strategy]
        )[mask20]

        error = (
            p - y
        ) ** 2

        better = np.mean(
            error < squash_error
        )

        worse = np.mean(
            error > squash_error
        )

        improvement = (
            squash_error.mean()
            - error.mean()
        ) / squash_error.mean()

        print(
            f"{strategy:10s} | "
            f"better={better*100:6.2f}% | "
            f"worse={worse*100:6.2f}% | "
            f"Brier improvement="
            f"{improvement*100:+7.2f}%"
        )


if __name__ == "__main__":
    main()