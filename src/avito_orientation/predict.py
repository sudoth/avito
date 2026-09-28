import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from avito_orientation.dataset import OrientationDataset
from avito_orientation.model import create_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = PROJECT_ROOT / "data" / "test"
SAMPLE_PATH = PROJECT_ROOT / "submissions" / "sample_submission.csv"

WIDE_AR_THRESHOLD = 20.0
CROP_AR = 12.0


def three_crops(image):
    w, h = image.size
    crop_w = round(CROP_AR * h)

    if w <= crop_w:
        return [image]

    starts = [
        0,
        (w - crop_w) // 2,
        w - crop_w,
    ]

    return [
        image.crop(
            (start, 0, start + crop_w, h)
        )
        for start in starts
    ]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            "models/convnextv2_tiny_e1_w768.pt",
            "models/convnextv2_tiny_e2_w768.pt",
            "models/convnextv2_tiny_e1_w768_seed1337.pt",
        ],
    )

    parser.add_argument(
        "--output",
        default="submissions/convnextv2_3model_tta_3crop.csv",
    )

    return parser.parse_args()


class TestDataset(Dataset):
    def __init__(self, image_ids):
        self.image_ids = image_ids

        self.preprocessor = OrientationDataset(
            dataset=[],
            height=64,
            max_width=768,
            train=False,
        )

    def __len__(self):
        return len(self.image_ids)

    def preprocess(self, image):
        image = self.preprocessor.resize_and_pad(image)

        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        return image

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        image = Image.open(
            TEST_DIR / f"{image_id}.png"
        ).convert("RGB")

        image_180 = image.rotate(180)

        x0 = self.preprocess(image)
        x180 = self.preprocess(image_180)

        return x0, x180


def load_model(path):
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

    return model


@torch.inference_mode()
def predict_model(model, loader):
    predictions = []

    for x0, x180 in tqdm(
        loader,
        desc="Predicting",
    ):
        x0 = x0.to(
            DEVICE,
            non_blocking=True,
        )

        x180 = x180.to(
            DEVICE,
            non_blocking=True,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(DEVICE == "cuda"),
        ):
            z0 = model(x0).squeeze(1)
            z180 = model(x180).squeeze(1)

        p0 = torch.sigmoid(
            z0.float()
        )

        p180 = torch.sigmoid(
            z180.float()
        )

        probs = (
            p0 + (1.0 - p180)
        ) / 2.0

        predictions.append(
            probs.cpu()
        )

    return torch.cat(predictions)


@torch.inference_mode()
def predict_wide_model(model, image_ids, preprocessor):
    predictions = {}

    for image_id in tqdm(
        image_ids,
        desc="Wide 3-crop",
    ):
        image = Image.open(
            TEST_DIR / f"{image_id}.png"
        ).convert("RGB")

        crops = three_crops(image)

        inputs = []

        for crop in crops:
            x0 = preprocessor.preprocess(crop)
            x180 = preprocessor.preprocess(
                crop.rotate(180)
            )

            inputs.append(x0)
            inputs.append(x180)

        x = torch.stack(inputs).to(DEVICE)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(DEVICE == "cuda"),
        ):
            logits = model(x).squeeze(1)

        p = torch.sigmoid(
            logits.float()
        )

        p0 = p[0::2]
        p180 = p[1::2]

        crop_probs = (
            p0 + (1.0 - p180)
        ) / 2.0

        predictions[image_id] = (
            crop_probs.mean().item()
        )

    return predictions


def main():
    args = parse_args()

    torch.set_float32_matmul_precision("high")

    sample = pd.read_csv(
        SAMPLE_PATH
    )

    image_ids = sample["image_id"].tolist()

    print(f"Device: {DEVICE}")
    print(f"Test images: {len(image_ids)}")
    print(f"Models: {len(args.checkpoints)}")

    dataset = TestDataset(
        image_ids
    )

    wide_ids = []

    for image_id in tqdm(
        image_ids,
        desc="Finding wide images",
    ):
        with Image.open(
            TEST_DIR / f"{image_id}.png"
        ) as image:
            w, h = image.size

        if w / h >= WIDE_AR_THRESHOLD:
            wide_ids.append(image_id)

    print(
        f"Wide images: "
        f"{len(wide_ids)} / {len(image_ids)}"
    )

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=True,
    )

    all_probs = []
    all_wide_probs = []

    for checkpoint_path in args.checkpoints:
        model = load_model(
            checkpoint_path
        )

        probs = predict_model(
            model,
            loader,
        )

        all_probs.append(probs)

        wide_probs = predict_wide_model(
            model,
            wide_ids,
            dataset,
        )

        all_wide_probs.append(
            wide_probs
        )

        del model

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    probs = torch.stack(
        all_probs,
        dim=0,
    ).mean(dim=0)

    id_to_idx = {
        image_id: idx
        for idx, image_id in enumerate(image_ids)
    }

    for image_id in wide_ids:
        wide_p = sum(
            model_probs[image_id]
            for model_probs in all_wide_probs
        ) / len(all_wide_probs)

        idx = id_to_idx[image_id]

        probs[idx] = wide_p

    sample["p_180"] = probs.numpy()

    output_path = PROJECT_ROOT / args.output

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample.to_csv(
        output_path,
        index=False,
    )

    print(f"\nSaved: {output_path}")

    print(
        f"min={probs.min().item():.6f} | "
        f"max={probs.max().item():.6f} | "
        f"mean={probs.mean().item():.6f}"
    )

    print(sample.head())


if __name__ == "__main__":
    main()