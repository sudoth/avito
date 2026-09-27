import torch.nn as nn
from torchvision.models import (
    EfficientNet_V2_S_Weights,
    efficientnet_v2_s,
)


def create_model():
    weights = EfficientNet_V2_S_Weights.DEFAULT

    model = efficientnet_v2_s(weights=weights)

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        1,
    )

    return model