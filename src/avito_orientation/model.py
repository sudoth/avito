import torch.nn as nn
import timm

from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_V2_S_Weights,
    convnext_tiny,
    efficientnet_v2_s,
)


def create_model(name="efficientnet_v2_s"):
    if name == "efficientnet_v2_s":
        model = efficientnet_v2_s(
            weights=EfficientNet_V2_S_Weights.DEFAULT,
        )

        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)

        return model

    if name == "convnext_tiny":
        model = convnext_tiny(
            weights=ConvNeXt_Tiny_Weights.DEFAULT,
        )

        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, 1)

        return model

    if name == "convnextv2_tiny":
        model = timm.create_model(
            "convnextv2_tiny.fcmae_ft_in22k_in1k",
            pretrained=True,
            num_classes=1,
        )

        return model

    if name == "mobilenetv4":
        model = timm.create_model(
            "mobilenetv4_conv_medium.e500_r256_in1k",
            pretrained=True,
            num_classes=1,
        )

        return model

    if name == "convnextv2_base":
        model = timm.create_model(
            "convnextv2_base.fcmae_ft_in22k_in1k",
            pretrained=True,
            num_classes=1,
        )
        return model

    if name == "convnext_dinov3_base":
        model = timm.create_model(
            "convnext_base.dinov3_lvd1689m",
            pretrained=True,
            num_classes=1,
        )
        return model

    if name == "convnext_dinov3_small":
        model = timm.create_model(
            "convnext_small.dinov3_lvd1689m",
            pretrained=True,
            num_classes=1,
        )
        return model