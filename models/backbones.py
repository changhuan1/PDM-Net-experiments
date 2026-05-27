from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision import models


@dataclass(frozen=True)
class BackboneSpec:
    module: nn.Module
    feature_dim: int


def _resnet18(pretrained: bool) -> BackboneSpec:
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    except AttributeError:
        model = models.resnet18(pretrained=pretrained)
    body = nn.Sequential(*list(model.children())[:-2])
    return BackboneSpec(module=body, feature_dim=512)


def _resnet50(pretrained: bool) -> BackboneSpec:
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
    except AttributeError:
        model = models.resnet50(pretrained=pretrained)
    body = nn.Sequential(*list(model.children())[:-2])
    return BackboneSpec(module=body, feature_dim=2048)


def _mobilenetv2(pretrained: bool) -> BackboneSpec:
    try:
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
    except AttributeError:
        model = models.mobilenet_v2(pretrained=pretrained)
    return BackboneSpec(module=model.features, feature_dim=1280)


class FeatureBackbone(nn.Module):
    def __init__(self, name: str = "resnet18", pretrained: bool = True) -> None:
        super().__init__()
        name = name.lower()
        if name == "resnet18":
            spec = _resnet18(pretrained)
        elif name == "resnet50":
            spec = _resnet50(pretrained)
        elif name in {"mobilenetv2", "mobilenet_v2"}:
            spec = _mobilenetv2(pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {name}")
        self.body = spec.module
        self.feature_dim = spec.feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)
