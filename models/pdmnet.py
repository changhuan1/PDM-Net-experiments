from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .attention import CBAMBlock, SEBlock
from .backbones import FeatureBackbone


class FERCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        features = self.features(x)
        embedding = self.pool(features).flatten(1)
        logits = self.classifier(embedding)
        return {"logits": logits, "logits_g": logits, "features": features, "embedding": embedding}


class BaselineClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet18",
        pretrained: bool = True,
        attention: str | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = FeatureBackbone(backbone, pretrained)
        channels = self.backbone.feature_dim
        if attention == "se":
            self.attention = SEBlock(channels)
        elif attention == "cbam":
            self.attention = CBAMBlock(channels)
        elif attention is None:
            self.attention = nn.Identity()
        else:
            raise ValueError(f"Unsupported attention: {attention}")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        features = self.attention(self.backbone(x))
        z = self.dropout(self.pool(features).flatten(1))
        logits = self.classifier(z)
        return {"logits": logits, "logits_g": logits, "features": features, "embedding": z}


class PrototypeClassifierBaseline(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet18",
        pretrained: bool = True,
        temperature: float = 0.1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = FeatureBackbone(backbone, pretrained)
        channels = self.backbone.feature_dim
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.prototypes = nn.Parameter(torch.randn(num_classes, channels) * 0.02)
        self.temperature = temperature

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        z = self.dropout(self.pool(features).flatten(1))
        logits = F.normalize(z, dim=1) @ F.normalize(self.prototypes, dim=1).t()
        logits = logits / self.temperature
        return {"logits": logits, "logits_p": logits, "features": features, "embedding": z}


class PDMNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet18",
        pretrained: bool = True,
        temperature: float = 0.1,
        fusion_alpha: float = 1.0,
        mask_scale: float = 5.0,
        use_gt_mask_train: bool = True,
        use_prototype_mask: bool = True,
        use_prototype_classifier: bool = True,
        class_agnostic_attention: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.fusion_alpha = fusion_alpha
        self.mask_scale = mask_scale
        self.use_gt_mask_train = use_gt_mask_train
        self.use_prototype_mask = use_prototype_mask
        self.use_prototype_classifier = use_prototype_classifier
        self.class_agnostic_attention = class_agnostic_attention

        self.backbone = FeatureBackbone(backbone, pretrained)
        channels = self.backbone.feature_dim
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.global_classifier = nn.Linear(channels, num_classes)
        self.prototypes = nn.Parameter(torch.randn(num_classes, channels) * 0.02)
        self.class_agnostic_mask = nn.Sequential(nn.Conv2d(channels, 1, kernel_size=1), nn.Sigmoid())

    def compute_masks(self, features: torch.Tensor) -> torch.Tensor:
        b, _, h, w = features.shape
        if not self.use_prototype_mask:
            return features.new_ones((b, self.num_classes, h, w))

        if self.class_agnostic_attention:
            mask = self.class_agnostic_mask(features)
            return mask.repeat(1, self.num_classes, 1, 1)

        local = F.normalize(features, dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)
        similarity = torch.einsum("bchw,kc->bkhw", local, prototypes)
        return torch.sigmoid(self.mask_scale * similarity)

    def _prototype_logits_from_shared_feature(self, z_m: torch.Tensor) -> torch.Tensor:
        prototypes = F.normalize(self.prototypes, dim=1)
        logits = F.normalize(z_m, dim=1) @ prototypes.t()
        return logits / self.temperature

    def _prototype_logits_from_class_masks(self, features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        masked = features.unsqueeze(1) * masks.unsqueeze(2)
        z_m = masked.mean(dim=(3, 4))
        z_m = F.normalize(z_m, dim=2)
        prototypes = F.normalize(self.prototypes, dim=1).unsqueeze(0)
        logits = (z_m * prototypes).sum(dim=2)
        return logits / self.temperature

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        z_g = self.dropout(self.pool(features).flatten(1))
        logits_g = self.global_classifier(z_g)
        masks = self.compute_masks(features)

        selected_mask = None
        if self.training and labels is not None and self.use_gt_mask_train:
            selected_mask = masks[torch.arange(x.shape[0], device=x.device), labels]
            z_m = self.dropout((features * selected_mask.unsqueeze(1)).mean(dim=(2, 3)))
            logits_p = self._prototype_logits_from_shared_feature(z_m)
        else:
            logits_p = self._prototype_logits_from_class_masks(features, masks)
            z_m = z_g

        if not self.use_prototype_classifier:
            logits_p = torch.zeros_like(logits_g)

        logits = logits_g + self.fusion_alpha * logits_p
        return {
            "logits": logits,
            "logits_g": logits_g,
            "logits_p": logits_p,
            "masks": masks,
            "selected_mask": selected_mask,
            "features": features,
            "embedding": z_m,
        }


def create_model(model_name: str, num_classes: int, cfg: dict[str, Any]) -> nn.Module:
    model_name = model_name.lower()
    backbone = cfg.get("backbone", "resnet18")
    pretrained = bool(cfg.get("pretrained", True))
    temperature = float(cfg.get("temperature", 0.1))
    dropout = float(cfg.get("dropout", 0.0))

    if model_name in {"fer_cnn", "fercnn"}:
        return FERCNN(num_classes=num_classes, dropout=dropout if dropout > 0 else 0.4)
    if model_name == "resnet18":
        return BaselineClassifier(num_classes, backbone="resnet18", pretrained=pretrained, dropout=dropout)
    if model_name == "resnet50":
        return BaselineClassifier(num_classes, backbone="resnet50", pretrained=pretrained, dropout=dropout)
    if model_name in {"mobilenetv2", "mobilenet_v2"}:
        return BaselineClassifier(num_classes, backbone="mobilenetv2", pretrained=pretrained, dropout=dropout)
    if model_name == "resnet18_se":
        return BaselineClassifier(num_classes, backbone="resnet18", pretrained=pretrained, attention="se", dropout=dropout)
    if model_name == "resnet18_cbam":
        return BaselineClassifier(num_classes, backbone="resnet18", pretrained=pretrained, attention="cbam", dropout=dropout)
    if model_name == "resnet50_cbam":
        return BaselineClassifier(num_classes, backbone="resnet50", pretrained=pretrained, attention="cbam", dropout=dropout)
    if model_name == "prototype":
        return PrototypeClassifierBaseline(
            num_classes,
            backbone=backbone,
            pretrained=pretrained,
            temperature=temperature,
            dropout=dropout,
        )
    if model_name == "pdmnet":
        return PDMNet(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=pretrained,
            temperature=temperature,
            fusion_alpha=float(cfg.get("fusion_alpha", 1.0)),
            mask_scale=float(cfg.get("mask_scale", 5.0)),
            use_gt_mask_train=bool(cfg.get("use_gt_mask_train", True)),
            use_prototype_mask=bool(cfg.get("use_prototype_mask", True)),
            use_prototype_classifier=bool(cfg.get("use_prototype_classifier", True)),
            class_agnostic_attention=bool(cfg.get("class_agnostic_attention", False)),
            dropout=dropout,
        )

    raise ValueError(f"Unsupported model name: {model_name}")
