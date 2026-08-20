import torch.nn as nn
from torchvision import models


def initialize_model(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = "DEFAULT" if pretrained else None
    if backbone.startswith("resnet"):
        model = getattr(models, backbone)(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone in ("vit_b_16", "vit_b_32"):
        model = getattr(models, backbone)(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"unsupported backbone: {backbone}")
    return model
