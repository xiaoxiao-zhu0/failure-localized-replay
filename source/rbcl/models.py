"""Model factory used by RBCL experiment scripts."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet152

from avalanche.models import SimpleCNN, SimpleMLP, SlimResNet18


def build_model(model_name: str, *, num_classes: int, benchmark_name: str):
    """Create a backbone while keeping baseline/model capacity comparable."""
    key = model_name.lower()
    benchmark_key = benchmark_name.lower()

    if key == "auto":
        if "mnist" in benchmark_key:
            key = "simple_mlp"
        else:
            key = "simple_cnn"

    if key == "simple_mlp":
        input_size = 28 * 28 if "mnist" in benchmark_key else 3 * 32 * 32
        return SimpleMLP(num_classes=num_classes, input_size=input_size)

    if key == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)

    if key == "frozen_resnet152":
        checkpoint = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "resnet152-394f9c45.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "frozen_resnet152 requires the local ImageNet checkpoint at "
                f"{checkpoint}; no download is attempted."
            )
        model = resnet152(weights=None)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("fc."))
        return model

    if key == "slim_resnet18":
        return SlimResNet18(nclasses=num_classes)

    raise ValueError(
        "Unknown model. Supported: auto, simple_mlp, simple_cnn, frozen_resnet152, slim_resnet18."
    )
