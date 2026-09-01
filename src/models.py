import torch
import timm

from torchvision.models import resnet18, ResNet18_Weights


def load_resnet18(device):
    weights = ResNet18_Weights.DEFAULT

    model = resnet18(weights=weights)
    model.eval()
    model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    transform = weights.transforms()

    return model, transform


def load_deit_tiny(device):
    model = timm.create_model(
        "deit_tiny_patch16_224.fb_in1k",
        pretrained=True
    )

    model.eval()
    model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    config = timm.data.resolve_model_data_config(model)

    transform = timm.data.create_transform(
        **config,
        is_training=False
    )

    return model, transform