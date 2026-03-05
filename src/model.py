import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes=4, freeze_backbone=True):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_featues = model.fc.in_features
    model.fc = nn.Linear(in_featues, num_classes)

    return model