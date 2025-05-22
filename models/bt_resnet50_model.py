import torch.nn as nn
from torchvision import models

def get_resnet50_model(num_classes=4, pretrained=True, freeze=True):
    # Load pretrained ResNet50 weights if pretrained=True
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully connected layer to match your number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


