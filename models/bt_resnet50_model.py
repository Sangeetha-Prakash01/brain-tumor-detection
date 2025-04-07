# models/bt_resnet50_model.py

import torch.nn as nn
from torchvision import models

def build_model():
    model = models.resnet50(pretrained=True)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)  # Assuming 2 classes: Tumor / No Tumor
    )

    return model
