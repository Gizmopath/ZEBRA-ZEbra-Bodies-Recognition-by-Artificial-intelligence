import torch.nn as nn
from torchvision import models

def get_efficientnet_b2(num_classes):
    model = models.efficientnet_b2(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
