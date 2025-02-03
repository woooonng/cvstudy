import torch
import os
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

resnet50 = models.resnet50(weights=None)

_logger = logging.getLogger('train')

class ResNet50Modified(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.backbone = nn.Sequential(
            resnet50.layer1,
            resnet50.layer2,
            resnet50.layer3,
            resnet50.layer4,
        )
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(resnet50.fc.in_features, 10)
    
    def forward(self, x):
        x = self.layer0(x)
        x = self.backbone(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x