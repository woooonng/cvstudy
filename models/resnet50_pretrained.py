import torch
import logging
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

resnet50_pretrained = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

_logger = logging.getLogger('train')

class PretrainedResNet50Modified(nn.Module):
    def __init__(self, freeze=True, pyramid=False):
        super().__init__()
        self.pyramid = pyramid

        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = resnet50_pretrained.layer1
        self.layer2 = resnet50_pretrained.layer2
        self.layer3 = resnet50_pretrained.layer3
        self.layer4 = resnet50_pretrained.layer4

        if freeze:
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for param in layer.parameters():
                    param.requires_grad = False
            _logger.info("Freezing the backbone is done")

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        # fc-layer
        if pyramid:
            in_features = 256 + 512 + 1024 + 2048
            hidden_dim = int(in_features / 2)
            self.fc = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(hidden_dim, 10)
            )
        else:
            self.fc = nn.Linear(resnet50_pretrained.fc.in_features, 10)
    
    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        if self.pyramid:
            layer1_pooled = self.pooling(layer1)
            layer2_pooled = self.pooling(layer2)
            layer3_pooled = self.pooling(layer3)
            layer4_pooled = self.pooling(layer4)
            y = torch.cat([layer1_pooled, layer2_pooled, layer3_pooled, layer4_pooled], dim=1)
            y = torch.flatten(y, 1)
        
        else:
            y = self.pooling(layer4)
            y = torch.flatten(y, 1)

        y = self.fc(y)
        return y