import timm
import torch.nn as nn

model = timm.create_model('vit_small_patch16_224.augreg_in1k', pretrained=True, num_classes=10)

class PretrainedVisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = model
    
    def forward(self, x):
        x = self.backbone(x)
        return x
        
