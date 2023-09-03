import torch
import torchvision.models as models
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self, backbone, head):
        super(TestModel, self).__init__()
        self.backbone = backbone
        self.head = head
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
    
        return x