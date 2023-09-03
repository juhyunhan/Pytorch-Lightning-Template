import torch
import torchvision.models as models
import torch.nn as nn

class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, 10)
        
    def forward(self, x):
        B, _, _, _ = x.shape
        x = self.avgpool(x)
        x = x.view(B, -1)
        x = self.linear(x)
        
        return x