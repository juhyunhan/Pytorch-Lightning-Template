import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

class TestLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, pred, gt):
        loss = self.criterion(pred, gt)
        
        return loss