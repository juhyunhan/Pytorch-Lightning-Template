import torch
from torchmetrics import Metric

class BaseAccuracyMetric(Metric):
    def __init__(self):
        super().__init__(dist_sync_on_step=False, compute_on_step=False)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds, target = self._input_format(preds, target)
        # assert preds.shape == target.shape
        
        _, predicted = torch.max(preds, 1)
    
        self.correct += (predicted == target).sum().item()
        self.total += target.numel()
        
    def compute(self):
        return {'accuracy' : self.correct.float() / self.total}