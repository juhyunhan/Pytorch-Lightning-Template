import torch
import pytorch_lightning as pl

class ModelModule(pl.LightningModule):
    def __init__(self, model, loss_func, metrics, optimizer_args, scheduler_args=None, cfg=None):
        super().__init__()
        
        self.save_hyperparameters(cfg,
                               ignore = ['backbone', 'head', 'loss_func', 'metrics', 'optimizer_args'])
        
        self.model = model
        self.loss_func = loss_func
        self.metrics = metrics
        
        self.optimizer_args = optimizer_args
        self.scheduler_args = scheduler_args
        
        self.param_groups = []
        
    def forward(self, inputs):
        
        output = self.model(inputs)
        return output
    
    def shared_step(self, batch, prefix='', on_step = False, return_output=True):
        img, label = batch
        pred = self(img)
        loss = self.loss_func(pred, label)
        
        self.metrics.update(pred, label)
        
        if self.trainer is not None:
            self.log(f'{prefix}/loss', loss.detach(), on_step = on_step, on_epoch = True, sync_dist=True)
            
        if return_output:
            return{'loss': loss, 'batch': batch, 'pred':pred}
        
        return {'loss':loss}
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train', True, False)
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val', False, False)
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, 'test', False, False)
    
    def test_epoch_end(self, outputs):
        self._log_epoch_metrics('test')
    
    def on_validation_start(self):
        self._log_epoch_metrics('train')
        self._enable_dataloader_shuffle(self.trainer.val_dataloaders)
        
    def validation_epoch_end(self, outputs):
        self._log_epoch_metrics('val')
        
    def _enable_dataloader_shuffle(self, dataloaders):
        for v in dataloaders:
            v.sampler.shuffle = True
            v.sampler.set_epoch(self.current_epoch)

    def _log_epoch_metrics(self, prefix: str):
        metrics = self.metrics.compute()
        
        for key, value in metrics.items():
            if isinstance(value, dict):
                for subkey, val in value.items():
                    self.log(f'{prefix}/metrics/{key}{subkey}', val, sync_dist=True)
                    
            else:
                self.log(f'{prefix}/metrics/{key}', value, sync_dist=True)
                
        self.metrics.reset()
                
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.optimizer_args.lr)
        return optimizer