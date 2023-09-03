import torch

from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from torchmetrics import MetricCollection
from pathlib import Path

from collections.abc import Callable
from typing import Tuple, Dict, Optional
from .models.model_module import ModelModule
from .data.data_module import DataModule

def setup_config(cfg : DictConfig, override: Optional[Callable] = None):
    OmegaConf.set_struct(cfg, False)
    
    if override is not None:
        override(cfg)
        
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, True)
    
    save_dir = Path(cfg.experiment.save_dir)
    save_dir.mkdir(parents=False, exist_ok=True)

def setup_network(cfg : DictConfig):
    network = instantiate(cfg.model)
    return network

def setup_model_module(cfg):
    model = setup_network(cfg)
    loss_func = instantiate(cfg.loss)
    metrics = instantiate(cfg.metrics)
    
    model_module = ModelModule(model, loss_func, metrics,
                               cfg.optimizer, cfg.scheduler,
                               cfg = cfg)
    return model_module

def setup_data_module(cfg : DictConfig):
    return DataModule(cfg.data.dataset, cfg.data, cfg.loader)


def setup_experiment(cfg : DictConfig):
    model_module = setup_model_module(cfg)
    data_module = setup_data_module(cfg)
    
    return model_module, data_module
    
    