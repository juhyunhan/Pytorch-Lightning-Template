import torch
import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T
import os

class DataModule(pl.LightningDataModule):
    def __init__(self, dataset_name:str, data_config: dict = None, loader_config: dict = None):
        super().__init__()
        
        if dataset_name == 'CIFAR10':
            self.trainset = torchvision.datasets.CIFAR10(os.getcwd(), train=True, download=True, transform=T.ToTensor())
            self.valset = torchvision.datasets.CIFAR10(os.getcwd(), train=False, download=True, transform=T.ToTensor())
        else :
            self.dataset = torchvision.datasets.CIFAR10(os.getcwd(), download=True, transform=T.ToTensor())
        # self.trainset, self.valset = random_split(self.dataset, [55000, 5000])
        
        self.data_config = data_config
        self.loader_config = loader_config
        
        
    def get_dataset(self, split, loader=True, shuffle=False):
        if split == 'train':
            dataset = self.trainset
        else:
            dataset = self.valset
            
        loader_config = dict(self.loader_config)
        
        return DataLoader(dataset, shuffle=shuffle, **loader_config)
    
                    
    def train_dataloader(self, shuffle=True):
        dataloader = self.get_dataset(split='train', shuffle=shuffle)
        return dataloader
    
    def val_dataloader(self, shuffle=True):
        dataloader = self.get_dataset(split='val', shuffle=shuffle)
        return dataloader
    
    def eval_dataloader(self, shuffle=False):
        dataloader = self.get_dataset(split='test', shuffle=shuffle)
        return dataloader
