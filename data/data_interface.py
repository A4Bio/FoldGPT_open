import inspect
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import torch.distributed as dist
import random
import json
from .datasets import PDBVQDataset

def custom_collate_fn(batch):
    batch = [one for one in batch if one is not None]
    if len(batch)==0:
        return None
    
    ret = {}
    for key in batch[0].keys():
        if type(batch[0][key])==torch.Tensor:
            ret[key] = torch.stack([one[key] for one in batch], dim=0)
        elif type(batch[0][key])== str:
            ret[key] = [one[key] for one in batch]

    return ret


class DInterface(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.kwargs = kwargs
        self.batch_size = kwargs['batch_size']
        print("batch_size", self.batch_size)
        self.load_data_module()
        self.trainset, self.valset, self.testset = None, None, None

    def setup(self, stage=None, mask_location=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            if self.trainset is None:
                self.trainset = self.instancialize(split = 'train', scaffold_prob=self.hparams.scaffold_prob, inpaint_prob=self.hparams.inpaint_prob)
            
            if self.valset is None:
                self.valset = self.instancialize(split='test', scaffold_prob=self.hparams.scaffold_prob, inpaint_prob=self.hparams.inpaint_prob)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            if self.testset is None:
                self.testset = self.instancialize(split='test')
        
        if stage == 'scaffold' or stage is None:
            if self.testset is None:
                self.testset = self.instancialize(split='scaffold')
        

    def train_dataloader(self, db=None, preprocess=False):
        return self.instancialize_module(DataLoader, dataset=self.trainset, shuffle=True, prefetch_factor=4, pin_memory = True, collate_fn=custom_collate_fn) # , prefetch_factor=3
    

    def val_dataloader(self, db=None, preprocess=False):
        return self.instancialize_module(DataLoader, dataset=self.valset, shuffle=False, collate_fn=custom_collate_fn) # , prefetch_factor=3
    

    def test_dataloader(self, db=None, preprocess=False, mask_location=None):
        return self.instancialize_module(DataLoader, dataset=self.testset, shuffle=False, collate_fn=custom_collate_fn)


    def load_data_module(self):
        self.data_module = PDBVQDataset

       

    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        class_args = list(inspect.signature(self.data_module.__init__).parameters)[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)
    
    def instancialize_module(self, module, **other_args):
        class_args =  list(inspect.signature(module.__init__).parameters)[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return module(**args1)




    