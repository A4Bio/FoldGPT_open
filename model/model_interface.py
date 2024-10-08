import inspect
import torch
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
import torch.nn as nn
import os
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from omegaconf import OmegaConf
from torcheval.metrics.text import Perplexity
from sampling import load_VQAE_model
vq_model = load_VQAE_model()

MAX_CAHIN = 1000
class MInterface(pl.LightningModule):
    def __init__(self,  model_name=None, steps_per_epoch=None, loss=None, lr=None, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()
        # os.makedirs(os.path.join(self.hparams.res_dir, self.hparams.ex_name), exist_ok=True)

        self.loss_ce = CrossEntropyLoss(reduction='none')
        self.loss_bce = BCEWithLogitsLoss(reduction='none')

    
    def compute_loss(self, batch, output):
        logits = output.logits
        label_ids = batch['input_ids']
        islabel = batch['label_mask'] == 1

        
        label_ids = label_ids[islabel]
        logits = logits[:, 0:-1][islabel[:,1:]]
    
        
        loss = self.loss_ce(logits, label_ids)
        vqid_acc = ((logits.argmax(dim=-1))==label_ids).float()
        
        vqid_acc = vqid_acc.mean()
        
        perp_metric = Perplexity()
        perp_metric.update(logits[None].cpu(), label_ids[None].cpu())
        perp = perp_metric.compute()


        loss_vqid = loss.mean()
        

        return {'loss': loss_vqid,
                'acc': vqid_acc,
                'perp':perp}

    def forward(self, batch):
        is_condition = batch['is_condition']==1
        output = self.model(batch['input_ids'], position_ids=batch['pos_ids'], attention_mask=batch['data_mask'], is_condition=is_condition)
        loss_dict = self.compute_loss(batch, output)
        return loss_dict
    
    def training_step(self, batch, batch_idx):
        results = self(batch)
        for key, val in results.items():
            self.log("train_"+key, val, on_step=True, on_epoch=True, prog_bar=True)

        return results['loss']

    def validation_step(self, batch, batch_idx, test=False):
        self.eval()
        with torch.no_grad(): 
            results = self(batch)
        
        log_dict = {'val_'+key: val for key, val in results.items()}
        self.log_dict(log_dict, on_epoch=True)
        return self.log_dict

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, test=True)

    def on_validation_epoch_end(self):
        self.print('')

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        
        optimizer = torch.optim.AdamW(filter(lambda p:p.requires_grad, self.parameters()), lr=self.hparams.lr, weight_decay=weight_decay)


        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'onecycle':
                scheduler = lrs.OneCycleLR(optimizer, max_lr=self.hparams.lr, total_steps=self.hparams.epoch * self.hparams.steps_per_epoch, three_phase=True)
            
            if self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer, T_max=self.hparams.lr_decay_steps)

            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        
    def lr_scheduler_step(self, *args, **kwargs):
        scheduler = self.lr_schedulers()
        scheduler.step()
    
    def configure_devices(self):
        self.device = torch.device(self.hparams.device)

    def configure_loss(self):
        self.NLLLoss = nn.NLLLoss(reduction='none')
        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')

    def find_all_linear_modules(self, model, freeze_vision_tower):
        r"""
        Finds all available modules to apply lora or galore.
        """
        forbidden_modules = {"lm_head"}

        if model.config.model_type == "chatglm":
            forbidden_modules.add("output_layer")
        elif model.config.model_type == "internlm2":
            forbidden_modules.add("output")
        elif model.config.model_type in ["llava", "paligemma"]:
            forbidden_modules.add("multi_modal_projector")

        if freeze_vision_tower:
            forbidden_modules.add("vision_tower")

        module_names = set()
        for name, module in model.named_modules():
            if any(forbidden_module in name for forbidden_module in forbidden_modules):
                continue

            if "Linear" in module.__class__.__name__ and "Embedding" not in module.__class__.__name__:
                module_names.add(name.split(".")[-1])

        return list(module_names)


    def load_model(self):
        from model.models import LlamaForCausalLM
        from transformers.models.llama.configuration_llama import LlamaConfig
        
        model_config  = OmegaConf.load('./model/config.yaml')
        model_config_dict = OmegaConf.to_container(model_config, resolve=True)

        model_config = LlamaConfig(**model_config_dict)
        self.model = LlamaForCausalLM(model_config, binary_code=self.hparams.binary_code)
        # self.model = self._setup_lora_tuning(self.model,True)
        print()

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = list(inspect.signature(Model.__init__).parameters)[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)
    
    
