import datetime
import os
import sys
sys.path.append(os.getcwd())
# os.environ["WANDB_API_KEY"] = "3afd3131afecd5d6e3eb1a05274f3a67bdbb2b1f"
os.environ["WANDB_API_KEY"] = "local-1d956c1346fedff67f8b1df03bd5739b81267171"
import warnings

warnings.filterwarnings("ignore")

import math
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as plog
from model.model_interface import MInterface
from data.data_interface import DInterface

seed_everything(0)




def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='FoldGPT')
    parser.add_argument('--data_path', default='/huyuqi/xmyu/FoldToken2/foldtoken2_data/pdb_vqids_ft4/pdb_256.jsonl', type=str)
    parser.add_argument('--mask_mode', default='unconditional', type=str)
    parser.add_argument('--pad', default=512, type=int)
    parser.add_argument('--epoch', default=1000, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='batch size')
    parser.add_argument('--lr', default=0.0001, type=float, metavar='N', help='learning rate')
    parser.add_argument('--lr_scheduler', default="cosine", choices=['onecycle', 'cosine', 'step'], type=str, help='learning rate scheduler')
    parser.add_argument('--num_workers', default=1, type=int, metavar='N', help='number of data loading workers (default: 8)')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--lr_decay_steps', default=1000, type=int)
    parser.add_argument('--binary_code', default=1, type=int)


    parser.add_argument('--offline', default=1, type=int)
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='debug', type=str)

    parser.add_argument('--scaffold_prob', default=0.3, type=float)
    parser.add_argument('--inpaint_prob', default=0.3, type=float)
    args = parser.parse_args()
    return args

def load_callbacks(args):
    callbacks = []
    logdir = str(os.path.join(args.res_dir, args.ex_name))
    
    ckptdir = os.path.join(logdir, "checkpoints")

    monitor_metric = 'val_perp'
    filename = 'best-{epoch:02d}-{val_perp:.3f}'
    
    args.monitor_metric = monitor_metric
        
    callbacks.append(plc.ModelCheckpoint(
        monitor= monitor_metric,
        filename=filename,
        save_top_k=10,
        mode='min',
        save_last=True,
        dirpath = ckptdir,
        verbose = True,
        every_n_epochs = args.check_val_every_n_epoch,
    ))

    
    now = datetime.datetime.now().strftime("%m-%dT%H-%M-%S")
    cfgdir = os.path.join(logdir, "configs")
    
    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval=None))
        
    return callbacks




if __name__ == "__main__":
    args = parse_args()
    pl.seed_everything(args.seed)

    data_module = DInterface(**vars(args))

    data_module.setup('fit')
    train_loader = data_module.train_dataloader()
    train_dataset = data_module.trainset
    gpu_count = torch.cuda.device_count()
    steps_per_epoch = math.ceil(len(data_module.trainset)/args.batch_size/gpu_count)
    callbacks = load_callbacks(args)
    
    model = MInterface(steps_per_epoch = steps_per_epoch, **vars(args))
    

    trainer_config = {
        'devices': -1,  # Use all available GPUs
        'precision': '32',  # Use 32-bit floating point precision
        'max_epochs': args.epoch,  # Maximum number of epochs to train for
        'num_nodes': 1,  # Number of nodes to use for distributed training
        "strategy": 'ddp',
        "accumulate_grad_batches": 1,
        'accelerator': 'cuda',  
        'callbacks': load_callbacks(args),
        'logger': [
                    plog.WandbLogger(
                    project = 'FoldGPT',
                    name=args.ex_name,
                    save_dir=str(os.path.join(args.res_dir, args.ex_name)),
                    offline = args.offline,
                    id = args.ex_name.replace('/', '-',5),
                    entity = "gzy"),
                   plog.CSVLogger(args.res_dir, name=args.ex_name)],
         'gradient_clip_val':0.5
    }

    trainer = Trainer(**trainer_config)
    trainer.fit(model, data_module, ckpt_path='/storage/huyuqi/gzy/FoldGPT/results/FoldGPT_AR/checkpoints/last_epoch233.ckpt')
    

