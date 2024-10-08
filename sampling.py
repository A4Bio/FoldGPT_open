import os
import torch
from omegaconf import OmegaConf
from data.datasets import PDBVQDataset
from src.chroma.data import Protein

def load_VQAE_model(FT4Path = '/huyuqi/xmyu/FoldToken4_share/foldtoken'):
    import sys; sys.path.append(FT4Path)
    from model_interface import MInterface
    config = f'{FT4Path}/model_zoom/FT4/config.yaml'
    checkpoint = f'{FT4Path}/model_zoom/FT4/ckpt.pth'

    config = OmegaConf.load(config)
    config = OmegaConf.to_container(config, resolve=True)
    model = MInterface(**config)
    checkpoint = torch.load(checkpoint, map_location=torch.device('cuda'))
    for key in list(checkpoint.keys()):
        if '_forward_module.' in key:
            checkpoint[key.replace('_forward_module.', '')] = checkpoint[key]
            del checkpoint[key]
    model.load_state_dict(checkpoint)
    model = model.to('cuda')
    model = model.eval()
    return model


def load_VQGPT_model(checkpoint='/huyuqi/xmyu/FoldMLM/FoldGPT/results/FoldGPT_AR/checkpoints/last.ckpt', config=None):
    from model.model_interface import MInterface
    config = OmegaConf.load(config)
    config = OmegaConf.to_container(config, resolve=True)
    model = MInterface(**config)
    checkpoint = torch.load(checkpoint, map_location=torch.device('cuda'))
    checkpoint = checkpoint['state_dict']
    for key in list(checkpoint.keys()):
        if '_forward_module.' in key:
            checkpoint[key.replace('_forward_module.', '')] = checkpoint[key]
            del checkpoint[key]
    model.load_state_dict(checkpoint)  # checkpoint['state_dict']
    model = model.to('cuda')
    model = model.eval()
    return model

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', default='results/conditional', type=str)
    parser.add_argument('--config', default='model_zoom/conditional/config.yaml', type=str)
    parser.add_argument('--checkpoint', default='model_zoom/conditional/params.ckpt', type=str)
    parser.add_argument('--temperature', default=0.5, type=float)
    parser.add_argument('--length', default=150, type=int)
    parser.add_argument('--nums', default=20, type=int)
    parser.add_argument('--mask_mode', default='conditional', type=str)
    parser.add_argument('--template', default='./8vrwB.pdb', type=str)
    parser.add_argument('--mask', default='10-30', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    vq_model = load_VQAE_model()
    model = load_VQGPT_model(args.checkpoint, args.config)
    level = 8

    temperature = args.temperature
    length = args.length
    nums = args.nums
    # for file_idx in tqdm(range(args.nums)):
    # ====================== encoder ========================
    # protein = Protein(file_name, device='cuda') 
    # L = torch.randint(30, 500, (1,)).item()
    
    if args.mask_mode == 'conditional':
        protein = Protein(args.template, device='cuda') 
        h_V_quat, vq_code,  batch_id, chain_encoding = vq_model.encode_protein(protein)
        mask_indices = []
        for segment in args.mask.split(','): 
            if '-' in segment:
                start, end = map(int, segment.split('-'))
                mask_indices.extend(list(range(start, end)))
            else:
                mask_indices.append(int(segment))
    else:
        vq_code = torch.ones(length, dtype=torch.long, device='cuda')
        chain_encoding = torch.ones_like(vq_code)
    
    with torch.no_grad():
        # # # ====================== GPT ========================
        num_res = vq_code.shape[0]
        if args.mask_mode == 'unconditional':
            vq_ids, label_mask, pos_ids, is_condition = PDBVQDataset.mask_data(vq_code, 'unconditional')
        else:
            vq_ids, label_mask, pos_ids, is_condition = PDBVQDataset.mask_data(vq_code, 'conditional', mask_indices=mask_indices)
        sep_idx = (vq_ids == PDBVQDataset.tokenizer.pad_token_id).float().argmax()

        vq_ids_in = [vq_ids[is_condition == 1].tolist() for _ in range(nums)]
        pos_ids_in = [pos_ids[is_condition == 1].tolist() for _ in range(nums)]
        is_condition_flags = [is_condition[is_condition == 1].tolist()  for _ in range(nums)]
        
        vq_ids_pred_list = model.model.generate(vq_ids_in, pos_ids_in, max_gen_len=label_mask.sum().long(), is_condition_flags=is_condition_flags, temperature=temperature)

        for idx, vq_ids_pred in enumerate(vq_ids_pred_list):
            # ====================== decoder ========================
            h_V = vq_model.model.vq.embed_id(vq_ids_pred, level=level)
            pred_protein = vq_model.model.decoding(h_V, chain_encoding + 1, batch_id=None, returnX=False)

            if args.mask_mode == 'unconditional':
                os.makedirs(f'{args.save_path}/temp{temperature}/pred_pdb_gpt_{length}/', exist_ok=True)
                pred_protein.to(f'{args.save_path}/temp{temperature}/pred_pdb_gpt_{length}/gen_{idx}.pdb')
            else:
                name = args.template.split('.')[-2]
                os.makedirs(f'{args.save_path}/temp{temperature}_{args.mask}', exist_ok=True)
                pred_protein.to(f'{args.save_path}/temp{temperature}_{args.mask}/{name}_gpt_{idx}.pdb', mask_indices=mask_indices)
