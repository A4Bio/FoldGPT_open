import torch
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import json
from data.pdb_tokenizer import PDBTokenizer
import random
import torch.nn as nn

class MultiFoldLang(nn.Module):
    def __init__(self, map_path=f'/huyuqi/xmyu/FoldGPT/FoldToken4/model_zoom'):
        super().__init__()
        self.register_buffer('map12to5', torch.load(f'{map_path}/map12to5.pt').cpu())
        self.register_buffer('map12to6', torch.load(f'{map_path}/map12to6.pt').cpu())
        self.register_buffer('map12to7', torch.load(f'{map_path}/map12to7.pt').cpu())
        self.register_buffer('map12to8', torch.load(f'{map_path}/map12to8.pt').cpu())
        self.register_buffer('map12to9', torch.load(f'{map_path}/map12to9.pt').cpu())
        self.register_buffer('map12to10', torch.load(f'{map_path}/map12to10.pt').cpu())
        self.register_buffer('map12to11', torch.load(f'{map_path}/map12to11.pt').cpu())
        
        self.register_buffer('map5to12', torch.load(f'{map_path}/map5to12.pt').cpu())
        self.register_buffer('map6to12', torch.load(f'{map_path}/map6to12.pt').cpu())
        self.register_buffer('map7to12', torch.load(f'{map_path}/map7to12.pt').cpu())
        self.register_buffer('map8to12', torch.load(f'{map_path}/map8to12.pt').cpu())
        self.register_buffer('map9to12', torch.load(f'{map_path}/map9to12.pt').cpu())
        self.register_buffer('map10to12', torch.load(f'{map_path}/map10to12.pt').cpu())
        self.register_buffer('map11to12', torch.load(f'{map_path}/map11to12.pt').cpu())
        

def generate_pos_sequence(max_length, known_indices):
    # 初始化结果列表，并将已知索引添加到结果列表中
    result = []
    
    # 使用集合快速查找已知索引
    known_set = set(known_indices)

    if len(known_set)==0:
        return list(range(max_length))
    
    while len(result)+len(known_indices) < max_length:
        # 找到所有可能的候选索引，它们与已知索引或之前预测的未知索引的差为1
        candidates = set()
        for idx in known_set:
            if idx - 1 >= 0 and (idx - 1) not in known_set:
                candidates.add(idx - 1)
            if idx + 1 < max_length and (idx + 1) not in known_set:
                candidates.add(idx + 1)
        
        # 如果有候选索引，从中随机选择一个，并将其添加到结果和已知索引中
        if candidates:
            chosen_idx = random.choice(list(candidates))
            result.append(chosen_idx)
            known_set.add(chosen_idx)
        else:
            break
    
    return result

def generate_mask(length, mode='scaffolding'):
    # Motif的长度在5到30之间
    motif_length = random.randint(5, int(length*0.5))
    # motif_length = random.randint(5, 20)
    
    # 随机选择motif的起始位置
    motif_start = random.randint(0, length - motif_length) # 第一个位置是BOS token，不掩码
    
    # 获取motif的结束位置
    motif_end = motif_start + motif_length
    
    if mode == 'scaffolding':
        # scaffolding模式，掩码掉motif以外的所有位置
        mask = [i for i in range(length) if i < motif_start or i >= motif_end]
    elif mode == 'inpainting':
        # inpainting模式，掩码掉motif
        mask = [i for i in range(motif_start, motif_end)]
    elif mode == 'MLM':
        mask = [i for i in range(length) if random.random() < 0.15]
    elif mode == 'GPT':
        mask = [i for i in range(0, length)]
    else:
        raise ValueError("Mode should be either 'scaffolding' or 'inpainting'")
    
    unmask = list(set(torch.arange(length).tolist())-set(mask))
    return sorted(unmask), sorted(mask)


def Segment(L, n, k):
    """
    将长度为L的序列划分为n个连续片段，每个片段长度至少为5，并从这些片段中随机选择k个片段。

    参数:
    L (int): 序列的总长度。
    n (int): 片段的数量。
    k (int): 需要选择的片段数量。

    返回:
    List[List[int]]: 被选中片段的索引列表。每个片段是一个整数列表，表示片段中的所有索引位置。

    异常:
    ValueError: 如果L不足以划分n个长度至少为5的片段，或选择的片段数量k超过总片段数n，将引发异常。
    """
    # 检查序列长度是否足够
    if n * 5 > L:
        raise ValueError("序列长度L太短，无法满足每个片段长度不小于5的要求")
    
    start = 0
    segments = []
    
    # 划分n个片段
    for i in range(n):
        if i == n - 1:
            # 最后一个片段包含剩余的所有元素
            segments.append(range(start, L))
        else:
            # 随机确定每个片段的结束位置
            end = random.randint(start + 5, L - (n - i - 1) * 5)
            segments.append(range(start, end))
            start = end
    
    # 从n个片段中随机选择k个片段，保证选择的片段互不重复
    if k > n:
        raise ValueError("选择的片段数量k不能超过总片段数n")
    
    selected_segments = random.sample(segments, k)
    
    unmask = [list(seg) for seg in selected_segments]
    unmask = [item for sublist in unmask for item in sublist]
    # mask = list(set(range(L))-set(unmask))
    
    # 输出每个被选中片段的索引列表
    return sorted(unmask)#, mask


class PDBVQDataset(Dataset):
    tokenizer = PDBTokenizer()
    def __init__(
        self,
        split='train',
        pad: int = 512,
        min_length: int = 40, 
        data_path: str = '/huyuqi/xmyu/FoldToken2/foldtoken2_data/pdb_vqids_ft4/pdb_256.jsonl',
        mask_mode: str = 'conditional',
        **kwargs
    ) -> None:
        super().__init__()
        self.split = split
        self.pad = pad+1
        self.min_length = min_length
        self.mask_mode = mask_mode
        

        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
            
        if split=='train':
            lines = lines[:-100]
        else:
            lines = lines[-100:]

        self.entrys = {}
        for line in lines:
            entry = json.loads(line)
            name = list(entry.keys())[0]
            self.entrys.update(entry)

        self.names = list(self.entrys.keys())
        

    
    def __len__(self):
        return len(self.names)

    def get_pos_ids(self, chain_ids):
        uni_chain_id = chain_ids.unique()
        pos_encodings = torch.zeros_like(chain_ids)
        for i in uni_chain_id:
            mask = chain_ids==i
            pos_encodings[mask] = torch.arange(mask.sum())
        return pos_encodings
    
    @classmethod
    def mask_data(self, vq_ids, mode='conditional', mask_indices=None):
        device = vq_ids.device
        Inpaint = torch.tensor([self.tokenizer.vocab['[Inpaint]']], device=device)
        ZERO = torch.tensor([0], device=device)
        PAD = torch.tensor([self.tokenizer.vocab['[PAD]']], device=device)
        PAD_POS = torch.tensor([1024], device=device)

        Prompt = Inpaint
        num_res = vq_ids.shape[0]
        
        if mask_indices is None:
            if mode == 'conditional':
                N = random.randint(3, 6)
                K = random.randint(1, N-1)
            else:
                N = 1
                K = 0
            
            unmask_indices = Segment(vq_ids.shape[0], N, K)
    
            
            mask_indices = generate_pos_sequence(num_res, unmask_indices)
        else:
            unmask_indices = list(set(range(num_res))-set(mask_indices))
        
        def add_prompt(vq_ids, prompt):
            return torch.stack([torch.ones_like(vq_ids)*prompt, vq_ids]).permute(1,0).reshape(-1)


        
        vq_ids = torch.cat([Prompt, 
                            vq_ids[unmask_indices], 
                            add_prompt(vq_ids[mask_indices], PAD)])
        label_mask = torch.cat([
                        ZERO,
                        torch.zeros(len(unmask_indices), device=device), 
                        add_prompt(torch.ones(len(mask_indices), device=device), ZERO)])
        pos_ids = torch.cat([
                        PAD_POS,
                        torch.tensor(unmask_indices, device=device), 
                        add_prompt(torch.tensor(mask_indices, device=device), torch.tensor(mask_indices, device=device))])
        is_condition = torch.cat([
                        torch.tensor([1], device=device),
                        torch.ones_like(vq_ids[unmask_indices]), 
                        add_prompt(torch.zeros_like(vq_ids[mask_indices]), ZERO)])
        return vq_ids, label_mask, pos_ids, is_condition

    def shuffle_chain(self, vq_ids, chain_ids):
        vq_ids_new = [vq_ids[0:1]] # BOS
        chain_ids_new = [chain_ids[0:1]] # BOS

        uni_cids = torch.unique(chain_ids)

        uni_cids = uni_cids[uni_cids<10000]
        uni_cids_list = uni_cids.tolist()
        random.shuffle(uni_cids_list)
        SEP = torch.tensor([65516])

        for i, id in enumerate(uni_cids_list):
            mask = chain_ids==id
            vq_ids_new.append(vq_ids[mask])
            vq_ids_new.append(SEP)
            chain_ids_new.append(torch.zeros_like(chain_ids[mask])+i)
            chain_ids_new.append(torch.tensor([i]))
        
        vq_ids_new.append(vq_ids[-1:])# EOS
        chain_ids_new.append(chain_ids[-1:])# EOS
        vq_ids_new = torch.cat(vq_ids_new)
        chain_ids_new = torch.cat(chain_ids_new)
        return vq_ids_new, chain_ids_new
        

    
    def __getitem__(
        self, index
    ):  
        try:
            entry = self.entrys[self.names[index]]
            vq_ids = entry['vqid']
            if entry.get('chain'):
                chain_ids = entry['chain']
            else:
                chain_ids = [1 for i in range(len(vq_ids))]

            vq_ids = torch.tensor(vq_ids)
            chain_ids = torch.tensor(chain_ids)
            
            cids = chain_ids.unique().tolist()
            np.random.shuffle(cids)
            for cid in cids:
                if (chain_ids==cid).sum()<self.min_length:
                    continue
                break
            if (chain_ids==cid).sum()<self.min_length:
                return None
            vq_ids = vq_ids[chain_ids==cid]
            chain_ids = chain_ids[chain_ids==cid]

            vq_ids, label_mask, pos_ids, is_condition = self.mask_data(vq_ids, self.mask_mode)
            length = vq_ids.shape[0]
            vq_ids = F.pad(vq_ids, (0, self.pad-length), value=self.tokenizer.eos_token_id)
            label_mask = F.pad(label_mask, (0, self.pad-length), value=0)
            data_mask = torch.zeros_like(label_mask)
            data_mask[:length] = 1
            pos_ids = F.pad(pos_ids, (0, self.pad-length), value=1024)
            is_condition = F.pad(is_condition, (0, self.pad-length), value=0)
            ret = {'input_ids': vq_ids, 'data_mask':data_mask, 'label_mask':label_mask, 'pos_ids': pos_ids.long(), 'is_condition': is_condition}
            return ret
        except:
            return None