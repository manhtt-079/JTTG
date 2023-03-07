import re
import json
import math
import random
import torch

from typing import List, Any, Dict
from torch.utils.data import Dataset, DataLoader, IterableDataset

def pad(data: List[List[int]], pad_id: int, width=None, pad_on_left=False, nearest_multiple_of=False):
    """
    Pad ``data`` with ``pad_id`` to ``width`` on the right by default but if
    ``pad_on_left`` then left.
    """
    if not width:
        width = max(len(d) for d in data)
    if nearest_multiple_of:
        width = math.ceil(width / nearest_multiple_of) * nearest_multiple_of
    if pad_on_left:
        rtn_data = [[pad_id] * (width - len(d)) + d for d in data]
    else:
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
    return rtn_data

def batch_collate(batch):
    
    result = {}
    e = batch[0]
    for key in e:
        feature_list = [d[key] for d in batch]
        if key == 'sent_rep_ids':
            feature_list = pad(feature_list, -1)
            sent_rep_token_ids = torch.tensor(feature_list)
            
            sent_rep_mask = ~(sent_rep_token_ids == -1)
            sent_rep_token_ids[sent_rep_mask == -1] = 0
            
            result['sent_rep_ids'] = sent_rep_token_ids
            result['sent_rep_mask'] = sent_rep_mask
            continue
        
        if key == 'decoder_input_ids':
            tgt_ids = feature_list
            
            attention_mask = [[1] * len(ids) for ids in tgt_ids]
            
            tgt_ids_width = max(len(ids) for ids in tgt_ids)
            tgt_ids = pad(tgt_ids, 0, width=tgt_ids_width)
            tgt_ids = torch.tensor(tgt_ids)
            
            attention_mask = pad(attention_mask, 0)
            attention_mask = torch.tensor(attention_mask)
            
            result['decoder_input_ids'] = tgt_ids
            result['decoder_attention_mask'] = attention_mask
            continue
        
        if key == 'src_ext_input_ids' or key == 'src_abs_input_ids':
            input_ids = feature_list
            
            attention_mask = [[1] * len(ids) for ids in input_ids]
            
            input_ids_width = max(len(ids) for ids in input_ids)
            input_ids = pad(input_ids, 0, width=input_ids_width)
            input_ids = torch.tensor(input_ids)
            
            attention_mask = pad(attention_mask, 0)
            attention_mask = torch.tensor(attention_mask)
            
            if key == 'src_ext_input_ids':
                result['src_ext_input_ids'] = input_ids
                result['src_ext_attention_mask'] = attention_mask
            else:
                result['src_abs_input_ids'] = input_ids
                result['src_abs_attention_mask'] = attention_mask
            continue
        
        if key in ('label', 'src_token_type_ids'):
            feature_list = pad(feature_list, 0)
        feature_list = torch.tensor(feature_list)
        result[key] = feature_list
    
    return result


class ExAbDataset(Dataset):
    def __init__(
        self,
        tokenizer: torch.nn.Module,
        data_path: str,
        src_max_length: int = 1024,
        tgt_max_length: int = 256
    ) -> None:
        
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.src_max_length = src_max_length
        self.tgt_max_length = tgt_max_length
        self.sep_token = self.tokenizer.sep_token
        self.sep_token_id = self.tokenizer.sep_token_id
        self.cls_token = self.tokenizer.cls_token
        self.cls_token_id = self.tokenizer.cls_token_id
        self.prefix = ''
        
        # add prefix if model architecture based on t5
        # vit5 is not
        if re.search(r'(?<!vi)(t5)', tokenizer.name_or_path):
            self.prefix = 'summarize: '
        
        self.data = self.load_json()
        
    
    def load_json(self) -> List[Dict[str, Any]]:
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def tokenize(self, src: List[List[str]], tgt: str, label: List[int]):
        
        tgt_ = tgt.replace('<q>', ' ')
        tgt_inputs = self.tokenizer(tgt_, truncation=True, max_length=self.tgt_max_length)
        
        src: List[str] = [' '.join(e) for e in src]
        src_abs: str = self.prefix + ' '.join(src)
        # [SEP] [CLS] sent1 [SEP] [CLS] sent2...
        src_ext: str = f' {self.sep_token} {self.cls_token} '.join(src)
        
        src_abs_inputs = self.tokenizer(src_abs, truncation=True, max_length=self.src_max_length)
        src_ext_inputs = self.tokenizer(src_ext, truncation=True, max_length=self.src_max_length)
        
        input_ids = src_ext_inputs['input_ids']
        if input_ids[-2] == self.sep_token_id:
            # case: xxxx [SEP] [SEP]
            src_ext_inputs['input_ids'] = input_ids[:-1]
        if input_ids[-2] == self.cls_token_id:
            # case: xxxx [SEP] [CLS] [SEP]
            src_ext_inputs['input_ids'] = input_ids[:-2]
        del input_ids
        
        sent_rep_ids = [-1] + [i for i, idx in enumerate(src_ext_inputs['input_ids']) if idx==self.sep_token_id]
        sent_rep_ids = sent_rep_ids[1:]
        
        label = label[:len(sent_rep_ids)]

        return src_ext_inputs['input_ids'], src_abs_inputs['input_ids'], tgt_inputs['input_ids'], label, sent_rep_ids   
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data[idx]
        src_ext_input_ids, src_abs_input_ids, decoder_input_ids, label, sent_rep_ids = self.tokenize(row['src'], row['tgt'], row['label'])
        
        return {
            "src_ext_input_ids": src_ext_input_ids, 
            "src_abs_input_ids": src_abs_input_ids,
            "decoder_input_ids": decoder_input_ids,
            "sent_rep_ids": sent_rep_ids, "label": label
        }
        
class IterableExAbDataset(IterableDataset):
    def __init__(self,
                 tokenizer: torch.nn.Module,
                 data_path: str,
                 src_max_length: int = 1024,
                 tgt_max_length: int = 256
    ) -> None:
        
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.src_max_length = src_max_length
        self.tgt_max_length = tgt_max_length
        self.sep_token = self.tokenizer.sep_token
        self.sep_token_id = self.tokenizer.sep_token_id
        self.cls_token = self.tokenizer.cls_token
        self.cls_token_id = self.tokenizer.cls_token_id
        self.random_seed = 42
        
        self.data = self.load_json()
        random.Random(self.random_seed).shuffle(self.data)
        
    
    def load_json(self) -> List[Dict[str, Any]]:
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def tokenize(self, src: List[List[str]], tgt: str, label: List[int]):
        
        tgt_ = tgt.replace('<q>', ' ')
        tgt_inputs = self.tokenizer(tgt_, truncation=True, max_length=self.tgt_max_length)
        
        src: str = f' {self.sep_token} {self.cls_token} '.join([' '.join(e) for e in src])
        src_inputs = self.tokenizer(src, truncation=True, max_length=self.src_max_length)
        input_ids = src_inputs['input_ids']
        if input_ids[-2] == self.sep_token_id:
            # case: xxxx [SEP] [SEP]
            src_inputs['input_ids'] = input_ids[:-1]
        if input_ids[-2] == self.cls_token_id:
            # case: xxxx [SEP] [CLS] [SEP]
            src_inputs['input_ids'] = input_ids[:-2]
        del input_ids
        
        src_inputs['attention_mask'] = src_inputs['attention_mask'][:len(src_inputs['input_ids'])]
        
        sent_rep_ids = [-1] + [i for i, idx in enumerate(src_inputs['input_ids']) if idx==self.sep_token_id]
        segs = [sent_rep_ids[i] - sent_rep_ids[i-1] for i in range(1, len(sent_rep_ids))]
        sent_rep_ids = sent_rep_ids[1:]
        segment_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segment_ids += s*[0]
            else:
                segment_ids += s*[1]

        label = label[:len(sent_rep_ids)]

        return src_inputs['input_ids'], segment_ids, tgt_inputs['input_ids'], label, sent_rep_ids   
        
    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for row in self.data:
            src_ids, src_token_type_ids, tgt_ids, label, sent_rep_ids = self.tokenize(row['src'], row['tgt'], row['label'])
            yield {
                "src_ids": src_ids, "src_token_type_ids": src_token_type_ids,
                "tgt_ids": tgt_ids, "label": label,
                "sent_rep_ids": sent_rep_ids
            }
            
            
def dataset(
    tokenizer: torch.nn.Module,
    data_path: str,
    shuffle: bool = True,
    src_max_length: int = 1024,
    tgt_max_length: int = 256,
    batch_size: int = 4,
    num_workers: int = 16
    ):

    dataset = ExAbDataset(tokenizer=tokenizer, data_path=data_path, src_max_length=src_max_length, tgt_max_length=tgt_max_length)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=batch_collate, shuffle=shuffle, num_workers=num_workers)