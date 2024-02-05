from transformers import T5Tokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import json
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import stopwordsiso
import numpy as np
import fasttext
import jieba
from sklearn.metrics.pairwise import cosine_similarity

from utils import load_tokenizer


class CQRDataset(Dataset):
    def __init__(self, cfg, fold):
        work_dir = cfg.work_dir
        cfg = cfg.dataset
        self.lang = cfg.lang
        self.tokenizer = load_tokenizer(cfg)
        self.pad_token = self.tokenizer.pad_token
        self.pad_id = self.tokenizer.pad_token_id
        self.sp_token1 = cfg.sp_token1
        self.sp_token2 = cfg.sp_token2
        self.len_dialog = cfg.len_dialog
        self.len_query = cfg.len_query
        self._load_json(f'{work_dir}/{cfg.data_root}/{fold}.json')
        self.label_type = cfg.label_type
        self._load_utils()
        self.span_tag_dtype = torch.long if self.label_type in ['hard','defined'] else torch.float32
        self.wv_path = f'{work_dir}/wv/wiki.{self.lang}.bin'
        self.ftmodel = fasttext.load_model(self.wv_path)


    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        source = self._get_source(index)
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target = self._get_gold(index)
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()
        span_tag = self._get_span_tag(index)
        token = self.tokens[index]
        query = self.queries[index]
        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_mask': target_mask.to(dtype=torch.long),
            'span_tag': span_tag.to(dtype=self.span_tag_dtype),
            'token': token,
            'query': query,
        }

    def _load_utils(self):
        if self.label_type=='defined':
            return
        skip_tokens = ['\r','\n',' ', '　']
        if self.lang == 'en':
            nltk.download('stopwords')
            self.ps = PorterStemmer()
            self.en_tokenizer = RegexpTokenizer(r'\w+') # get rid of punctuation.
            self.stop_words = stopwords.words('english') + skip_tokens #stopwordsiso's stopwords for english contains words too much.
        elif self.lang == 'zh':
            self.stop_words = list(stopwordsiso.stopwords(self.lang)) + skip_tokens

    def _load_json(self, json_path):
        def _get_list(item):
            item_l = []
            for data in self.json_data:
                item_l.append(data[item])
            return item_l

        with open(json_path, 'r') as f:
            self.json_data = json.load(f)

        self.dialogs = _get_list('dialog')
        self.queries = _get_list('query')
        self.golds = _get_list('gold')
        self.tokens = _get_list('token')

    def _insert_sp_token(self, text_l):
        text = ''
        for t in text_l:
            text += t+self.sp_token1
        return text

    def _get_source(self, index):
        dialog_query = self._insert_sp_token(self.dialogs[index])+self.queries[index]+self.sp_token2
        source = self.tokenizer.batch_encode_plus(
            [dialog_query],
            max_length=self.len_dialog+self.len_query,
            padding='max_length',
            truncation=True,
            add_special_tokens=True, #add special token <\s> at the tail for T5.
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        return source

    def _get_gold(self, index):
        target = self.tokenizer.batch_encode_plus(
            [self.golds[index]],
            max_length=self.len_query,
            padding='max_length',
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        return target

    def _tokens2span(self,tokens):
        tokens_span = []
        start = 0 #tokens starts with ''
        for token in tokens:
            end = start + len(token)
            tokens_span.append([start, end])
            start = end
        return tokens_span

    def _which_idx(self, start, end, tokens_span):
        for idx, span in enumerate(tokens_span):
            lower, upper = span
            if lower <= start <= upper-1:
                s_idx = idx
            if lower <= end-1 <= upper-1:
                e_idx = idx
        return s_idx, e_idx

    def _get_tag_idx(self, entity, tokens):
        entity = re.sub(r'(\+|\*)', r'\\\1', entity) # escape regular expression.
        tokens = [t for t in tokens if t != self.pad_token] #skip padding part for matching.
        tokens_span = self._tokens2span(tokens)
        tag_idx=[]
        if entity == 'X':
            entity = re.compile(r'(?<!\[)X(?![1-2]\])') # avoid matching with special token "[X1]" and "[X2]"

        for m in re.finditer(entity, ''.join(tokens)):
            start, end = m.span()
            s_idx, e_idx = self._which_idx(start, end, tokens_span)
            tag_idx.append([s_idx, e_idx]) # entity exist in the idx of range (s_idx, e_idx).
        return tag_idx

    def _span_tokenize(self, text):
        if self.lang=='zh':
            tokens = list(jieba.cut(text))
        if self.lang=='en':
            tokens = self.en_tokenizer.tokenize(text)
        return tokens

    def _stemming(self, tokens):
        if self.lang == 'en':
            stem_tokens = [self.ps.stem(token) for token in tokens]
        if self.lang == 'zh':
            stem_tokens = tokens # no need to stem for chinese
        return stem_tokens

    def _remove_stopwords(self, tokens):
        tokens = list(filter(lambda x: x.lower() not in self.stop_words, tokens))
        return tokens

    def _get_span_tag(self, index):
        source = self._get_source(index)
        source_ids = source['input_ids'].squeeze()
        span_tag = torch.tensor([0]*len(source_ids), dtype=self.span_tag_dtype)
        span_tag = torch.where(source_ids!=self.pad_id, span_tag, torch.tensor(-100, dtype=span_tag.dtype)) #use tag=100 for padding token, where criterion in torch.nn neglect loss calculation for -100 in default.
        src_tokens = self.tokenizer.convert_ids_to_tokens(source_ids)
        if self.label_type == 'defined':
            tokens = self.tokens[index]
            for token in tokens:
                tag_idx = self._get_tag_idx(token, src_tokens)
                for s,e in tag_idx:
                    span_tag[s]=1
                    span_tag[s+1:e+1]=2
            return span_tag

        dialog = ' '.join(self.dialogs[index])
        query = self.queries[index]
        gold = self.golds[index]
        d_tokens, q_tokens, g_tokens = self._span_tokenize(dialog), self._span_tokenize(query), self._span_tokenize(gold)
        d_tokens, q_tokens, g_tokens = self._remove_stopwords(d_tokens), self._remove_stopwords(q_tokens), self._remove_stopwords(g_tokens)
        candidates_tokens = list(set(g_tokens) - set(q_tokens))
        if candidates_tokens == []:
            return span_tag

        d_tokens = list(set(d_tokens))
        d_stems, c_stems = self._stemming(d_tokens), self._stemming(candidates_tokens)
        d_vecs = np.stack([self.ftmodel.get_word_vector(d) for d in d_stems])
        c_vecs = np.stack([self.ftmodel.get_word_vector(c) for c in c_stems])
        mat_sim = cosine_similarity(c_vecs, d_vecs)
        sims = np.max(mat_sim, axis=0)
        sims = np.where(0.99<sims, 1, sims) #ceil the floating point.

        if self.label_type == 'soft':
            sims = torch.tensor(sims, dtype=span_tag.dtype)
            for d_token, sim in zip(d_tokens, sims):
                tag_idx = self._get_tag_idx(d_token, src_tokens)
                for s_idx, e_idx in tag_idx:
                    span_tag[s_idx:e_idx+1] = sim

        elif self.label_type == 'hard':
            sims = np.where(sims!=1, 0, sims)
            sims = torch.tensor(sims, dtype=span_tag.dtype)
            idxs = np.where(sims==1)[0]
            for idx in idxs:
                d_token = d_tokens[idx]
                tag_idx = self._get_tag_idx(d_token, src_tokens)
                for s_idx, e_idx in tag_idx:
                    span_tag[s_idx]=1
                    span_tag[s_idx+1:e_idx+1]=2

        return span_tag


def collate_fn(batch):
    return {
        'source_ids': torch.stack([x['source_ids'] for x in batch], dim=0),
        'source_mask': torch.stack([x['source_mask'] for x in batch], dim=0),
        'target_ids': torch.stack([x['target_ids'] for x in batch], dim=0),
        'target_mask': torch.stack([x['target_mask'] for x in batch], dim=0),
        'span_tag': torch.stack([x['span_tag'] for x in batch], dim=0),
        'token': [x['token'] for x in batch],
        'query': [x['query'] for x in batch],
    }


def load_dataiter(cfg, folds):
    folds_iter = []
    for fold in folds:
        fold_data = CQRDataset(cfg, fold)
        fold_iter = DataLoader(fold_data, batch_size=cfg.dataset.batch_size[fold], shuffle=cfg.dataset.shuffle[fold], collate_fn=collate_fn)
        folds_iter.append(fold_iter)
    if len(folds)==1:
        return folds_iter[0]
    return folds_iter