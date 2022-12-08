from time import time
from typing import Dict, List
import re
import nltk
import numpy as np
from tqdm import tqdm
from loguru import logger
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt', quiet=True)
import json
import os
import re
import pandas as pd
from time import time
from typing import List, Tuple
from functools import partial
from multiprocessing import Pool
import numpy as np
import gzip
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
from rouge_score import rouge_scorer
from loguru import logger
from config.dataset import DataPreparationConf, Reddit_TIFU_DataPreparationConf

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
SAVE_DIR = "/home/manhtt/anaconda3"
REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"', '\xa0': ""}

class DataPreparationBase(ABC):
    def __init__(self, conf: DataPreparationConf) -> None:
        self.conf = conf
        self.data_sep = ['is_train', 'is_val', 'is_test']
        self.setup()

    # def setup(self):
    #     if not os.path.exists(self.conf.output_path):
    #         os.system(f"mkdir -p {self.conf.output_path}")
    #         os.system(f"chmod -R 777 {self.conf.output_path}")
        
    #     return True

    def read_json(self, file_path: str):
        with open(file=file_path, mode='r', encoding='utf-8') as f:
            data = [json.loads(r.strip()) for r in f.readlines()]
        
        return data
    
    @abstractmethod
    def clean_text(self, text: str) -> str:
        raise NotImplementedError()
    
    @abstractmethod
    def sent_tokenize(self, doc: str) -> List[str]:
        raise NotImplementedError()
    
    @abstractmethod
    def tokenize(self, docs: List[str], name: str):
        raise NotImplementedError()
    
    def preprocess(self, example: List[List[str]], labels: List[int]):
        """
            Filter samples with no.tokens range and no.sents range
        """
        idxs = [idx for idx, s in enumerate(example) if len(s) > self.conf.min_ntokens]
        example = [example[idx][:self.conf.max_ntokens] for idx in idxs]
        labels = [labels[idx] for idx in idxs]

        example = example[:self.conf.min_nsents]
        labels = labels[:self.conf.max_nsents]

        if len(example) < self.conf.min_nsents:
            return None
        return example, labels
    
    @staticmethod
    def gen_ext_label(src: List[List[str]], tgt: str, top_k: int) -> List[int]:
        """Gen extractive label per doc `src`

        Args:
            src (List[List[str]]): list sentences, each sentence contains tokens
            tgt (str): the target text summarization
            top_k (int): no. sentences will be in the summarization

        Returns:
            List[int]: label
        """
        src_ = [' '.join(sent) for sent in src]

        r2_fscores = np.asarray([scorer.score(target=tgt, prediction=s)['rouge2'].fmeasure for s in src_])
        label = np.zeros_like(r2_fscores, dtype=np.int32)
        
        sorted_scores = (-r2_fscores).argsort()[:top_k]
        label[sorted_scores] = 1
        
        return label.tolist()

    def example_processor(self, example: Tuple[List[List[str]], str]):
        src, tgt = example
        ext_labels = self.gen_ext_label(src=src, tgt=tgt, top_k=self.conf.top_k)

        assert len(src) == len(ext_labels), f"The above document and label are not equal in length: {len(src)} and {len(ext_labels)}"

        preprocessed_data = self.preprocess(example=src, labels=ext_labels) 
        
        return preprocessed_data, tgt
    
    def process_data(self, df: pd.DataFrame, output_dir: str, file_name: str):
        file_path = os.path.join(output_dir, file_name)
    
        sources = df[self.conf.src_col_name]
        targets = [d.lower() for d in df[self.conf.tgt_col_name]]
        sources_tokenized = self.tokenize(docs=sources, name='source')
        del sources
    
        _example_processor = partial(self.example_processor)
        
        dataset = []
        pool = Pool(self.conf.n_processes)
        logger.info(f"Multiprocessing with {self.conf.n_processes} processes.")
        tic = time()

        for (preprocessed_data, target) in pool.map(_example_processor, zip(sources_tokenized, targets)):
            if preprocessed_data is not None:
                to_append = {'src': preprocessed_data[0], 'label': preprocessed_data[1]}
                to_append['tgt'] = '<q>'.join(self.sent_tokenize(target))
                dataset.append(to_append)
        pool.close()
        pool.join()
        
        del sources_tokenized
        del targets
        logger.info("Done in %.2f seconds"%(time()-tic))
        logger.info(f"Storaging data to: {file_path}")
        self.save(json_to_save=dataset, output_path=file_path)
                  
    def process_and_save_data(self, df: pd.DataFrame, output_dir: str):
        if not all(col_name  in df.columns for col_name in self.data_sep):
            raise ValueError(f"DataFrame missing one of columns: {self.data_sep}")
        
        for col_name in self.data_sep:
            df_temp: pd.DataFrame = df[df[col_name]]
            name = col_name.replace('is_', '')
            logger.info(f"Processing {name} data with no. samples: {len(df_temp)}")
            self.process_data(df=df_temp, output_dir=output_dir ,file_name=name+'.json')
                  
    @staticmethod
    def save(json_to_save, output_path, compression=False):
        """
        Save ``json_to_save`` to ``output_path`` with optional gzip compresssion
        specified by ``compression``.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logger.info("Saving to %s"%output_path)
        if compression:
            # https://stackoverflow.com/a/39451012
            json_str = json.dumps(json_to_save)
            json_bytes = json_str.encode("utf-8")
            with gzip.open((output_path + ".gz"), "w") as save_file:
                save_file.write(json_bytes)
        else:
            with open(output_path, "w") as save_file:
                save_file.write(json.dumps(json_to_save))

class VIDataPreparation(DataPreparationBase):
    
    def __init__(self, conf: DataPreparationConf) -> None:
        from py_vncorenlp import VnCoreNLP
        self.segmenter = VnCoreNLP(annotators=['wseg'], save_dir=SAVE_DIR)
        
        super().__init__(conf)
    
    @abstractmethod
    def build_dataframe(self, *args):
        raise NotImplementedError()
        
    def clean_text(self, text: str) -> str:
        text = re.sub(r"\s{2,}", '', text)
        return text.strip().lower()
    
    def sent_tokenize(self, doc: str) -> List[str]:
        sents = self.segmenter.word_segment(text=doc)
        
        return [sent.replace('_', ' ') for sent in sents]
    
    def tokenize(self, docs: List[str], name: str) -> List[List[List[str]]]:
        doc_segments: List[List[str]] = []
        for doc in tqdm(docs, desc=f'({name}) segment doc', ncols=100, nrows=5, total=len(docs)):
            d = self.segmenter.word_segment(text=doc)
            doc_segments.append(d)

        tokenized: List[List[List[str]]] = ([[token for token in sentence.split()] for sentence in doc] for doc in tqdm(
            tokenized, desc=f"({name}) segment to tokens", ncols=100, nrows=5, total=len(tokenized)))

        return tokenized

class ENDataPreparation(DataPreparationBase):
    def __init__(self, conf: DataPreparationConf) -> None:
        super().__init__(conf)
    
    @abstractmethod
    def build_dataframe(self, *args):
        raise NotImplementedError()
    
    def clean_text(self, text: str) -> str:
        text = re.sub('|'.join(list(REMAP.keys())), lambda m: REMAP.get(m.group()), text)
        text = re.sub(r'\s{1,}', ' ', text)
        return text.strip().lower()
    
    def sent_tokenize(self, doc: str) -> List[str]:
        return nltk.tokenize.sent_tokenize(text=doc)
    
    def tokenize(self, docs: List[str], name: str) -> List[List[List[str]]]:
        """Split docs into document, each document contains n sentences, each sentence contains n tokens.

        Args:
            `docs` (List[str]): contains documents

        Returns:
            List[List[List[str]]]: list of docs, where each doc is list of sentences, where each sentence is list of tokens.
        """
        tokenized: List[List[str]] = []
        for doc in tqdm(docs, desc=f'({name}) segment to sentences', ncols=100, nrows=5, total=len(docs)):
            doc_tokenized = self.sent_tokenize(doc.lower())
            tokenized.append(doc_tokenized)

        tokenized: List[List[List[str]]] = ([[token for token in nltk.tokenize.word_tokenize(sentence)] for sentence in doc] for doc in tqdm(
            tokenized, desc=f"({name}) segment to tokens", ncols=100, nrows=5, total=len(tokenized)))
        
        return tokenized
    
class VNDSDataPreparation(VIDataPreparation):
    def __init__(self, conf: DataPreparationConf) -> None:
        super().__init__(conf)

class RedditTIFUDataPreparation(ENDataPreparation):
    def __init__(self, conf: Reddit_TIFU_DataPreparationConf) -> None:
        super().__init__(conf)
    
    def build_data(self):
        posts = self.read_json(self.conf.file_path)
        if not all(k in posts[0].keys() for k in ['tldr', 'title', 'selftext', 'selftext_without_tldr']):
            raise KeyError("Missing one of keys: ['tldr', 'title', 'selftext', 'selftext_without_tldr']")
        
        long_posts = List[Dict[str, str]] = []
        short_posts = List[Dict[str, str]] = []
        
        for post in posts:
            if post['selftext_without_tldr']:
                if post['tldr']:
                    long_posts.append({'article': post['selftext_without_tldr'], 'summary': post['tldr']})
                if post['title']:
                    short_posts.append({'article': post['selftext_without_tldr'], 'summary': post['title']})
                    
        if not long_posts or not short_posts:
            raise ValueError()
        
        df_long = self.filter_and_clean(posts=long_posts)
        df_short = self.filter_and_clean(posts=short_posts)
        
        df_long = self.split_data(df=df_long, is_long=True)
        df_short = self.split_data(df=df_short, is_long=False)
        
        logger.info(f"Processing {self.conf.long_dir}")
        self.process_and_save_data(df=df_long, output_dir=self.conf.long_dir)
        logger.info(f"Processing {self.conf.short_dir}")
        self.process_and_save_data(df=df_short, output_dir=self.conf.short_dir)
        
    
    def filter_and_clean(self, posts: List[Dict[str, str]], article_col: str = 'article', summary_col: str = 'summary') -> pd.DataFrame:
        df = pd.DataFrame(data=posts)
        df[article_col] = df[article_col].apply(lambda x: self.clean_text(x))
        df[summary_col] = df[summary_col].apply(lambda x: self.clean_text(x))
        df = df[(df[article_col]!='') & (df[summary_col]!='')]
        df.drop_duplicates(subset= [article_col, summary_col], inplace=True)
        
        return df
    
    def split_data(self, df: pd.DataFrame, is_long: bool = True):
        test_size: int = self.conf.test_size_long if is_long else self.conf.test_size_short
        train_df_, df_test = train_test_split(df, test_size=test_size, shuffle=True, random_state=self.conf.random_state)
        df_train, df_val = train_test_split(train_df_, test_size=test_size)
        del train_df_

        is_train = [True]*len(df_train) + [False]*(len(df_val) + len(df_test))
        is_val = [False]*len(df_train) + [True]*len(df_val) + [False]*len(df_test)
        is_test = [False]*len(df_train) + [False]*len(df_val) + [True]*len(df_test)

        df = pd.concat([df_train, df_val, df_test])
        df['is_train'] = is_train
        df['is_val'] = is_val
        df['is_test'] = is_test
        
        return df


class BillSumDataPreparation(ENDataPreparation):
    def __init__(self, conf: DataPreparationConf) -> None:
        super().__init__(conf)
        
if __name__=='__main__':
    pass