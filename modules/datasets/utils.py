import json
import os
import re
import pandas as pd
from time import time
from typing import List, Union, Tuple, Optional
from functools import partial
from multiprocessing import Pool
import numpy as np
import gzip
from abc import ABC, abstractmethod
from rouge_score import rouge_scorer
from loguru import logger
from config.config import DataPreparationConf

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

class DataPreparationBase(ABC):
    def __init__(self, conf: DataPreparationConf) -> None:
        self.conf = conf
        self.data_sep = ['is_train', 'is_val', 'is_test']
        self.setup()

    def setup(self):
        if not os.path.exists(self.conf.output_path):
            os.system(f"mkdir -p {self.conf.output_path}")
            os.system(f"chmod -R 777 {self.conf.output_path}")
        
        return True

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

        if self.conf.no_preprocess:
            preprocessed_data = src, ext_labels
        else:
            preprocessed_data = self.preprocess(example=src, labels=ext_labels) 
        
        return preprocessed_data, tgt
    
    def process_data(self, df: pd.DataFrame, file_name: str):
        file_path = os.path.join(self.conf.output_path, file_name)
        df: pd.DataFrame = pd.read_csv(filepath_or_buffer=self.conf.file_path)
    
        sources = df[self.conf.src_col_name]
        targets = [d.lower() for d in df[self.conf.tgt_col_name]]
        sources_tokenized = self.tokenize(docs=sources, name='source')
        del sources
    
        _example_processor = partial(self.example_processor)
        
        dataset = []
        pool = Pool(self.conf.n_processes)
        logger.info("Processing")
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
        
    def process_and_save_data(self):
        if self.conf.is_concat:
            df: pd.DataFrame = pd.read_csv(filepath_or_buffer=self.conf.file_path)
            
            if not all(col_name  in df.columns for col_name in self.data_sep):
                raise ValueError(f"DataFrame missing one of columns: {self.data_sep}")
            
            for sep in self.data_sep:
                df_temp: pd.DataFrame = df[df[sep]]
                name = sep.replace('is_', '')
                logger.info(f"Processing {name} data")
                self.process_data(df=df_temp, file_name=name+'.json') 
            return
        
        if self.conf.val_path and self.conf.test_path:
            for p, name in zip([self.conf.file_path, self.conf.val_path, self.conf.test_path], ['train', 'val', 'test']):
                df: pd.DataFrame = pd.read_csv(filepath_or_buffer=p)
                logger.info(f"Processing {name} data")
                self.process_data(df=df, file_name=name+'.json')
                  
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


if __name__=='__main__':
    pass