from time import time
from typing import List
import re
import nltk
import numpy as np
from tqdm import tqdm
from loguru import logger
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt', quiet=True)

from .utils import DataPreparationBase
from config.config import DataPreparationConf


SAVE_DIR = "/home/manhtt/anaconda3"
class VIDataPreparation(DataPreparationBase):
    
    def __init__(self, conf: DataPreparationConf) -> None:
        from py_vncorenlp import VnCoreNLP
        self.segmenter = VnCoreNLP(annotators=['wseg'], save_dir=SAVE_DIR)
        
        super().__init__(conf)
        
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



REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"', '\xa0': ""}
class ENDataPreparation(DataPreparationBase):
    def __init__(self, conf: DataPreparationConf) -> None:
        super().__init__(conf)
    
    def clean_text(self, text: str) -> str:
        text = re.sub('|'.join(list(REMAP.keys())), lambda m: REMAP.get(m.group()), text)
        text = re.sub(r'\s{2,}', '', text)
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