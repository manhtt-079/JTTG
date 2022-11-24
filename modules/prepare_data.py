import copy
import json
import os
import logging
import math
import torch
import re
import pandas as pd
from time import time
from typing import List, Union, Optional, Any, Tuple
from functools import partial
from multiprocessing import Pool
import itertools
import argparse
import numpy as np
from tqdm import tqdm
import gzip
from dataclasses import dataclass
from rouge_score import rouge_scorer

from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt', quiet=True)

logger = logging.getLogger(__name__)
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"', '\xa0': ""}

def clean_text(text: str):
    text = re.sub('|'.join(list(REMAP.keys())), lambda m: REMAP.get(m.group()), text)
    text = re.sub(r'\s{2,}', '', text)
    return text.strip().lower()


def do_tokenize(docs: List[str], name: str) -> List[List[List[str]]]:
    """Split docs into document, each document contains n sentences, each sentence contains n tokens.

    Args:
        `docs` (List[str]): contains documents

    Returns:
        List[List[List[str]]]: list of docs, where each doc is list of sentences, where each sentence is list of tokens.
    """
    tokenized: List[List[str]] = []
    for doc in tqdm(docs, desc=f'({name}) segment to sentences', ncols=100, nrows=5, total=len(docs)):
        doc_tokenized = sent_tokenize(doc.lower())
        tokenized.append(doc_tokenized)

    tic = time()
    tokenized: List[List[List[str]]] = ([[token for token in word_tokenize(sentence)] for sentence in doc] for doc in tqdm(
        tokenized, desc=f"({name}) segment to tokens", ncols=100, nrows=5, total=len(tokenized)))

    logger.info("Done in %.2f seconds", time()-tic)

    return tokenized

def save(json_to_save, output_path, compression=False):
    """
    Save ``json_to_save`` to ``output_path`` with optional gzip compresssion
    specified by ``compression``.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info("Saving to %s", output_path)
    if compression:
        # https://stackoverflow.com/a/39451012
        json_str = json.dumps(json_to_save)
        json_bytes = json_str.encode("utf-8")
        with gzip.open((output_path + ".gz"), "w") as save_file:
            save_file.write(json_bytes)
    else:
        with open(output_path, "w") as save_file:
            save_file.write(json.dumps(json_to_save))

def preprocess(example: List[List[str]],
               labels: List[int],
               min_ntokens: int = 5,
               max_ntokens: int = 200,
               min_nsents: int = 3,
               max_nsents: int = 100) -> Union[None, Tuple[List[List[str]], List[int]]]:
    
    """Removes sentences that are too long/short and examples that have few/many sentences.

    Args:
        `example` (List[List[str]]): Input sample: List sentences and each sentence is list tokens
        `labels` (List[int]): Label for extractive summarization, annotate each sentence in [0, 1]
        `min_ntokens` (int, optional): minimum number of tokens in each sentence. Defaults to 5.
        `max_ntokens` (int, optional): maximum number of tokens in each sentence. Defaults to 200.
        `min_nsents` (int, optional): minimum number of sentences in `example`. Defaults to 3.
        `max_nsents` (int, optional): maximum number of sentences in `example`. Defaults to 100.

    Returns:
        Union[None, Tuple[List[List[str]], List[int]]]: `example` and `labels` were processed/ None
    """

    idxs = [idx for idx, s in enumerate(example) if len(s) > min_ntokens]

    example = [example[idx][:max_ntokens] for idx in idxs]
    labels = [labels[idx] for idx in idxs]

    example = example[:max_nsents]
    labels = labels[:max_nsents]

    if len(example) < min_nsents:
        return None
    return example, labels

def gen_label(source: List[List[str]], target: str, top_k: int):
    source_ = [' '.join(sent) for sent in source]

    r2_fscores = np.asarray([scorer.score(target=target, prediction=s)['rouge2'].fmeasure for s in source_])
    label = np.zeros_like(r2_fscores, dtype=np.int32)
    
    sorted_scores = (-r2_fscores).argsort()[:top_k]
    label[sorted_scores] = 1
    
    return label.tolist()

def example_processor(example: Tuple[List[List[str]], str], args: 'ArgsConf'):
    source, target = example
    label = gen_label(source=source, target=target, top_k=args.top_k)

    assert len(source) == len(label), f"The above document and label are not equal in length: {len(source)} and {len(label)}"

    if args.no_preprocess:
        preprocessed_data = source, label
    else:
        preprocessed_data = preprocess(source, label, args.min_ntokens, args.max_ntokens, args.min_nsents, args.max_nsents) 
    
    return preprocessed_data, target

def save(json_to_save, output_path, compression=False):
    """
    Save ``json_to_save`` to ``output_path`` with optional gzip compresssion
    specified by ``compression``.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info("Saving to %s", output_path)
    if compression:
        # https://stackoverflow.com/a/39451012
        json_str = json.dumps(json_to_save)
        json_bytes = json_str.encode("utf-8")
        with gzip.open((output_path + ".gz"), "w") as save_file:
            save_file.write(json_bytes)
    else:
        with open(output_path, "w") as save_file:
            save_file.write(json.dumps(json_to_save))

def process_data(file_path: str, args: 'ArgsConf'):
    df: pd.DataFrame = pd.read_csv(filepath_or_buffer=file_path)
    
    src = df[args.src_col_name]
    tgt = [d.lower() for d in df[args.tgt_col_name]]
    source_tokenized = do_tokenize(docs=src, name='src')
    del src
    
    _example_processor = partial(example_processor, args=args)
    
    dataset = []
    pool = Pool(args.n_processes)
    logger.info("Processing")
    tic = time()

    for (preprocessed_data, target) in pool.map(_example_processor, zip(source_tokenized, tgt)):
        if preprocessed_data is not None:
            to_append = {'src': preprocessed_data[0], 'label': preprocessed_data[1]}
            to_append['tgt'] = '<q>'.join(sent_tokenize(target))
            dataset.append(to_append)
    pool.close()
    pool.join()
    
    del source_tokenized
    del tgt
    logger.info("Done in %.2f seconds", time()-tic)
    logger.info("Storaged data to ./to_test.json")
    save(json_to_save=dataset, output_path="./to_test.json")

@dataclass
class ArgsConf:
    src_col_name: str = 'article'
    tgt_col_name: str = 'highlights'
    top_k: int = 3
    n_processes: int = 2
    no_preprocess: bool = False
    min_ntokens: int = 5
    max_ntokens: int = 200
    min_nsents: int = 3
    max_nsents: int = 100
    
    
if __name__=='__main__':
    args = ArgsConf()
    logger.info("Preparing data...")
    process_data(file_path="./to_test.csv", args=args)
    
    
