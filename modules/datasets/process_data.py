import re
import nltk
import numpy as np
from tqdm import tqdm
import json
import unicodedata
import os
import pandas as pd
from time import time
from typing import List, Tuple, Dict
from functools import partial
from multiprocessing import Pool
import gzip
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
from rouge_score import rouge_scorer
from loguru import logger
import underthesea
from config.dataset import (DataPreparationConf, 
                            Reddit_TIFU_DataPreparationConf, 
                            BillSum_DataPreparationConf, 
                            VnDS_DataPreparationConf,
                            ViNewsQA_DataPRConf,
                            ViQuAD_DataPRConf
                            )
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt', quiet=True)

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

class DataPreparationBase(ABC):
    def __init__(self, conf: DataPreparationConf) -> None:
        self.conf = conf
        self.data_sep = ['is_train', 'is_val', 'is_test']


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

        example = example[:self.conf.max_nsents]
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
        targets = [d for d in df[self.conf.tgt_col_name]]
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
                save_file.write(json.dumps(json_to_save, ensure_ascii=False))

class VIDataPreparation(DataPreparationBase):
    VI_RE_MAP: Dict[str, str] = {"òa": "oà", "Òa": "Oà", "ÒA": "OÀ", "óa": "oá", "Óa": "Oá", "ÓA": "OÁ", "ỏa": "oả", "Ỏa": "Oả", "ỎA": "OẢ", 
                 "õa": "oã", "Õa": "Oã", "ÕA": "OÃ", "ọa": "oạ", "Ọa": "Oạ", "ỌA": "OẠ", "òe": "oè", "Òe": "Oè", "ÒE": "OÈ",
                 "óe": "oé", "Óe": "Oé", "ÓE": "OÉ", "ỏe": "oẻ", "Ỏe": "Oẻ", "ỎE": "OẺ", "õe": "oẽ", "Õe": "Oẽ", "ÕE": "OẼ", 
                 "ọe": "oẹ", "Ọe": "Oẹ", "ỌE": "OẸ", "ùy": "uỳ", "Ùy": "Uỳ", "ÙY": "UỲ", "úy": "uý", "Úy": "Uý", "ÚY": "UÝ", 
                 "ủy": "uỷ", "Ủy": "Uỷ", "ỦY": "UỶ", "ũy": "uỹ", "Ũy": "Uỹ", "ŨY": "UỸ", "ụy": "uỵ", "Ụy": "Uỵ", "ỤY": "UỴ"}
    
    def __init__(self, conf: DataPreparationConf) -> None:
        super().__init__(conf)
    
    @abstractmethod
    def build_data(self):
        raise NotImplementedError()
        
    def clean_text(self, text: str) -> str:
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(pattern='|'.join(self.VI_RE_MAP.keys()), repl=lambda m: self.VI_RE_MAP.get(m.group()), string=text)
        text = re.sub(r"\s{1,}", ' ', text)
        
        return text.strip()
    
    def sent_tokenize(self, doc: str) -> List[str]:
        return underthesea.sent_tokenize(doc)
    
    def tokenize(self, docs: List[str], name: str) -> List[List[List[str]]]:
        doc_segments: List[List[str]] = []
        for doc in tqdm(docs, desc=f'({name}) segment doc', ncols=100, nrows=5, total=len(docs)):
            d = self.sent_tokenize(doc)
            doc_segments.append(d)

        tokenized: List[List[List[str]]] = ([[token for token in underthesea.word_tokenize(sentence)] for sentence in doc] for doc in tqdm(
            doc_segments, desc=f"({name}) segment to tokens", ncols=100, nrows=5, total=len(doc_segments)))
        del doc_segments
        return tokenized

class ENDataPreparation(DataPreparationBase):
    EN_RE_MAP: Dict[str, str] = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
                                "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"', '\xa0': ""}
    
    def __init__(self, conf: DataPreparationConf) -> None:
        super().__init__(conf)
    
    @abstractmethod
    def build_data(self):
        raise NotImplementedError()
    
    def clean_text(self, text: str) -> str:
        text = re.sub('|'.join(list(self.EN_RE_MAP.keys())), lambda m: self.EN_RE_MAP.get(m.group()), text)
        text = re.sub(r'\s{1,}', ' ', text)
        return text.strip()
    
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
            doc_tokenized = self.sent_tokenize(doc)
            tokenized.append(doc_tokenized)

        tokenized: List[List[List[str]]] = ([[token for token in nltk.tokenize.word_tokenize(sentence)] for sentence in doc] for doc in tqdm(
            tokenized, desc=f"({name}) segment to tokens", ncols=100, nrows=5, total=len(tokenized)))
        
        return tokenized
    
class VNDSDataPreparation(VIDataPreparation):
    def __init__(self, conf: VnDS_DataPreparationConf) -> None:
        super().__init__(conf)
        self.setup()
        logger.info("Setup done!")
    
    def setup(self):
        if not os.path.exists(self.conf.output_dir):
            os.system(f"mkdir -p {self.conf.output_dir}")
            os.system(f"chmod -R 777 {self.conf.output_dir}")
        
        return True
    
    def filter_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.conf.src_col_name] = df[self.conf.src_col_name].apply(lambda x: self.clean_text(x))
        df[self.conf.tgt_col_name] = df[self.conf.tgt_col_name].apply(lambda x: self.clean_text(x))
        df = df[(df[self.conf.src_col_name]!='') & (df[self.conf.tgt_col_name]!='')]
        df.drop_duplicates(subset= [self.conf.src_col_name, self.conf.tgt_col_name], inplace=True)
        
        return df
    
    def process_and_save_data(self, df: pd.DataFrame, output_dir: str, name: str):
        logger.info(f"Processing {name} data with no. samples: {len(df)}")
        self.process_data(df=df, output_dir=output_dir, file_name=name+'.json')
    
    def build_data(self):
        logger.info("Reading raw_data")
        df: pd.DataFrame = pd.read_csv(self.conf.data_path)

        logger.info('Filter and clean data')
        df_train = self.filter_and_clean(df[df.is_train].copy(deep=True))
        df_val = self.filter_and_clean(df[df.is_val].copy(deep=True))
        df_test = self.filter_and_clean(df[df.is_test].copy(deep=True))
        
        logger.info(f"Processing training data")
        self.process_and_save_data(df=df_train, output_dir=self.conf.output_dir, name='train')
        logger.info(f"Processing val data")
        self.process_and_save_data(df=df_val, output_dir=self.conf.output_dir, name='val')
        logger.info(f"Processing test data")
        self.process_and_save_data(df=df_test, output_dir=self.conf.output_dir, name='test')


class ViNewsQADataPreparation(VIDataPreparation):
    def __init__(self, conf: ViNewsQA_DataPRConf) -> None:
        super().__init__(conf)
        self.setup()
        logger.info("Setup done!")
    
    def setup(self):
        if not os.path.exists(self.conf.output_dir):
            os.system(f"mkdir -p {self.conf.output_dir}")
            os.system(f"chmod -R 777 {self.conf.output_dir}")
        
        return True

    def filter_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        context, answer = self.conf.src_col_name.split(',')
        text = df[context] + ' ' + df[answer]
        text = text.apply(lambda x: self.clean_text(x))
        question = df[self.conf.tgt_col_name].apply(lambda x: self.clean_text(x))
        
        df = pd.DataFrame(zip(text, question), columns=[self.conf.src_col_name, self.conf.tgt_col_name])
        df = df[(df[self.conf.src_col_name]!='') & (df[self.conf.tgt_col_name]!='')]
        df.drop_duplicates(subset=[self.conf.src_col_name, self.conf.tgt_col_name], inplace=True)
        
        return df
    
    def process_and_save_data(self, df: pd.DataFrame, output_dir: str, name: str):
        logger.info(f"Processing {name} data with no. samples: {len(df)}")
        self.process_data(df=df, output_dir=output_dir, file_name=name+'.json')
    
    def build_data(self):
        logger.info("Reading raw_data")
        df_train: pd.DataFrame = pd.read_csv(self.conf.train_path)
        df_val: pd.DataFrame = pd.read_csv(self.conf.val_path)
        df_test: pd.DataFrame = pd.read_csv(self.conf.test_path)

        logger.info('Filter and clean data')
        df_train = self.filter_and_clean(df_train)
        df_val = self.filter_and_clean(df_val)
        df_test = self.filter_and_clean(df_test)
        
        logger.info(f"Processing training data")
        self.process_and_save_data(df=df_train, output_dir=self.conf.output_dir, name='train')
        logger.info(f"Processing val data")
        self.process_and_save_data(df=df_val, output_dir=self.conf.output_dir, name='val')
        logger.info(f"Processing test data")
        self.process_and_save_data(df=df_test, output_dir=self.conf.output_dir, name='test')

class ViQuADDataPreparation(VIDataPreparation):
    def __init__(self, conf: ViQuAD_DataPRConf) -> None:
        super().__init__(conf)
        self.setup()
        logger.info("Setup done!")
    
    def setup(self):
        if not os.path.exists(self.conf.output_dir):
            os.system(f"mkdir -p {self.conf.output_dir}")
            os.system(f"chmod -R 777 {self.conf.output_dir}")
        
        return True

    def filter_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        context, answer = self.conf.src_col_name.split(',')
        text = df[context] + ' ' + df[answer]
        text = text.apply(lambda x: self.clean_text(x))
        question = df[self.conf.tgt_col_name].apply(lambda x: self.clean_text(x))
        
        df = pd.DataFrame(zip(text, question), columns=[self.conf.src_col_name, self.conf.tgt_col_name])
        df = df[(df[self.conf.src_col_name]!='') & (df[self.conf.tgt_col_name]!='')]
        df.drop_duplicates(subset=[self.conf.src_col_name, self.conf.tgt_col_name], inplace=True)
        
        return df
    
    def process_and_save_data(self, df: pd.DataFrame, output_dir: str, name: str):
        logger.info(f"Processing {name} data with no. samples: {len(df)}")
        self.process_data(df=df, output_dir=output_dir, file_name=name+'.json')
    
    def build_data(self):
        logger.info("Reading raw_data")
        df_train: pd.DataFrame = pd.read_csv(self.conf.train_path)
        df_val: pd.DataFrame = pd.read_csv(self.conf.val_path)
        df_test: pd.DataFrame = pd.read_csv(self.conf.test_path)

        logger.info('Filter and clean data')
        df_train = self.filter_and_clean(df_train)
        df_val = self.filter_and_clean(df_val)
        df_test = self.filter_and_clean(df_test)
        
        logger.info(f"Processing training data")
        self.process_and_save_data(df=df_train, output_dir=self.conf.output_dir, name='train')
        logger.info(f"Processing val data")
        self.process_and_save_data(df=df_val, output_dir=self.conf.output_dir, name='val')
        logger.info(f"Processing test data")
        self.process_and_save_data(df=df_test, output_dir=self.conf.output_dir, name='test')    
    
class RedditTIFUDataPreparation(ENDataPreparation):
    def __init__(self, conf: Reddit_TIFU_DataPreparationConf) -> None:
        super().__init__(conf)
        self.setup()
        logger.info('Setup done!')
    
    def setup(self):
        for dir in [self.conf.long_dir, self.conf.short_dir]:
            if not os.path.exists(dir):
                os.system(f"mkdir -p {dir}")
                os.system(f"chmod -R 777 {dir}")
        
        return True
    
    def filter_and_clean(self, posts: List[Dict[str, str]]) -> pd.DataFrame:
        df = pd.DataFrame(data=posts)
        df[self.conf.src_col_name] = df[self.conf.src_col_name].apply(lambda x: self.clean_text(x))
        df[self.conf.tgt_col_name] = df[self.conf.tgt_col_name].apply(lambda x: self.clean_text(x))
        df = df[(df[self.conf.src_col_name]!='') & (df[self.conf.tgt_col_name]!='')]
        df.drop_duplicates(subset= [self.conf.src_col_name, self.conf.tgt_col_name], inplace=True)
        
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

    def build_data(self):
        posts = self.read_json(self.conf.file_path)
        if not all(k in posts[0].keys() for k in ['tldr', 'title', 'selftext', 'selftext_without_tldr']):
            raise KeyError("Missing one of keys: ['tldr', 'title', 'selftext', 'selftext_without_tldr']")
        
        long_posts: List[Dict[str, str]] = []
        short_posts: List[Dict[str, str]] = []
        
        for post in posts:
            if post['selftext_without_tldr']:
                if post['tldr']:
                    long_posts.append({self.conf.src_col_name: post['selftext_without_tldr'], self.conf.tgt_col_name: post['tldr']})
                if post['title']:
                    short_posts.append({self.conf.src_col_name: post['selftext_without_tldr'], self.conf.tgt_col_name: post['title']})
                    
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


class BillSumDataPreparation(ENDataPreparation):
    
    USC_re = re.compile('[Uu]\.*[Ss]\.*[Cc]\.]+')
    PAREN_re = re.compile('\([^(]+\ [^\(]+\)')
    BAD_PUNCT_RE = re.compile(r'([%s])' % re.escape('"#%&\*\+/<=>@[\]^{|}~_'), re.UNICODE)
    BULLET_RE = re.compile('\n[\ \t]*`*\([a-zA-Z0-9]*\)')
    DASH_RE = re.compile('--+')
    WHITESPACE_RE = re.compile('\s+')
    EMPTY_SENT_RE = re.compile('[,\.]\ *[\.,]')
    FIX_START_RE = re.compile('^[^A-Za-z]*')
    FIX_PERIOD = re.compile('\.([A-Za-z])')
    SECTION_HEADER_RE = re.compile('SECTION [0-9]{1,2}\.|\nSEC\.* [0-9]{1,2}\.|Sec\.* [0-9]{1,2}\.')
    FIX_PERIOD = re.compile('\.([A-Za-z])')
    SECTION_HEADER_RE = re.compile('SECTION [0-9]{1,2}\.|\nSEC\.* [0-9]{1,2}\.|Sec\.* [0-9]{1,2}\.')
    
    def __init__(self, conf: BillSum_DataPreparationConf) -> None:
        super().__init__(conf)
        self.setup()
        logger.info('Setup done!')

    def setup(self):
        if not os.path.exists(self.conf.output_dir):
            os.system(f"mkdir -p {self.conf.output_dir}")
            os.system(f"chmod -R 777 {self.conf.output_dir}")
        
        return True
    

    def clean_text(self, text: str):
        """
        Borrowed from the FNDS text processing with additional logic added in.
        Note: we do not take care of token breaking - assume SPACY's tokenizer
        will handle this for us.
        
        Get from https://github.com/FiscalNote/BillSum
        """

        # Indicate section headers, we need them for features
        text = self.SECTION_HEADER_RE.sub('SECTION-HEADER', text)
        # For simplicity later, remove '.' from most common acronym
        text = text.replace("U.S.", "US")
        text = text.replace('SEC.', 'Section')
        text = text.replace('Sec.', 'Section')
        text = self.USC_re.sub('USC', text)

        # Remove parantheticals because they are almost always references to laws 
        # We could add a special tag, but we just remove for now
        # Note we dont get rid of nested parens because that is a complex re
        text = self.PAREN_re.sub('', text)
        text = self.BULLET_RE.sub(' ',text)             # Get rid of enums as bullets or ` as bullets
        text = self.BAD_PUNCT_RE.sub('', text)          # Remove annoying punctuation, that's not relevant
        text = self.DASH_RE.sub( ' ', text)             # Get rid of long sequences of dashes - these are formating
        text = self.WHITESPACE_RE.sub(' ', text)        # removing newlines, tabs, and extra spaces
        text = self.EMPTY_SENT_RE.sub('.', text)        # If we ended up with "empty" sentences - get rid of them.
        text = self.FIX_START_RE.sub('', text)          # Get rid of anything thats not a word from the start of the text
        text = self.FIX_PERIOD.sub(". \g<1>", text)     # Sometimes periods get formatted weird, make sure there is a space between periods and start of sent   
        text = text.replace('&lt;all&gt;', '')          # clean html
        text = text.replace('``', '"')                  # Fix quotes
        text = text.replace('\'\'', '"')

        # Add special punct back in
        text = text.replace('SECTION-HEADER', '<SECTION-HEADER>')
        text = re.sub(r'\s{1,}', ' ', text)

        return text
    
    def process_and_save_data(self, df: pd.DataFrame, output_dir: str, name: str):
        logger.info(f"Processing {name} data with no. samples: {len(df)}")
        self.process_data(df=df, output_dir=output_dir, file_name=name+'.json')
    
    def change_col_name_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [self.conf.src_col_name, self.conf.tgt_col_name]
        df[self.conf.src_col_name] = df[self.conf.src_col_name].apply(lambda x: self.clean_text(x))
        df[self.conf.tgt_col_name] = df[self.conf.tgt_col_name].apply(lambda x: self.clean_text(x))
        df = df[(df[self.conf.src_col_name]!='') & (df[self.conf.tgt_col_name]!='')]
        df.drop_duplicates(subset= [self.conf.src_col_name, self.conf.tgt_col_name], inplace=True)
        
        return df
    
    
    def split_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_train, df_val = train_test_split(df, test_size=self.conf.test_size, shuffle=True, random_state=self.conf.random_state)

        is_train = [True]*len(df_train) + [False]*len(df_val)
        is_val = [False]*len(df_train) + [True]*len(df_val)

        df = pd.concat([df_train, df_val])
        df['is_train'] = is_train
        df['is_val'] = is_val
        
        return df
        
        
    def build_data(self):
        logger.info("Reading raw_data")
        us_train_df: pd.DataFrame = pd.read_json(self.conf.us_train_path, lines=True)[['text', 'summary']]
        us_test_df: pd.DataFrame = pd.read_json(self.conf.us_test_path, lines=True)[['text', 'summary']]
        ca_test_df: pd.DataFrame = pd.read_json(self.conf.ca_test_path, lines=True)[['text', 'summary']]
        
        us_train_df = self.change_col_name_and_clean(df=us_train_df)
        us_train_df = self.split_data(df=us_train_df)
        
        us_test_df = self.change_col_name_and_clean(df=us_test_df)
        ca_test_df = self.change_col_name_and_clean(df=ca_test_df)
        
        logger.info(f"Processing training data")
        self.process_and_save_data(df=us_train_df[us_train_df['is_train']], output_dir=self.conf.output_dir, name='train')
        self.process_and_save_data(df=us_train_df[us_train_df['is_val']], output_dir=self.conf.output_dir, name='val')
        logger.info(f"Processing us_test data")
        self.process_and_save_data(df=us_test_df, output_dir=self.conf.output_dir, name='us_test')
        logger.info(f"Processing ca_test data")
        self.process_and_save_data(df=ca_test_df, output_dir=self.conf.output_dir, name='ca_test')
      

if __name__=='__main__':
    pass