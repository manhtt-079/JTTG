import configparser
from dataclasses import dataclass
from typing import List

DATASET_ARCHIVE_LIST: List[str] = [
    'reddit_tifu',
    'bill_sum',
    'vnds'
]
@dataclass
class DataPreparationConf:
    name: str
    src_col_name: str
    tgt_col_name: str
    top_k: int
    min_nsents: int
    max_nsents: int
    file_path: str
    is_concat: bool
    val_path: str
    test_path: str
    output_path: str
    lang: str
    min_ntokens: int = 5
    max_ntokens: int = 200
    n_processes: int = 2
    no_preprocess: bool = False
    
    def __post_init__(self):
        self.top_k = int(self.top_k)
        self.min_nsents = int(self.min_nsents)
        self.max_nsents = int(self.max_nsents)
        self.is_concat = True if self.is_concat=='True' else False
        if self.val_path=='None':
            self.val_path=None
        if self.test_path=='None':
            self.test_path=None
        

@dataclass
class DatasetConf:
    name: str
    batch_size: int
    max_length: int
    data_dir: str
    train_path: str
    test_path: str
    valid_path: str
    
    def __post_init__(self):
        self.batch_size = int(self.batch_size)
        self.max_length = int(self.max_length)

@dataclass
class ModelConf:
    name: str
    pre_trained_name: str
    sent_rep_tokens: bool
    pooler_dropout: float
    dropout_prop: float
    nhead: int
    ffn_dim: int
    num_layers: int
    n_classes: int

    def __post_init__(self):
        self.sent_rep_tokens = True if self.sent_rep_tokens.lower() == 'true' else False
        self.pooler_dropout = float(self.pooler_dropout)
        self.dropout_prop = float(self.dropout_prop)
        self.nhead = int(self.nhead)
        self.ffn_dim = int(self.ffn_dim)
        self.num_layers = int(self.num_layers)
        self.n_classes = int(self.n_classes)
    
  
@dataclass
class TrainerConf:
    epochs: int
    losses: list
    n_losses: int
    accumulation_steps: int
    n_training_steps: int
    n_warmup_steps: int
    checkpoint: str
    delta: float
    eval_steps: int
    log: str
    lr: float
    no_decay: list
    num_freeze_layers: int
    patience: int
    warmup_prop: float
    weight_decay: float
    best_checkpoint: str
    
    def __post_init__(self):
        self.epochs = int(self.epochs)
        self.losses = self.losses.split(',')
        self.n_losses = int(self.n_losses)
        self.accumulation_steps = int(self.accumulation_steps)
        self.n_training_steps = int(self.n_training_steps)
        self.n_warmup_steps = int(self.n_warmup_steps)
        self.delta = float(self.delta)
        self.eval_steps = int(self.eval_steps)
        self.lr = float(self.lr)
        self.no_decay = self.no_decay.split(',')
        self.num_freeze_layers = int(self.num_freeze_layers)
        self.patience = int(self.patience)
        self.warmup_prop = float(self.warmup_prop)
        self.weight_decay = float(self.weight_decay)
        
        
class Conf:
    def __init__(self, config_file: str, dataset_name: str, model_name: str) -> None:
        
        self.config = self.read_conf(conf_file=config_file)
        self.dataset = self.gen_dataset_conf(dataset_name=dataset_name)
        self.model = self.gen_model_conf(model_name=model_name)
        self.trainer = self.gen_trainer_conf(model_name=model_name)

    @staticmethod
    def read_conf(conf_file) -> configparser.ConfigParser:
        config =  configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        config.read(conf_file)
    
        return config
    
    def __check_exist(self, section_name: str):
        if section_name not in self.config.sections():
            raise ValueError(f"Section: {section_name} not in {self.config.sections()}")
    
    def gen_data_preparation_conf(self, name: str):
        if name not in DATASET_ARCHIVE_LIST:
            raise ValueError(f"Support only dataset in: {DATASET_ARCHIVE_LIST}!")
        
        return DataPreparationConf(name=name, **self.config[name])

    def gen_dataset_conf(self, dataset_name: str) -> DatasetConf:
        self.__check_exist(dataset_name)
        return DatasetConf(name=dataset_name, **self.config[dataset_name])

    def gen_model_conf(self, model_name: str) -> ModelConf:
        self.__check_exist(model_name)
        return ModelConf(name=model_name, **self.config[model_name])
    
    def gen_trainer_conf(self, model_name: str) -> TrainerConf:
        model_name = model_name + '-trainer'
        self.__check_exist(model_name)
        
        return TrainerConf(**self.config[model_name])
        