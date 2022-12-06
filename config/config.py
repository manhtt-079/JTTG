import configparser
from dataclasses import dataclass
from typing import List

DATASET_ARCHIVE_LIST: List[str] = [
    'reddit_tifu',
    'bill_sum',
    'vnds'
]


def read_conf(conf_file):
    config =  configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(conf_file)
    
    return config

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
    tokenizer: str
    batch_size: int
    max_length: int
    train_path: str
    test_path: str
    val_path: str

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
    
@dataclass
class Conf:
    dataset: DatasetConf
    model: ModelConf
    trainer: TrainerConf
    
def gen_datapreparation_conf(conf: configparser.ConfigParser, name: str):
    if name not in DATASET_ARCHIVE_LIST:
        raise ValueError(f"Support only dataset in: {DATASET_ARCHIVE_LIST}!")
    
    return DataPreparationConf(name=name, **conf[name])


def gen_conf(config_file: str) -> Conf:
    config = read_conf(conf_file=config_file)
    
    dataset_conf = DatasetConf(tokenizer=config['dataset']['tokenizer'],
                               batch_size=int(config['dataset']['batch_size']),
                               max_length=int(config['dataset']['max_length']),
                               train_path=config['dataset']['train_path'],
                               val_path=config['dataset']['valid_path'],
                               test_path=config['dataset']['test_path']
                    )

    model_conf = ModelConf(name=config['model']['name'],
                           pre_trained_name=config['model']['pre_trained_name'],
                           sent_rep_tokens=True if config['model']['sent_rep_tokens'].lower() == 'true' else False,
                           pooler_dropout=float(config['model']['pooler_dropout']),
                           dropout_prop=float(config['model']['dropout_prop']),
                           nhead=int(config['model']['nhead']),
                           ffn_dim=int(config['model']['ffn_dim']),
                           num_layers=int(config['model']['num_layers']),
                           n_classes=int(config['model']['n_classes']))

    trainer_conf = TrainerConf(epochs=int(config['trainer']['epochs']),
                               losses=config['trainer']['losses'].split(','),
                               n_losses=int(config['trainer']['n_losses']),
                               accumulation_steps=int(config['trainer']['accumulation_steps']),
                               n_training_steps=int(config['trainer']['n_training_steps']),
                               n_warmup_steps=int(config['trainer']['n_warmup_steps']),
                               checkpoint=config['trainer']['checkpoint'],
                               delta=float(config['trainer']['delta']),
                               eval_steps=int(config['trainer']['eval_steps']),
                               log=config['trainer']['log'],
                               lr=float(config['trainer']['lr']),
                               no_decay=config['trainer']['no_decay'].split(','),
                               num_freeze_layers=int(config['trainer']['num_freeze_layers']),
                               patience=int(config['trainer']['patience']),
                               warmup_prop=float(config['trainer']['warmup_prop']),
                               weight_decay=float(config['trainer']['weight_decay']))

    conf = Conf(dataset=dataset_conf, model=model_conf, trainer=trainer_conf)
    
    return conf, config