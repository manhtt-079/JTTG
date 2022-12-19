import os
import configparser
from dataclasses import dataclass

class DatasetBaseConf(object):
    def __init__(self, 
                 config: configparser.ConfigParser, 
                 sec_name: str,
                 is_long: bool = True,
                 use_us_test: bool = True) -> None:
        
        self.config = config
        self.sec_name = sec_name
        self.batch_size = int(self.config[self.sec_name]['batch_size'])
        self.src_max_length = int(self.config[self.sec_name]['src_max_length'])
        self.tgt_max_length = int(self.config[self.sec_name]['tgt_max_length'])
        self.data_dir = self.config[self.sec_name].get('data_dir', None)
        self.is_long = is_long
        self.use_us_test = use_us_test

    def __repr__(self) -> str:
        return str(self.__dict__)
    
class VnDsDatasetConf(DatasetBaseConf):
    def __init__(self, config: configparser.ConfigParser, sec_name: str = 'vnds_dataset', is_long: bool = True, use_us_test: bool = True) -> None:
        super().__init__(config, sec_name, is_long, use_us_test)
        
        self.train_path = self.config[self.sec_name]['train_path']
        self.valid_path = self.config[self.sec_name]['valid_path']
        self.test_path = self.config[self.sec_name]['test_path']

class BillSumDatasetConf(DatasetBaseConf):
    def __init__(self, config: configparser.ConfigParser, sec_name: str = 'bill_sum_dataset', is_long: bool = True, use_us_test: bool = True) -> None:
        super().__init__(config, sec_name, is_long, use_us_test)
        self.train_path = self.config[self.sec_name]['train_path']
        self.valid_path = self.config[self.sec_name]['valid_path']
        self.test_path = self.config[self.sec_name]['us_test_path'] if self.use_us_test else self.config[self.sec_name]['ca_test_path']
        
        
class RedditTifuDatasetConf(DatasetBaseConf):
    def __init__(self, config: configparser.ConfigParser, sec_name: str = 'reddit_tifu_dataset', is_long: bool = True, use_us_test: bool = True) -> None:
        super().__init__(config, sec_name, is_long, use_us_test)
        
        self.data_dir = self.config[self.sec_name]['long_dir'] if self.is_long else self.config[self.sec_name]['short_dir']
        self.train_path = self.join_path(self.data_dir, self.config[self.sec_name]['train_path'])
        self.valid_path = self.join_path(self.data_dir, self.config[self.sec_name]['valid_path'])
        self.test_path = self.join_path(self.data_dir, self.config[self.sec_name]['test_path'])
    
    @staticmethod
    def join_path(p1, p2):
        return os.path.join(p1, p2)
    
    

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
    
class TrainerBase(object):
    def __init__(self, config: configparser.ConfigParser) -> None:
        self.config = config
        
        self.delta = float(self.config['trainer-base']['delta'])
        self.eval_steps = int(self.config['trainer-base']['eval_steps'])
        self.losses = self.config['trainer-base']['losses'].split(',')
        self.n_losses = int(self.config['trainer-base']['n_losses'])
        self.no_decay = self.config['trainer-base']['no_decay'].split(',')
        self.patience = int(self.config['trainer-base']['patience'])
        self.warmup_prop = float(self.config['trainer-base']['warmup_prop'])
        self.weight_decay = float(self.config['trainer-base']['weight_decay'])
        self.num_beams = int(self.config['trainer-base']['num_beams'])
    
    def __repr__(self) -> str:
        return str(self.__dict__)

class ExAbDatasetTrainer(TrainerBase):
    def __init__(self, config: configparser.ConfigParser, sec_name: str) -> None:
        super().__init__(config)
        self.sec_name = sec_name

        self.accumulation_steps = int(self.config[self.sec_name]['accumulation_steps'])
        self.checkpoint = self.config[self.sec_name]['checkpoint']
        self.epochs = int(self.config[self.sec_name]['epochs'])
        self.log = self.config[self.sec_name]['log']
        self.lr = float(self.config[self.sec_name]['lr'])
        self.n_training_steps = int(self.config[self.sec_name]['n_training_steps'])
        self.n_warmup_steps = int(self.config[self.sec_name]['n_warmup_steps'])
        self.num_freeze_layers = int(self.config[self.sec_name]['num_freeze_layers'])
        self.best_checkpoint = self.config[self.sec_name]['best_checkpoint']
        
        
class T5DatasetTrainer(TrainerBase):
    def __init__(self, config: configparser.ConfigParser, dataset_name: str) -> None:
        super().__init__(config)
        self.sec_name = 't5-sum-trainer' + dataset_name

        self.accumulation_steps = int(self.config[self.sec_name]['accumulation_steps'])
        self.checkpoint = self.config[self.sec_name]['checkpoint']
        self.epochs = int(self.config[self.sec_name]['epochs'])
        self.log = self.config[self.sec_name]['log']
        self.lr = float(self.config[self.sec_name]['lr'])
        self.n_training_steps = int(self.config[self.sec_name]['n_training_steps'])
        self.n_warmup_steps = int(self.config[self.sec_name]['n_warmup_steps'])
        self.num_freeze_layers = int(self.config[self.sec_name]['num_freeze_layers'])
        self.best_checkpoint = self.config[self.sec_name]['best_checkpoint']
        
class BartPhoDatasetTrainer(TrainerBase):
    def __init__(self, config: configparser.ConfigParser, dataset_name: str) -> None:
        super().__init__(config)
        self.sec_name = 'bartpho-sum-trainer' + dataset_name

        self.accumulation_steps = int(self.config[self.sec_name]['accumulation_steps'])
        self.checkpoint = self.config[self.sec_name]['checkpoint']
        self.epochs = int(self.config[self.sec_name]['epochs'])
        self.log = self.config[self.sec_name]['log']
        self.lr = float(self.config[self.sec_name]['lr'])
        self.n_training_steps = int(self.config[self.sec_name]['n_training_steps'])
        self.n_warmup_steps = int(self.config[self.sec_name]['n_warmup_steps'])
        self.num_freeze_layers = int(self.config[self.sec_name]['num_freeze_layers'])
        self.best_checkpoint = self.config[self.sec_name]['best_checkpoint']
        
class ViT5DatasetTrainer(TrainerBase):
    def __init__(self, config: configparser.ConfigParser, dataset_name: str) -> None:
        super().__init__(config)
        self.sec_name = 'vit5-sum-trainer' + dataset_name

        self.accumulation_steps = int(self.config[self.sec_name]['accumulation_steps'])
        self.checkpoint = self.config[self.sec_name]['checkpoint']
        self.epochs = int(self.config[self.sec_name]['epochs'])
        self.log = self.config[self.sec_name]['log']
        self.lr = float(self.config[self.sec_name]['lr'])
        self.n_training_steps = int(self.config[self.sec_name]['n_training_steps'])
        self.n_warmup_steps = int(self.config[self.sec_name]['n_warmup_steps'])
        self.num_freeze_layers = int(self.config[self.sec_name]['num_freeze_layers'])
        self.best_checkpoint = self.config[self.sec_name]['best_checkpoint']

class Config(object):
    
    DATASET_CONF_ARCHIVE_MAP = {
        'reddit_tifu': RedditTifuDatasetConf,
        'bill_sum': BillSumDatasetConf,
        'vnds':  VnDsDatasetConf
    }
    
    MODEL_CONF_ARCHIVE_LIST = {
        'bart-sum',
        't5-sum',
        'vit5-sum',
        'bartpho-sum'
    }
    
    def __init__(self, 
                 config_file: str,
                 dataset_name: str,
                 model_name: str,
                 is_long: bool = None,
                 use_us_test: bool = None
                 ) -> None:
        self.config_file = config_file
        self.config = self.read_conf(conf_file=config_file)
        
        if dataset_name not in self.DATASET_CONF_ARCHIVE_MAP:
            raise ValueError(f"Dataset must be in {self.DATASET_CONF_ARCHIVE_MAP.keys()}")
        if model_name not in self.MODEL_CONF_ARCHIVE_LIST:
            raise ValueError(f"Model name must be in {self.MODEL_CONF_ARCHIVE_LIST}")
        
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.trainer_sec = self.model_name + '-trainer-' + self.dataset_name
        self.is_long = is_long
        self.use_us_test = use_us_test
        
    
    @staticmethod
    def read_conf(conf_file) -> configparser.ConfigParser:
        config =  configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        config.read(conf_file)
    
        return config
    
    @property
    def trainer(self):
        return ExAbDatasetTrainer(config=self.config, sec_name=self.trainer_sec)
        
    @property
    def model(self):
        return ModelConf(name=self.model_name, pre_trained_name=self.config[self.model_name]['pre_trained_name'], **self.config['model-base'])
    
    @property
    def dataset(self):
        return self.DATASET_CONF_ARCHIVE_MAP[self.dataset_name](config=self.config, is_long=self.is_long, use_us_test=self.use_us_test)
        
if __name__=='__main__':
    pass