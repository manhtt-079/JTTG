import configparser
from dataclasses import dataclass

def read_conf(conf_file):
    return configparser.ConfigParser().read(conf_file)

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
    

def gen_conf(config_file: configparser.ConfigParser) -> Conf:
    config = configparser.ConfigParser()
    config.read(config_file)
    
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
    
    return conf