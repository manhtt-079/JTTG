from dataclasses import dataclass
from loguru import logger
from config.config import Config
# from main import Trainer
@dataclass
class Experiment:
    dataset: str
    model: str
    is_long: bool = True
    use_us_test: bool = True
    
class Worker(object):
    
    EXPERIMENT_ARCHIVE_MAP = {
        'task1': {
            'dataset_name': 'reddit_tifu',
            'model_name': 'bart-sum',
            'is_long': True
        },
        'task2': {
            'dataset_name': 'bill_sum',
            'model_name': 'bart-sum',
            'use_us_test': True
        },
        'task3': {
            'dataset_name': 'reddit_tifu',
            'model_name': 't5-sum',
            'is_long': True
        },
        'task4': {
            'dataset_name': 'bill_sum',
            'model_name': 't5-sum',
            'use_us_test': True
        },
        'task5': {
            'dataset_name': 'vnds',
            'model_name': 'bartpho-sum'
        },
        'task6': {
            'dataset_name': 'vnds',
            'model_name': 'vit5-sum'
        }
    }
    def __init__(self, config_file: str) -> None:
        self.config_file = config_file
        self.experiments = []
    
    def run(self):
        logger.info('Starting experiment')
        
        for k, v in self.EXPERIMENT_ARCHIVE_MAP.items():
            logger.info(f'Task: {k}, args: {str(v)}')
            conf = Config(config_file=self.config_file, **v)
            self.experiments.append(conf)
            logger.info(conf.dataset)
            logger.info(conf.model)