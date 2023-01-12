from dataclasses import dataclass
import argparse
import torch
from loguru import logger
from config.config import Config
from pytorch_trainer import Trainer
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
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def run(self, task_name: str):
        kwargs = self.EXPERIMENT_ARCHIVE_MAP[task_name]
        conf = Config(config_file=self.config_file, **kwargs)
        
        trainer = Trainer(conf=conf, device=self.device)
        trainer.fit()
        
    def test(self, task_name: str):
        kwargs = self.EXPERIMENT_ARCHIVE_MAP[task_name]
        config = Config(config_file=self.config_file, **kwargs)
        
        trainer = Trainer(config=config, device=self.device)
        logger.info(f"Loading the best model from {config.config[config.trainer_args.sec_name]['best_checkpoint']}...")
        current_epoch = trainer.load_model(path=config.config[config.trainer_args.sec_name]['best_checkpoint'])
        logger.info(f"With epoch: {current_epoch}")
        trainer.test()
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='./config/config.ini', help="Path to the config file")
    parser.add_argument('--task_name', type=str, default='task2')


    args = parser.parse_args()
    task = Worker(config_file=args.config_file)
    task.run(task_name=args.task_name)

if __name__=='__main__':
    main()