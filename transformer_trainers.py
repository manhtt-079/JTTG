import argparse
import os
import torch
import torch.nn as nn
from typing import List, Dict, Any, Union, Tuple, Callable, Optional
from loguru import logger
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DefaultDataCollator, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from collections.abc import Mapping
import numpy as np

from modules.datasets.dataset import dataset, ExAbDataset
from modules.model.utils import AutomaticWeightedLoss
from modules.model.exa_model import ExAb
from modules.utils import set_seed
from config.config import Config

def default_data_collator(features, return_tensors="pt"):
    if return_tensors == "pt":
        return torch_default_data_collator(features)

def torch_default_data_collator(features):
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch

class ExAbDataCollator(DefaultDataCollator):
    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        if return_tensors is None:
            return_tensors = self.return_tensors
        return default_data_collator(features=features, return_tensors=return_tensors)


class ExAbTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=...,
        preprocess_logits_for_metrics=None
    ):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
    
    def configure_optimizer(self) -> torch.optim.Optimizer:
        param_optimizer = [[name, param] for name, param in self.exab.model.named_parameters() if param.requires_grad]
        optimizer_grouped_parameters = [
            {'params': [param for name, param in param_optimizer if not any(nd in name for nd in self.trainer_args.no_decay)],
            'weight_decay': self.trainer_args.weight_decay},
            {'params': [param for name, param in param_optimizer if any(nd in name for nd in self.trainer_args.no_decay)],
            'weight_decay': 0.0}
        ]

        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.trainer_args.lr)
        
        return self.optimizer
    
    def create_optimizer(self):
        return super().create_optimizer()
    
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        return super().create_scheduler(num_training_steps, optimizer)
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        return super().create_optimizer_and_scheduler(num_training_steps)
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        return super().training_step(model, inputs)

    def prediction_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool, ignore_keys: Optional[List[str]] = None) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    def evaluate(self, eval_dataset: Optional[Dataset] = None, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "eval", **gen_kwargs) -> Dict[str, float]:
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix, **gen_kwargs)
    
    def predict(self, test_dataset: Dataset, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "test", **gen_kwargs) -> PredictionOutput:
        return super().predict(test_dataset, ignore_keys, metric_key_prefix, **gen_kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        return super().compute_loss(model, inputs, return_outputs)    


class Worker(object):
    def __init__(self, conf: Config, device: torch.device):
        
        self.config_file = conf.config_file
        self.conf = conf
        self.config_parser = self.conf.config
        self.device = device     
        
        logger.info(f'Loading model: {self.conf.model.pre_trained_name}')
        self.model: nn.Module = ExAb(conf=self.conf.model)
        self.model.to(self.device)
        
        logger.info(f"Loading tokenizer: {self.conf.model.pre_trained_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.conf.model.pre_trained_name)
        if 't5' in self.conf.model.pre_trained_name:
            self.tokenizer.add_special_tokens({'cls_token': '<s>', 'sep_token': '</s>'})
            self.model.model.resize_token_embeddings(len(self.tokenizer))

        self.gradient_accumulation_steps = self.conf.trainer.accumulation_steps
        self.epochs = self.conf.trainer.epochs
        self.log = self.conf.trainer.log
        self.log_loss = os.path.join(self.log, 'loss.txt')
        self.lr = self.conf.trainer.lr
        self.num_freeze_layers = self.conf.trainer.num_freeze_layers
        self.weight_decay = self.conf.trainer.weight_decay
        self.no_decay = self.conf.trainer.no_decay
        self.patience = self.conf.trainer.patience
        self.delta = self.conf.trainer.delta
        self.eval_steps = self.conf.trainer.eval_steps

        logger.info("Get dataloader")
        self.train_dataset = self.get_dataloader(data_path=self.conf.dataset.train_path, shuffle=True)
        self.val_dataset = self.get_dataloader(data_path=self.conf.dataset.valid_path)
        self.test_dataloader = None

        self.num_training_steps = len(self.train_dataset) * self.conf.trainer.epochs
        self.num_warmup_steps = int(self.conf.trainer.warmup_prop * self.num_training_steps)

        self.extractive_loss = self.gen_loss_funct(loss_funct=self.conf.trainer.losses[0])
        self.abstractive_loss = self.gen_loss_funct(loss_funct=self.conf.trainer.losses[1])
        
        self.auto_weighted_loss = AutomaticWeightedLoss(n_losses=self.conf.trainer.n_losses)
        self.data_collator = ExAbDataCollator(return_tensors='pt')
        # self.optimizer, self.scheduler = self.create_optimizer_scheduler()
        
        self.checkpoint = self.conf.trainer.checkpoint
        self.do_train = True
        self.do_eval = False
        self.evaluation_strategy = 'epoch'
        self.prediction_loss_only = True
        # self.best_checkpoint = self.conf.trainer.best_checkpoint
        # self.config_parser[self.conf.trainer.sec_name]['n_training_steps'] = str(self.num_training_steps)
        # self.config_parser[self.conf.trainer.sec_name]['n_warmup_steps'] = str(self.num_warmup_steps)
        
        # self.save_config()
        # self.setup()
    @staticmethod
    def gen_loss_funct(loss_funct: str):

        if loss_funct not in ['CrossEntropyLoss', 'NLLLoss', 'BCELoss', 'BCEWithLogitsLoss']:
            raise ValueError(f"{loss_funct} not in current loss functions are supported: ['CrossEntropyLoss', 'NLLLoss', 'BCELoss', 'BCEWithLogitsLoss']")

        if loss_funct == "CrossEntropyLoss":
            return nn.CrossEntropyLoss()
        elif loss_funct == "NLLLoss":
            return nn.NLLLoss()
        elif loss_funct == "BCELoss":
            return nn.BCELoss()
        elif loss_funct == "BCEWithLogitsLoss":
            return nn.BCEWithLogitsLoss()
    
    def get_dataset(self, file_path: str):
        return ExAbDataset(tokenizer=self.tokenizer,
                           data_path=file_path,
                           src_max_length=self.conf.dataset.src_max_length,
                           tgt_max_length=self.conf.dataset.tgt_max_length)
    
    @property
    def training_args(self) -> Seq2SeqTrainingArguments:
        return Seq2SeqTrainingArguments(output_dir=self.checkpoint,
                                        do_train=self.do_train,
                                        do_eval=self.do_eval,
                                        evaluation_strategy=self.evaluation_strategy,
                                        prediction_loss_only=self.prediction_loss_only,
                                        per_device_train_batch_size=self.conf.dataset.batch_size,
                                        per_device_eval_batch_size=self.conf.dataset.batch_size,
                                        learning_rate=self.lr,
                                        weight_decay=0.015,
                                        num_train_epochs=self.epochs,
                                        warmup_ratio=self.conf.trainer.warmup_prop,
                                        logging_dir=self.log,
                                        logging_strategy='epoch',
                                        save_strategy='epoch',
                                        save_total_limit=3,
                                        seed=42,
                                        data_seed=42,
                                        gradient_accumulation_steps=self.gradient_accumulation_steps,
                                        group_by_length=True,
                                        fp16=True,
                                        load_best_model_at_end=True,
                                        report_to='none'
                                        )

    @property
    def trainer(self) -> ExAbTrainer:
        return ExAbTrainer(model=self.model,
                           args=self.training_args,
                           train_dataset=self.train_dataset,
                           eval_dataset=self.val_dataset,
                           data_collator=self.data_collator)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='./config/config.ini', help="Path to the config file")
    parser.add_argument('--task_name', type=str, default='task2')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    wk = Worker(config_file=args.config_file, device=device)
    wk.trainer.train()
    
# TODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO