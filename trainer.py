import argparse
import traceback
import os
import time
import torch
import torch.nn as nn
import numpy as np
import transformers
import nltk
import evaluate
from tqdm import tqdm
import transformers
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from typing import Iterator, Tuple
from loguru import logger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

from modules.datasets.dataset import dataset
from modules.model.utils import AutomaticWeightedLoss
from modules.model.exa_model import ExAb

from modules.utils import set_seed
from config.config import Config

class ExAbModel(pl.LightningModule):
    def __init__(self, config: Config):
        super(ExAbModel, self).__init__()
        self.config = config
        logger.info(f'Init and loading model_checkpoint: {self.config.model.pre_trained_name}')
        self.exab: nn.Module = ExAb(conf=self.config.model)        
        logger.info(f"Loading {self.config.model.pre_trained_name} tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.pre_trained_name)
        if 't5' in self.config.model.pre_trained_name:
            self.tokenizer.add_special_tokens({'cls_token': '<s>', 'sep_token': '</s>'})
            self.exab.model.resize_token_embeddings(len(self.tokenizer))
        
        self.prefix = 'layers' if 'bart' in self.config.model.pre_trained_name else 'block'
        self.log = self.config.trainer.log
        self.checkpoint = self.config.trainer.checkpoint
        
        self.exab = ExAb(conf=self.config.model)
        self.ext_criterion = nn.BCELoss()
        self.abst_criterion = nn.CrossEntropyLoss()

    def make_dir(self, dir_path: str):
        if not os.path.exists(dir_path):
            os.system(f'mkdir -p {dir_path}')
            os.system(f"chmod -R 777 {self.log}")
    
    def setup(self, stage):
        self.make_dir(dir_path=self.log)
        self.make_dir(dir_path=self.checkpoint)
        
        self.freeze_layers()
        
        return True
    
    def freeze_layers(self) -> None:
        """Freeze some layers of pre-trained model
        """
        freeze_layers = [f'{e}.{self.prefix}.{str(idx)}.' for idx in range(self.num_freeze_layers) for e in ['encoder', 'decoder']]
        for name, param in self.exab.model.named_parameters():
            if param.requires_grad and any(freeze_layer in name for freeze_layer in freeze_layers):
                param.requires_grad = False
    
    def get_dataloader(self, data_path: str, shuffle: bool = False):
        return dataset(tokenizer=self.tokenizer, 
                       data_path=data_path,
                       shuffle=shuffle,
                       src_max_length=self.config.dataset.src_max_length,
                       tgt_max_length=self.config.dataset.tgt_max_length,
                       batch_size=self.config.dataset.batch_size)
    
    def train_dataloader(self):
        return self.get_dataloader(self.config.dataset.train_path)
    
    def val_dataloader(self):
        return self.get_dataloader(self.config.dataset.valid_path)
    
    def test_dataloader(self):
        return self.get_dataloader(self.config.dataset.test_path)
    
    def configure_scheduler(self, optimizer: torch.optim.Optimizer):
        scheduler: torch.optim.lr_scheduler.LambdaLR = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                                                       num_warmup_steps=self.num_warmup_steps,
                                                                                       num_training_steps=self.num_training_steps)
        
        return scheduler
    
    def configure_optimizers(self):
        param_optimizer = [[name, param] for name, param in self.exab.model.named_parameters() if param.requires_grad]
        optimizer_grouped_parameters = [
            {'params': [param for name, param in param_optimizer if not any(nd in name for nd in self.no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [param for name, param in param_optimizer if any(nd in name for nd in self.no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer: torch.optim.Optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)
        scheduler = self.configure_scheduler(optimizer)

        return [optimizer], [scheduler]

    def compute_loss(self, batch):
        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, sent_rep_ids, sent_rep_mask, label = batch
        outputs: Tuple[torch.Tensor, torch.Tensor] = self.exab(input_ids=input_ids,
                                                                attention_mask=attention_mask,
                                                                decoder_input_ids=decoder_input_ids,
                                                                decoder_attention_mask=decoder_attention_mask,
                                                                sent_rep_ids=sent_rep_ids,
                                                                sent_rep_mask=sent_rep_mask)
        ext_loss = self.ext_criterion(outputs[0], label.float())

        abst_label = decoder_input_ids.detach().clone()[:, 1:].contiguous().view(-1)
        logits = outputs[1][:, :-1].contiguous().view(-1, outputs[1].size(-1))
        abst_loss = self.abst_criterion(logits, abst_label)
        
        loss: torch.Tensor = self.auto_weighted_loss(ext_loss, abst_loss)

        return (loss, abst_loss)
    
    def training_step(self, batch, batch_idx):
        loss: torch.Tensor = self.compute_loss(batch=batch)[0]
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, abst_loss = self.compute_loss(batch=batch)
        
        d = {'val_loss': loss, 'abst_loss': abst_loss}
        self.log_dict(d, prog_bar=True)        
        return d
    
    def validation_epoch_end(self, validation_step_outputs):
        loss = torch.stack([batch['val_loss'] for batch in validation_step_outputs]).mean()
        abst_loss = torch.stack([batch['abst_loss'] for batch in validation_step_outputs]).mean()
        
        return {
            'val_loss': loss,
            'abst_loss': abst_loss
        }
    
    def forward(self, x):
        outputs = self.exab.model.generate(
            input_ids=x['input_ids'],
            max_length=self.config.dataset.tgt_max_length,
            attention_mask=x['attention_mask'],
            num_beams=self.config.trainer.num_beams
        )
        outputs = [self.tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in outputs]
        # Replace -100 in the labels as we can't decode them
        labels = np.where(batch['decoder_input_ids'][:, 1:] != -100, batch['decoder_input_ids'][:, 1:], self.tokenizer.pad_token_id)
        actuals = [self.tokenizer.decode(lb, clean_up_tokenization_spaces=False, skip_special_tokens=True) for lb in labels]
        
    def predict_step(self, batch, batch_idx):
        labels, actuals = self.exab(x=batch)
        
        return labels, actuals

def main(task_name: str, config_file: str):
    EXPERIMENT_MAP = {
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
    kwargs = EXPERIMENT_MAP[task_name]
    config = Config(config_file=config_file, **kwargs)
    seed_everything(42)
    wandb_logger = WandbLogger(project=task_name)
    early_stopping = EarlyStopping(monitor='val_loss',
                                   mode='min',
                                   min_delta=config.trainer.delta,
                                   patience=config.trainer.patience,
                                   verbose=True)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=config.trainer.checkpoint,
        filename=config.model_name + '-{epoch}-{step}-{val_loss:.2f}',
        mode='min',
        save_top_k=3,
        save_on_train_epoch_end=True
    )
    trainer = Trainer(
        accelerator='gpu',
        accumulate_grad_batches=config.trainer.accumulation_steps,
        callbacks=[early_stopping, checkpoint_callback],
        amp_backend='native',
        auto_lr_find=False,
        auto_scale_batch_size=False,
        auto_select_gpus=False,
        enable_checkpointing=True,
        default_root_dir=config.trainer.checkpoint,
        logger=wandb_logger,
        log_every_n_steps=config.trainer.eval_steps,
        precision=16,
        max_epochs=config.trainer.epochs
    )
    config = Config('./config/config.ini', 'bill_sum', 'bart-sum')
    model = ExAbModel(config)
    trainer.fit(model)


if __name__=='__main__':
    main(config_file="./config/config.ini", task_name='task2')
