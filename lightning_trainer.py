import argparse
import os
import gc
import pandas as pd
import evaluate
import torch
import torch.nn as nn
import pytorch_lightning as pl
from loguru import logger
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from typing import Iterator, Tuple, Dict
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything

from modules.datasets.dataset import dataset
from modules.model.utils import AutomaticWeightedLoss
from modules.model.exa_model import ExAb

from config.config import Config

metrics = evaluate.load('rouge')
class ExAbModel(pl.LightningModule):
    def __init__(self, config: Config):
        super(ExAbModel, self).__init__()
        self.config = config
        logger.info(f'Init and loading model_checkpoint: {self.config.model_args.pre_trained_name}')
        self.exab: nn.Module = ExAb(conf=self.config.model_args)        
        logger.info(f"Loading tokenizer: {self.config.model_args.pre_trained_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_args.pre_trained_name)
        if 't5' in self.config.model_args.pre_trained_name:
            self.tokenizer.add_special_tokens({'cls_token': '<s>', 'sep_token': '</s>'})
            self.exab.model.resize_token_embeddings(len(self.tokenizer))
        
        self.prefix = 'layers' if 'bart' in self.config.model_args.pre_trained_name else 'block'
        self.log_dir = self.config.trainer_args.log
        self.checkpoint = self.config.trainer_args.checkpoint
        
        self.bce_loss = self.configure_loss_func(loss_func=self.config.trainer_args.losses[0])
        self.cross_ent_loss = self.configure_loss_func(loss_func=self.config.trainer_args.losses[1])
        self.auto_weighted_loss = AutomaticWeightedLoss(n_losses=self.config.trainer_args.n_losses)
        
        self.num_training_steps = len(self.train_dataloader()) * self.config.trainer_args.max_epochs
        self.num_warmup_steps = int(self.config.trainer_args.warmup_ratio * self.num_training_steps)

    def configure_loss_func(self, loss_func: str) -> nn.Module:
        """Create loss function based on ``loss_func`` name

        Args:
            loss_func (str): the name of loss function

        Returns:
            nn.Module: loss function
        """
        if loss_func not in ['CrossEntropyLoss', 'NLLLoss', 'BCELoss', 'BCEWithLogitsLoss']:
            raise ValueError(f"{loss_func} is not supported. Supported loss functions are: ['CrossEntropyLoss', 'NLLLoss', 'BCELoss', 'BCEWithLogitsLoss']")

        if loss_func == "CrossEntropyLoss":
            return nn.CrossEntropyLoss()
        elif loss_func == "NLLLoss":
            return nn.NLLLoss()
        elif loss_func == "BCELoss":
            return nn.BCELoss()
        elif loss_func == "BCEWithLogitsLoss":
            return nn.BCEWithLogitsLoss()
    
    def make_dir(self, dir_path: str):
        if not os.path.exists(dir_path):
            os.system(f'mkdir -p {dir_path}')
            os.system(f"chmod -R 777 {dir_path}")
    
    def setup(self, stage):
        self.make_dir(dir_path=self.log_dir)
        self.make_dir(dir_path=self.checkpoint)
        
        self.freeze_layers()
        
        return True
    
    def freeze_layers(self) -> None:
        """Freeze some layers of pre-trained model
        """
        freeze_layers = [f'{e}.{self.prefix}.{str(idx)}.' for idx in range(self.config.trainer_args.num_freeze_layers) for e in ['encoder', 'decoder']]
        for name, param in self.exab.model.named_parameters():
            if param.requires_grad and any(freeze_layer in name for freeze_layer in freeze_layers):
                param.requires_grad = False
    
    def get_dataloader(self, data_path: str, shuffle: bool = False):
        return dataset(tokenizer=self.tokenizer, 
                       data_path=data_path,
                       shuffle=shuffle,
                       src_max_length=self.config.dataset_args.src_max_length,
                       tgt_max_length=self.config.dataset_args.tgt_max_length,
                       batch_size=self.config.dataset_args.batch_size,
                       num_workers=self.config.trainer_args.num_workers)
    
    def train_dataloader(self):
        return self.get_dataloader(self.config.dataset_args.train_path, shuffle=True)
    
    def val_dataloader(self):
        return self.get_dataloader(self.config.dataset_args.valid_path)
    
    def test_dataloader(self):
        return self.get_dataloader(self.config.dataset_args.test_path)
    
    def configure_scheduler(self, optimizer: torch.optim.Optimizer):
        scheduler: torch.optim.lr_scheduler.LambdaLR = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                                                       num_warmup_steps=self.num_warmup_steps,
                                                                                       num_training_steps=self.num_training_steps)
        
        return scheduler
    
    def configure_optimizers(self):
        param_optimizer = [[name, param] for name, param in self.exab.model.named_parameters() if param.requires_grad]
        optimizer_grouped_parameters = [
            {'params': [param for name, param in param_optimizer if not any(nd in name for nd in self.config.trainer_args.no_decay)],
             'weight_decay': self.config.trainer_args.weight_decay},
            {'params': [param for name, param in param_optimizer if any(nd in name for nd in self.config.trainer_args.no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer: torch.optim.Optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config.trainer_args.lr)
        scheduler = self.configure_scheduler(optimizer)

        return [optimizer], [scheduler]

    def compute_loss(self, batch: Iterator):
        label = batch['label']

        # outputs[0]: extractive loss, outputs[1]: abstractive loss
        outputs: Tuple[torch.Tensor, torch.Tensor] = self.exab(
            src_ext_input_ids=batch['src_ext_input_ids'],
            src_ext_attention_mask=batch['src_ext_attention_mask'],
            src_abs_input_ids=batch['src_abs_input_ids'],
            src_abs_attention_mask=batch['src_abs_attention_mask'],
            decoder_input_ids=batch['decoder_input_ids'],
            decoder_attention_mask=batch['decoder_attention_mask'],
            sent_rep_ids=batch['sent_rep_ids'],
            sent_rep_mask=batch['sent_rep_mask'])
        
        ext_loss = self.bce_loss(outputs[0], label.float())

        abs_label = batch['decoder_input_ids'].detach().clone()[:, 1:].contiguous().view(-1)
        logits = outputs[1][:, :-1].contiguous().view(-1, outputs[1].size(-1))
        abs_loss = self.cross_ent_loss(logits, abs_label)
        
        loss: torch.Tensor = self.auto_weighted_loss(ext_loss, abs_loss)

        return (loss, abs_loss)
   
    def training_step(self, batch: Iterator, batch_idx):
        loss: torch.Tensor = self.compute_loss(batch=batch)[0]
        
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch: Iterator, batch_idx):
        loss, abs_loss = self.compute_loss(batch=batch)
        d = {'val_loss': loss, 'abs_loss': abs_loss}
        self.log_dict(d)
        
        return d
    
    def validation_epoch_end(self, validation_step_outputs):
        loss = torch.stack([batch['val_loss'] for batch in validation_step_outputs]).mean()
        abs_loss = torch.stack([batch['abs_loss'] for batch in validation_step_outputs]).mean()
        
        return {
            'val_loss': loss,
            'abs_loss': abs_loss
        }
    
    def forward(self, x: Dict[str, torch.Tensor]):
        outputs = self.exab.model.generate(
            input_ids=x['src_abs_input_ids'],
            max_length=self.config.dataset_args.tgt_max_length,
            attention_mask=x['src_abs_attention_mask'],
            num_beams=self.config.trainer_args.num_beams
        )
        outputs = [self.tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in outputs]
        # Replace -100 in the labels as we can't decode them
        labels = torch.where(x['decoder_input_ids'][:, 1:] != -100,
                             x['decoder_input_ids'][:, 1:], 
                             self.tokenizer.pad_token_id)
        
        actuals = [self.tokenizer.decode(lb, clean_up_tokenization_spaces=False, skip_special_tokens=True) for lb in labels]

        return outputs, actuals
        
    def predict_step(self, batch: Iterator, batch_idx):
        outputs, actuals = self.forward(x=batch)
        results = metrics.compute(predictions=outputs, references=actuals)
        return results



# ----- MAIN process -----
def main(config: Config, task_name: str):
    wandb_logger = WandbLogger(project=task_name, save_dir=config.trainer_args.log)
    early_stopping = EarlyStopping(
        monitor=config.trainer_args.monitor,
        mode='min',
        min_delta=config.trainer_args.delta,
        patience=config.trainer_args.patience,
        verbose=True
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.trainer_args.checkpoint,
        filename=config.model_name+'-{epoch}-{step}-{val_loss:.2f}',
        monitor=config.trainer_args.monitor,
        mode='min',
        save_top_k=config.trainer_args.save_top_k,
        save_on_train_epoch_end=config.trainer_args.save_on_train_epoch_end
    )
    
    trainer = Trainer(
        enable_progress_bar=False,
        accelerator=config.trainer_args.accelerator,
        devices=config.trainer_args.devices,
        accumulate_grad_batches=config.trainer_args.accumulate_grad_batches,
        amp_backend=config.trainer_args.amp_backend,
        auto_lr_find=config.trainer_args.auto_lr_find,
        auto_scale_batch_size=config.trainer_args.auto_scale_batch_size,
        auto_select_gpus=config.trainer_args.auto_select_gpus,
        callbacks=[early_stopping, checkpoint_callback],
        default_root_dir=config.trainer_args.checkpoint,
        enable_model_summary=config.trainer_args.enable_model_summary,
        enable_checkpointing=config.trainer_args.enable_checkpointing,
        max_epochs=config.trainer_args.max_epochs,
        logger=wandb_logger,
        log_every_n_steps=config.trainer_args.eval_steps,
        precision=config.trainer_args.precision
    )
    
    gc.collect()
    model = ExAbModel(config)
    trainer.fit(model)
    
    logger.info('----- Testing -----')
    predictions = trainer.predict(dataloaders=model.test_dataloader(), ckpt_path='best')
    rouge_scores = pd.DataFrame(predictions).mean().to_dict()
    logger.info(rouge_scores)


 
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='The random seed for reproducibility.')
    parser.add_argument('--task', type=str, default='task1')
    parser.add_argument('--config_file', type=str, default='./config/config.ini', help='The configuration file.')
    
    args = parser.parse_args()
    seed_everything(args.seed)
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

    kwargs = EXPERIMENT_MAP[args.task]
    config = Config(config_file=args.config_file, **kwargs)
    main(config=config, task_name=args.task)
    