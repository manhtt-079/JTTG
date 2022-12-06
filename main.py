import argparse
import os
import time
import torch
import torch.nn as nn
import numpy as np
import transformers
import yaml
import configparser
from yaml import Loader
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from typing import Iterator, Tuple
from loguru import logger

from modules.datasets.data import dataset
from modules.model.utils import AutomaticWeightedLoss, EarlyStopping
from modules.model.bart_sum import BartSum
from modules.model.t5_sum import T5Sum

from modules.utils import set_seed
from config.config import gen_conf

MODEL_ARCHIVE_MAP = {
    'bart-sum': BartSum,
    't5-sum': T5Sum,
    'vit5-sum': None
}

class Trainer:
    def __init__(self, config_file: str, device: torch.device):
        
        self.config_file = config_file
        self.conf, self.config = gen_conf(config_file=config_file)
        self.device = device
        
        if self.conf.model.name not in MODEL_ARCHIVE_MAP:
            raise ValueError(f"Model name must be in: {MODEL_ARCHIVE_MAP.keys()}")        
        
        self.model: nn.Module = MODEL_ARCHIVE_MAP[self.conf.model.name](self.conf.model)
        self.model.to(self.device)
        
        logger.info(f"Loading {self.conf.dataset.tokenizer} tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.conf.dataset.tokenizer)

        self.num_freeze_layers = self.conf.trainer.num_freeze_layers
        self.epochs = self.conf.trainer.epochs
        self.lr = self.conf.trainer.lr
        self.accumulation_steps = self.conf.trainer.accumulation_steps
        self.weight_decay = self.conf.trainer.weight_decay
        self.no_decay = self.conf.trainer.no_decay
        self.patience = self.conf.trainer.patience
        self.delta = self.conf.trainer.delta
        self.eval_steps = self.conf.trainer.eval_steps

        logger.info("Get dataloader")
        self.train_dataloader = self.get_dataloader(data_path=self.conf.dataset.train_path, shuffle=True)
        self.val_dataloader = self.get_dataloader(data_path=self.conf.dataset.val_path, shuffle=False)
        self.test_dataloader = None

        self.num_training_steps = len(self.train_dataloader) * self.conf.trainer.epochs
        self.num_warmup_steps = int(self.conf.trainer.warmup_prop * self.num_training_steps)

        self.ex_criterion = self.gen_criterion(criterion=self.conf.trainer.losses[0])
        self.ab_criterion = self.gen_criterion(criterion=self.conf.trainer.losses[1])
        
        self.auto_weighted_loss = AutomaticWeightedLoss(n_losses=self.conf.trainer.n_losses)

        self.optimizer, self.scheduler = self.create_optimizer_scheduler()

        self.checkpoint = self.conf.trainer.checkpoint
        self.log = self.conf.trainer.log
        self.setup()

    def setup(self):
        if not os.path.exists('./log'):
            os.system(f"mkdir ./log")
            os.system(f"chmod -R 777 ./log")
        
        if not os.path.exists('./checkpoint'):
            os.system(f"mkdir ./checkpoint")
            os.system(f"chmod -R 777 ./checkpoint")
        
        return True
    
    def get_dataloader(self, data_path: str, shuffle: bool):
        
        return dataset(tokenizer=self.tokenizer, 
                       data_path=data_path,
                       max_len=self.conf.dataset.max_length,
                       batch_size=self.conf.dataset.batch_size,
                       shuffle=shuffle)
    
    def save_config(self):
        with open(self.config_file, 'w') as f:
            self.config.write(f)
            
        return True
    
    @staticmethod
    def gen_criterion(criterion: str):

        if criterion not in ['CrossEntropyLoss', 'NLLLoss', 'BCELoss', 'BCEWithLogitsLoss']:
            raise ValueError(f"{criterion} not in current loss functions are supported: ['CrossEntropyLoss', 'NLLLoss', 'BCELoss', 'BCEWithLogitsLoss']")

        if criterion == "CrossEntropyLoss":
            return nn.CrossEntropyLoss()
        elif criterion == "NLLLoss":
            return nn.NLLLoss()
        elif criterion == "BCELoss":
            return nn.BCELoss()
        elif criterion == "BCEWithLogitsLoss":
            return nn.BCEWithLogitsLoss()

    def create_optimizer(self):
        
        freeze_layers = [f'{e}.layers.{str(idx)}' for idx in range(self.num_freeze_layers) for e in ['encoder', 'decoder']]
        for name, param in self.model.named_parameters():
            if param.requires_grad and any(freeze_layer in name for freeze_layer in freeze_layers):
                param.requires_grad = False

        param_optimizer = [[name, param] for name,
                        param in self.model.named_parameters() if param.requires_grad]
        optimizer_grouped_parameters = [
            {'params': [param for name, param in param_optimizer if not any(nd in name for nd in self.no_decay)],
            'weight_decay': self.weight_decay},
            {'params': [param for name, param in param_optimizer if any(nd in name for nd in self.no_decay)],
            'weight_decay': 0.0}
        ]

        optimizer: torch.optim.Optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)
        
        return optimizer
    
    def create_scheduler(self, optimizer: torch.optim.Optimizer):
        scheduler: torch.optim.lr_scheduler.LambdaLR = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                                                       num_warmup_steps=self.num_warmup_steps,
                                                                                       num_training_steps=self.num_training_steps)
        
        return scheduler

    def create_optimizer_scheduler(self):
        optimizer: torch.optim.Optimizer = self.create_optimizer()
        scheduler: torch.optim.lr_scheduler.LambdaLR = self.create_scheduler(optimizer=optimizer)
        
        return optimizer, scheduler

    def to_input(self, batch: Iterator):
        return tuple(t.to(self.device) for t in [batch[k] for k in ['src_ids', 'src_mask', 'tgt_ids', 'tgt_mask', 'sent_rep_token_ids', 'sent_rep_mask', 'label']])
    
    def step(self, batch: Iterator):
        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, sent_rep_ids, sent_rep_mask, label = self.to_input(batch)

        outputs: Tuple[torch.Tensor, torch.Tensor] = self.model(input_ids=input_ids,
                                                                attention_mask=attention_mask,
                                                                decoder_input_ids=decoder_input_ids,
                                                                decoder_attention_mask=decoder_attention_mask,
                                                                sent_rep_ids=sent_rep_ids,
                                                                sent_rep_mask=sent_rep_mask)
        
        ex_loss = self.ex_criterion(outputs[0], label.float())
        
        ab_label = decoder_input_ids.detach().clone()[:, 1:].contiguous().view(-1)
        logits = outputs[1][:, :-1].contiguous().view(-1, outputs[1].size(-1))
        ab_loss = self.ab_criterion(logits, ab_label)
        
        loss: torch.Tensor = self.auto_weighted_loss(ex_loss, ab_loss)
        
        loss = loss/self.accumulation_steps
        
        loss.backward()
        
        return loss.item()*self.accumulation_steps

    def validate(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            running_loss: float = 0.0
            for batch in dataloader:
                input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, sent_rep_ids, sent_rep_mask, label = self.to_input(batch)
                outputs: Tuple[torch.Tensor, torch.Tensor] = self.model(input_ids=input_ids,
                                                                        attention_mask=attention_mask,
                                                                        decoder_input_ids=decoder_input_ids,
                                                                        decoder_attention_mask=decoder_attention_mask,
                                                                        sent_rep_ids=sent_rep_ids,
                                                                        sent_rep_mask=sent_rep_mask)
                ex_loss = self.ex_criterion(outputs[0], label.float())
        
                ab_label = decoder_input_ids.detach().clone()[:, 1:].contiguous().view(-1)
                logits = outputs[1][:, :-1].contiguous().view(-1, outputs[1].size(-1))
                ab_loss = self.ab_criterion(logits, ab_label)
                
                loss: torch.Tensor = self.auto_weighted_loss(ex_loss, ab_loss)
                running_loss += loss.item()

            loss: float = running_loss/(len(dataloader))

        return loss

    def train(self):
        early_stopping = EarlyStopping(patience=self.patience, delta=self.delta)
        train_losses, val_losses = [], []
        for epoch in range(self.epochs):
            start: float = time.time()
            self.model.train()

            running_loss: float = 0.0
            n_iters = len(self.train_dataloader)
            self.optimizer.zero_grad()
            for idx, batch in enumerate(self.train_dataloader):
                
                loss: float = self.step(batch=batch)
                running_loss += loss
                if (idx+1) % self.accumulation_steps == 0 or (idx+1) == n_iters:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                if (idx+1) % self.eval_steps == 0 or idx == 0:
                    print("Epoch: {}/{} - iter: {}/{} - train_loss: {}".format(epoch+1, self.epochs, idx+1, n_iters, running_loss/(idx+1)))
            else:
                train_loss = running_loss/n_iters
                print("Epochs: {}/{} - iter: {}/{} - train_loss: {}\n".format(epoch+1, self.epochs, idx+1, n_iters, train_loss))

                print("Evaluating...")
                val_loss = self.validate(dataloader=self.val_dataloader)
                print("     Val loss: {}\n".format(val_loss))

                train_losses.append(train_losses)
                val_losses.append(val_losses)

                early_stopping(val_loss=val_loss)
                if early_stopping.is_save:
                    ckp_path: str = os.path.join(self.checkpoint, 'ckp'+str(epoch+1)+'.pt')
                    logger.info(f"Saving model to: {ckp_path}")
                    self.config['trainer']['best_checkpoint'] = ckp_path
                    self.save_config()
                    self.save_model(current_epoch=epoch+1, path=ckp_path)
                if early_stopping.early_stop:
                    logger.info(f"Early stopping. Saving log loss to: {os.path.join(self.log, 'loss.txt')}")
                    break
            logger.info(f"Total time per epoch: {time.time()-start} seconds")
        train_losses, val_losses = np.array(train_loss).reshape(-1,1), np.array(val_loss).reshape(-1,1)
        np.savetxt(os.path.join(self.log, 'loss.txt'), np.hstack((train_losses, val_losses)), delimiter='#')

    def test(self):
        del self.train_dataloader
        del self.val_dataloader
        self.test_dataloader = self.get_dataloader(data_path=self.conf.dataset.test_path, shuffle=False)
        logger.info("Testing...")
        loss = self.validate(dataloader=self.test_dataloader)
        logger.info("     Test loss: {}\n".format(loss))

        return loss

    def fit(self):
        logger.info("Start training...")
        self.train()
        logger.info("Finish training.\n")
        logger.info("Start testing...")
        logger.info(f"Loading the best model from {self.config['trainer']['best_checkpoint']}...")
        current_epoch = self.load_model(path=self.config['trainer']['best_checkpoint'])
        logger.info(f"With epoch: {current_epoch}")
        self.test()
        logger.info("Finish testing.")

    def load_model(self, path: str):
        ckp = torch.load(path)
        self.model.load_state_dict(ckp['model_state_dict'])
        self.optimizer.load_state_dict(ckp['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckp['scheduler_state_dict'])
        current_epoch = ckp['current_epoch']

        return current_epoch

    def save_model(self, current_epoch: int, path: str):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "current_epoch": current_epoch
        }, path)

    def resume(self):
        current_epoch = self.load_model(path=self.config['trainer']['resume']['path'])
        logger.info(f"Continuing training model from epoch {current_epoch}...")
        early_stopping = EarlyStopping(patience=self.patience, delta=self.delta)
        train_losses, val_losses = [], []
        for epoch in range(current_epoch, current_epoch + self.config['trainer']['resume']['epochs']):
            start = time.time()
            self.model.train()

            running_loss = 0.0
            n_iters = len(self.train_dataloader)
            for idx, batch in enumerate(self.train_dataloader):
                
                loss: float = self.step(batch=batch)
                running_loss += loss
                if (idx+1) % self.accumulation_steps == 0 or (idx+1) == n_iters:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                if idx+1 % self.eval_steps == 0 or idx == 0:
                    print("Epoch: {}/{} - iter: {}/{} - train_loss: {}".format(epoch+1, self.epochs, idx+1, n_iters, running_loss/(idx+1)))
            else:
                train_loss = running_loss/n_iters
                print("Epochs: {}/{} - iter: {}/{} - train_loss: {}\n".format(epoch+1, self.epochs, idx+1, n_iters, train_loss))

                print("Evaluating...")
                val_loss, acc, f1, auc = self.validate(dataloader=self.val_dataloader)
                print("     Val loss: {} - accuracy: {} - f1-score: {} - auc: {}\n".format(val_loss, acc, f1, auc))

                train_losses.append(train_losses)
                val_losses.append(val_losses)


                early_stopping(val_loss=val_loss)
                if early_stopping.is_save:
                    ckp_path = os.path.join(self.checkpoint, 'ckp'+str(epoch+1)+'.pt')
                    logger.info(f"Saving model to: {ckp_path}")
                    self.config['storage']['resume_best_checkpoint'] = ckp_path
                    self.save_config()
                    self.save_model(current_epoch=epoch+1, path=ckp_path)
                if early_stopping.early_stop:
                    logger.info(f"Early stopping. Saving log loss to: {os.path.join(self.log, 'resume_loss.txt')}")
                    break

            logger.info(f"Total time per epoch: {time.time()-start} seconds")
        train_losses, val_losses = np.array(train_loss).reshape(-1,1), np.array(val_loss).reshape(-1,1)
        np.savetxt(os.path.join(self.log, 'resume_loss.txt'), np.hstack((train_losses, val_losses)), delimiter='#')

        logger.info("Start testing resume...")
        logger.info(f"Loading the best model from {self.config['storage']['resume_best_checkpoint']}...")
        current_epoch = self.load_model(path=self.config['storage']['resume_best_checkpoint'])
        logger.info(f"With epoch: {current_epoch}")
        self.test()
        logger.info("Finish testing.")

def str2bool(s: str):
    if isinstance(s, bool):
        return s
    if s.lower() in ['yes', 'true', '1']:
        return True
    elif s.lower() in ['none', 'false', 'null', '0']:
        return False
    else:
        return argparse.ArgumentTypeError("Boolen values are expected!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/config.ini', help="Path to the config file")
    parser.add_argument('--resume', type=str2bool, const=True, nargs="?", default=False, help="Whether training resume from a checkpoint or not")

    args = parser.parse_args()
    set_seed()
    
    # config = configparser.ConfigParser()
    # config.read(args.config)
    
    
    transformers.logging.set_verbosity_error()
    torch.cuda.set_device(2)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # tokenizer = AutoTokenizer.from_pretrained(config['dataset']['tokenizer'])

    # logger.info("Preparing dataloaders...")
    logger.info("Initializing trainer...")
    trainer = Trainer(device=device, config_file=args.config)

    if not args.resume:
        trainer.fit()
    else:
        trainer.resume()

if __name__=='__main__':
    main()