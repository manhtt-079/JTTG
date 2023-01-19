import argparse
import traceback
import os
import time
import torch
import torch.nn as nn
import numpy as np
import transformers
import evaluate
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from typing import Iterator, Tuple
from loguru import logger

from modules.datasets.dataset import dataset
from modules.model.utils import AutomaticWeightedLoss, EarlyStopping
from modules.model.exa_model import ExAb

from modules.utils import set_seed
from config.config import Config


class Trainer(object):
    def __init__(self, config: Config, device: torch.device):
        
        self.config = config
        self.config_parser = self.config.config
        self.config_file = self.config.config_file
        self.device = device
        self.model_args = self.config.model_args
        self.dataset_args = self.config.dataset_args
        self.trainer_args = self.config.trainer_args
        
        logger.info(f'Init and loading model_checkpoint: {self.model_args.pre_trained_name}')
        self.exab: nn.Module = ExAb(conf=self.model_args)
        self.exab.to(self.device)
        
        logger.info(f"Loading tokenizer: {self.model_args.pre_trained_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_args.pre_trained_name)
        if 't5' in self.model_args.pre_trained_name:
            self.tokenizer.add_special_tokens({'cls_token': '<s>', 'sep_token': '</s>'})
            self.exab.model.resize_token_embeddings(len(self.tokenizer))
            
        self.prefix = 'layer' if 'bart' in self.model_args.pre_trained_name else 'block'

        logger.info("Get dataloader")
        self.train_dataloader = self.get_dataloader(data_path=self.dataset_args.train_path, shuffle=True)
        self.val_dataloader = self.get_dataloader(data_path=self.dataset_args.valid_path)
        self.test_dataloader = None

        self.num_training_steps = len(self.train_dataloader) * self.trainer_args.max_epochs
        self.num_warmup_steps = int(self.trainer_args.warmup_ratio * self.num_training_steps)

        self.bce_loss = self.configure_loss_func(loss_func=self.trainer_args.losses[0])
        self.cross_ent_loss = self.configure_loss_func(loss_func=self.trainer_args.losses[1])
        
        self.auto_weighted_loss = AutomaticWeightedLoss(n_losses=self.trainer_args.n_losses)

        self.optimizer, self.scheduler = self.configure_optimizer_scheduler()
        
        self.best_checkpoint = self.trainer_args.best_checkpoint
        
        self.save_config()
        self.setup()

    def save_config(self):
        """Overwrite the config file"""
        
        try:
            with open(self.config_file, 'w') as f:
                self.config_parser.write(f)
        except Exception:
            traceback.print_exc()
    
    def make_dir(self, dir_path: str):
        if not os.path.exists(dir_path):
            os.system(f'mkdir -p {dir_path}')
            os.system(f"chmod -R 777 {dir_path}")
    
    def setup(self):
        self.make_dir(dir_path=self.trainer_args.log)
        self.make_dir(dir_path=self.trainer_args.checkpoint)
        
        self.freeze_layers()
        
        return True
    
    def get_dataloader(self, data_path: str, shuffle: bool = False):
        
        return dataset(tokenizer=self.tokenizer, 
                       data_path=data_path,
                       shuffle=shuffle,
                       src_max_length=self.dataset_args.src_max_length,
                       tgt_max_length=self.dataset_args.tgt_max_length,
                       batch_size=self.dataset_args.batch_size)
    
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

    def freeze_layers(self) -> None:
        """Freeze some layers of pre-trained model
        """
        freeze_layers = [f'{e}.{self.prefix}.{str(idx)}.' for idx in range(self.trainer_args.num_freeze_layers) for e in ['encoder', 'decoder']]
        for name, param in self.exab.model.named_parameters():
            if param.requires_grad and any(freeze_layer in name for freeze_layer in freeze_layers):
                param.requires_grad = False
                
    def configure_optimizer(self) -> torch.optim.Optimizer:
        param_optimizer = [[name, param] for name, param in self.exab.model.named_parameters() if param.requires_grad]
        optimizer_grouped_parameters = [
            {'params': [param for name, param in param_optimizer if not any(nd in name for nd in self.trainer_args.no_decay)],
            'weight_decay': self.trainer_args.weight_decay},
            {'params': [param for name, param in param_optimizer if any(nd in name for nd in self.trainer_args.no_decay)],
            'weight_decay': 0.0}
        ]

        optimizer: torch.optim.Optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.trainer_args.lr)
        
        return optimizer
    
    def configure_scheduler(self, optimizer: torch.optim.Optimizer):
        scheduler: torch.optim.lr_scheduler.LambdaLR = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                                                       num_warmup_steps=self.trainer_args.warmup_ratio,
                                                                                       num_training_steps=self.num_training_steps)
        
        return scheduler

    def configure_optimizer_scheduler(self):
        optimizer: torch.optim.Optimizer = self.configure_optimizer()
        scheduler: torch.optim.lr_scheduler.LambdaLR = self.configure_scheduler(optimizer=optimizer)
        
        return optimizer, scheduler

    def to_input(self, batch: Iterator):
        return tuple(t.to(self.device) for t in [batch[k] for k in ['input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask', 'sent_rep_ids', 'sent_rep_mask', 'label']])
    
    def step(self, batch: Iterator):
        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, sent_rep_ids, sent_rep_mask, label = self.to_input(batch)

        # outputs[0]: extractive loss, outputs[1]: abstractive loss
        outputs: Tuple[torch.Tensor, torch.Tensor] = self.exab(input_ids=input_ids,
                                                               attention_mask=attention_mask,
                                                               decoder_input_ids=decoder_input_ids,
                                                               decoder_attention_mask=decoder_attention_mask,
                                                               sent_rep_ids=sent_rep_ids,
                                                               sent_rep_mask=sent_rep_mask)
        
        ext_loss = self.bce_loss(outputs[0], label.float())
        
        # decoder_input_ids: [100,1,2,3,4, 5 ]
        # decoder_labels:    [ 1 ,2,3,4,5]
        # logits: init len=decoder_input_id len: --> logits[:, :-1] == len(labels)
        abs_label = decoder_input_ids.detach().clone()[:, 1:].contiguous().view(-1)
        logits = outputs[1][:, :-1].contiguous().view(-1, outputs[1].size(-1))
        abs_loss = self.cross_ent_loss(logits, abs_label)
        
        loss: torch.Tensor = self.auto_weighted_loss(ext_loss, abs_loss)
        loss = loss/self.trainer_args.accumulate_grad_batches
        
        loss.backward()
        
        return loss.item()*self.trainer_args.accumulate_grad_batches

    def validate(self, dataloader):
        self.exab.eval()
        with torch.no_grad():
        
            running_loss: float = 0.0
            abs_running_loss: float = 0.0
            
            for batch in dataloader:
                input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, sent_rep_ids, sent_rep_mask, label = self.to_input(batch)
                outputs: Tuple[torch.Tensor, torch.Tensor] = self.exab(input_ids=input_ids,
                                                                       attention_mask=attention_mask,
                                                                       decoder_input_ids=decoder_input_ids,
                                                                       decoder_attention_mask=decoder_attention_mask,
                                                                       sent_rep_ids=sent_rep_ids,
                                                                       sent_rep_mask=sent_rep_mask)
                ex_loss = self.bce_loss(outputs[0], label.float())
        
                ab_label = decoder_input_ids.detach().clone()[:, 1:].contiguous().view(-1)
                logits = outputs[1][:, :-1].contiguous().view(-1, outputs[1].size(-1))
                ab_loss = self.cross_ent_loss(logits, ab_label)
                
                loss: torch.Tensor = self.auto_weighted_loss(ex_loss, ab_loss)
                running_loss += loss.item()
                abs_running_loss += ab_loss.item()

            loss: float = running_loss/(len(dataloader))
            abs_loss: float = abs_running_loss/len(dataloader)
        return loss, abs_loss

    def train(self):
        early_stopping = EarlyStopping(patience=self.trainer_args.patience, delta=self.trainer_args.delta)
        
        train_losses, val_losses = [], []
        for epoch in range(self.trainer_args.max_epochs):
            start: float = time.time()
            self.exab.train()

            running_loss: float = 0.0
            n_iters = len(self.train_dataloader)
            self.optimizer.zero_grad()
            for idx, batch in enumerate(self.train_dataloader):
                
                loss: float = self.step(batch=batch)
                running_loss += loss
                if (idx+1) % self.trainer_args.accumulate_grad_batches == 0 or (idx+1) == n_iters:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                if (idx+1) % self.trainer_args.eval_steps == 0 or idx == 0:
                    logger.info("Epoch: {}/{} - iter: {}/{} - train_loss: {}".format(epoch+1, self.trainer_args.max_epochs, idx+1, n_iters, running_loss/(idx+1)))
            else:
                train_loss = running_loss/n_iters
                logger.info("Epochs: {}/{} - iter: {}/{} - train_loss: {}\n".format(epoch+1, self.trainer_args.max_epochs, idx+1, n_iters, train_loss))

                logger.info("-----:----- Evaluating -----:-----")
                val_loss, abs_loss = self.validate(dataloader=self.val_dataloader)
                logger.info("     Val loss: {}\n".format(val_loss))
                logger.info("     Abtractive loss: {}\n".format(abs_loss))

                train_losses.append(train_losses)
                val_losses.append(val_losses)

                early_stopping(val_loss=val_loss)
                if early_stopping.is_save:
                    ckp_path: str = os.path.join(self.trainer_args.checkpoint, 'ckp'+str(epoch+1)+'.pt')
                    logger.info(f"Saving model to: {ckp_path}")
                    self.config_parser.set(self.trainer_args.sec_name, 'best_checkpoint', ckp_path)
                    self.save_config()
                    self.save_model(current_epoch=epoch+1, path=ckp_path)
                if early_stopping.early_stop:
                    logger.info(f"Early stopping.")
                    break
        
        logger.info(f"Total time per epoch: {time.time()-start} seconds")
        logger.info(f"Saving log loss to: {os.path.join(self.trainer_args.log, 'loss.txt')}")
        train_losses, val_losses = np.array(train_losses).reshape(-1,1), np.array(val_losses).reshape(-1,1)
        np.savetxt(os.path.join(self.trainer_args.log, 'loss.txt'), np.hstack((train_losses, val_losses)), delimiter='#')


    def compute_rouge_score(self, dataloader):
        predictions = []
        references = []
        for _, batch in enumerate(tqdm(dataloader)):
            outputs = self.exab.model.generate(
                input_ids=batch['input_ids'].to('cuda'),
                max_length=self.dataset_args.tgt_max_length,
                attention_mask=batch['attention_mask'].to('cuda'),
                num_beams=self.trainer_args.num_beams
            )
            outputs = [self.tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in outputs]
            # Replace -100 in the labels as we can't decode them
            labels = np.where(batch['decoder_input_ids'][:, 1:] != -100, batch['decoder_input_ids'][:, 1:], self.tokenizer.pad_token_id)
            actuals = [self.tokenizer.decode(lb, clean_up_tokenization_spaces=False, skip_special_tokens=True) for lb in labels]
            
            predictions.extend(outputs)
            references.extend(actuals)

        metrics = evaluate.load('rouge')
        results = metrics.compute(predictions=predictions, references=references)
        
        return results
        
    def test(self):
        del self.train_dataloader
        del self.val_dataloader
        self.test_dataloader = self.get_dataloader(data_path=self.dataset_args.test_path)
        rouge_score = self.compute_rouge_score(dataloader=self.test_dataloader)
        logger.info("     Rouge_score: {}\n".format(rouge_score))

    def fit(self):
        logger.info("-----:----- Training -----:-----")
        logger.info(f"Epoch: {self.trainer_args.max_epochs}")
        logger.info(f"Num training steps: {self.num_training_steps}")
        logger.info(f"Num warmup steps: {self.num_warmup_steps}")
        logger.info(f"Accumulation steps: {self.trainer_args.accumulate_grad_batches}")
        logger.info(f"Eval steps: {self.trainer_args.eval_steps}")
        logger.info(f"Weight decay: {self.trainer_args.weight_decay}")
        logger.info(f"Learning rate: {self.trainer_args.lr}")
        logger.info(f"Num freeze layers: {self.trainer_args.num_freeze_layers}")
        self.train()
        logger.info("Finish training.\n")
        logger.info("Start testing...")
        logger.info(f"Loading the best model from {self.config_parser[self.trainer_args.sec_name]['best_checkpoint']}...")
        current_epoch = self.load_model(path=self.config_parser[self.trainer_args.sec_name]['best_checkpoint'])
        logger.info(f"With epoch: {current_epoch}")
        self.test()
        logger.info("-----:----- Finish testing. -----:-----")

    def load_model(self, path: str):
        ckp = torch.load(path)
        self.exab.load_state_dict(ckp['model_state_dict'])
        self.optimizer.load_state_dict(ckp['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckp['scheduler_state_dict'])
        current_epoch = ckp['current_epoch']

        return current_epoch

    def save_model(self, current_epoch: int, path: str):
        torch.save({
            "model_state_dict": self.exab.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "current_epoch": current_epoch
        }, path)

    # don't use anymore
    def resume(self, epochs_to_resume: int):
        current_epoch = self.load_model(path=self.config_parser[self.conf.trainer.sec_name]['best_checkpoint'])
        logger.info(f"Resume training model from epoch {current_epoch}...")
        early_stopping = EarlyStopping(patience=self.patience, delta=self.delta)
        train_losses, val_losses = [], []
        for epoch in range(current_epoch, current_epoch + epochs_to_resume):
            start = time.time()
            self.exab.train()

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
                    logger.info("Epoch: {}/{} - iter: {}/{} - train_loss: {}".format(epoch+1, self.epochs, idx+1, n_iters, running_loss/(idx+1)))
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
                    self.config_parser.set(self.conf.trainer.sec_name, 'resume_best_checkpoint', ckp_path)
                    self.save_config()
                    self.save_model(current_epoch=epoch+1, path=ckp_path)
                if early_stopping.early_stop:
                    logger.info(f"Early stopping. Saving log loss to: {os.path.join(self.log, 'resume_loss.txt')}")
                    break

            logger.info(f"Total time per epoch: {time.time()-start} seconds")
        train_losses, val_losses = np.array(train_losses).reshape(-1,1), np.array(val_losses).reshape(-1,1)
        np.savetxt(os.path.join(self.log, 'resume_loss.txt'), np.hstack((train_losses, val_losses)), delimiter='#')

        logger.info("Start testing...")
        logger.info(f"Loading the best model from {self.config_parser[self.conf.trainer.sec_name]['resume_best_checkpoint']}...")
        current_epoch = self.load_model(path=self.config_parser[self.conf.trainer.sec_name]['resume_best_checkpoint'])
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


def set_gpu(idx: int, cuda_visible_devices: str = '0,1,2'):
    transformers.logging.set_verbosity_error()
    torch.cuda.set_device(idx)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

def set_multi_gpu(cuda_visible_devices: str = '0,1,2,3'):
    transformers.logging.set_verbosity_error()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='./config/config.ini', help="Path to the config file")
    parser.add_argument('--dataset_name', type=str, default='reddit_tifu_dataset', help="Path to the config file")
    parser.add_argument('--model_name', type=str, default='bart-sum', help="Path to the config file")
    parser.add_argument('--is_long', type=str2bool, const=True, nargs="?", default=False)
    parser.add_argument('--use_us_test', type=str2bool, const=True, nargs="?", default=False)
    # parser.add_argument('--resume', type=str2bool, const=True, nargs="?", default=False, help="Whether training resume from a checkpoint or not")
    parser.add_argument('--gpu_idx', type=int, default=1, help="Path to the config file")


    args = parser.parse_args()
    set_seed()
    # set_multi_gpu()
    set_gpu(idx=args.gpu_idx)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config(config_file=args.config_file, dataset_name=args.dataset_name, model_name=args.model_name, is_long=args.is_long, use_us_test=args.use_us_test)
    
    logger.info("Initializing trainer...")
    trainer = Trainer(conf=config ,device=device)
    
    trainer.fit()

if __name__=='__main__':
    pass