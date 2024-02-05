from transformers import T5Tokenizer
import mlflow
from mlflow.tracking import MlflowClient
from mlflow import pytorch
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
import json
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from hydra import compose, initialize_config_dir, initialize



get_model_name = lambda cfg: f'{cfg.model.name}_{cfg.dataset.name}_{cfg.dataset.label_type}'

def load_tokenizer(cfg):
    tokenizer = T5Tokenizer.from_pretrained(cfg.pretrained_model)
    tokenizer.add_special_tokens({'additional_special_tokens': [cfg.sp_token1,cfg.sp_token2]})
    return tokenizer


class MyCriterion:
    def __init__(self, cfg):
        self.label_type = cfg.dataset.label_type
        if self.label_type == 'soft':
            self.criterion = nn.KLDivLoss(reduction='none')
        elif self.label_type in ['defined', 'hard']:
            self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def __call__(self, input, target):
        if self.label_type in ['defined', 'hard']:
            input = input.permute(0,2,1) #(batch,timestep,output_dim) -> #(batch,output_dim,timestep), output_dim=3
            loss = self.criterion(input, target)
        elif self.label_type == 'soft':
            input = input.squeeze(-1) #(batch,timestep,output_dim) -> #(batch,timestep), output_dim=1
            loss = self.criterion(F.log_softmax(input, dim=1), target).sum(axis=1).mean()
        return loss


class MlflowWriter():
    def __init__(self, experiment_name, run_name, run_id=None, **kwargs):
        self.client = MlflowClient(**kwargs)
        try:
            self.experiment_id = self.client.create_experiment(experiment_name)
        except:
            self.experiment_id = self.client.get_experiment_by_name(experiment_name).experiment_id

        if run_id is not None:
            self.run_id = run_id
        else:
            self.run_id = self.client.create_run(self.experiment_id).info.run_id
        self.client.set_tag(self.run_id, "mlflow.runName", run_name)


    def log_params_from_omegaconf_dict(self, params):
        for param_name, element in params.items():
            self._explore_recursive(param_name, element)

    def _explore_recursive(self, parent_name, element):
        if isinstance(element, DictConfig):
            for k, v in element.items():
                if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                    self._explore_recursive(f'{parent_name}.{k}', v)
                else:
                    self.client.log_param(self.run_id, f'{parent_name}.{k}', v)
        elif isinstance(element, ListConfig):
            for i, v in enumerate(element):
                self.client.log_param(self.run_id, f'{parent_name}.{i}', v)

    def log_torch_model(self, model):
        with mlflow.start_run(self.run_id):
            pytorch.log_model(model, 'models')

    def log_param(self, key, value):
        self.client.log_param(self.run_id, key, value)

    def log_metric(self, key, value, **kwargs):
        self.client.log_metric(self.run_id, key, value, **kwargs)

    def log_artifact(self, local_path):
        self.client.log_artifact(self.run_id, local_path)

    def set_terminated(self):
        self.client.set_terminated(self.run_id)


def get_logger(name='root', log_level=logging.INFO, simple=False):
    """
    Get or create logging object
    :param name: name of logging object, a logging file with the same name will also be created
    :param level: logging level
    :return:
    """

    logger = logging.getLogger(name=name)
    if not logger.handlers:
        if simple:
            formatter = logging.Formatter('%(message)s')
        else:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # file
        os.makedirs('./log', exist_ok=True)
        fh = TimedRotatingFileHandler(f'./log/{name}.txt', encoding='utf-8', when="midnight")
        fh.suffix = "%Y-%m-%d"
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # console
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        logger.setLevel(log_level)
    return logger


def save_json(savedir, fname, data):
    with open(os.path.join(savedir, fname), 'wt', encoding='utf8') as fp:
        json.dump(data, fp, ensure_ascii=False, indent=4)
