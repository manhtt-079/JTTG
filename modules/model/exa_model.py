import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM
import pytorch_lightning as pl
from .utils import Pooling, TransformerEncoderClassifier
from config.config import ModelConf


class ExAb(nn.Module):

    def __init__(self, conf: ModelConf) -> None:
        super(ExAb, self).__init__()

        self.conf = conf
        self.name = self.conf.name
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.conf.pre_trained_name)
        self.pooling = Pooling(sent_rep_tokens=self.conf.sent_rep_tokens)
        self.tfm_classifier = TransformerEncoderClassifier(n_classes=self.conf.n_classes,
                                                           d_model=self.model.config.hidden_size,
                                                           nhead=self.conf.nhead,
                                                           dim_feedforward=self.conf.ffn_dim,
                                                           dropout=self.conf.pooler_dropout,
                                                           num_layers=self.conf.num_layers
                                                           )
        if 'bart' in self.conf.pre_trained_name:
            self.encoder = self.model.model.encoder
        else:
            self.encoder = self.model.encoder

    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask: torch.LongTensor,
                decoder_input_ids: torch.LongTensor,
                decoder_attention_mask: torch.LongTensor,
                sent_rep_ids: torch.Tensor,
                sent_rep_mask: torch.Tensor):

        # batch_size, seq_len, embed_dim
        word_vectors = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]

        # (batch_size, seq_len, vocab_size)
        lm_logits = self.model(input_ids=input_ids,
                               attention_mask=attention_mask,
                               decoder_input_ids=decoder_input_ids,
                               decoder_attention_mask=decoder_attention_mask).logits

        # batch_size, no. sent_rep_ids, embed_dim
        sent_vecs, sent_mask = self.pooling(word_vectors, sent_rep_ids, sent_rep_mask)
        # batch_size, no. sent_rep_ids, embed_dim
        cls_logits = self.tfm_classifier(sent_vecs, sent_mask)

        return (cls_logits, lm_logits)


class ExAbModel(pl.LightningModule):
    def __init__(self, conf: ModelConf):
        super(ExAbModel, self).__init__()
        self.exab = ExAb(conf=conf)

    def configure_optimizers(self):
        param_optimizer = [[name, param] for name, param in self.exab.model.named_parameters() if param.requires_grad]
        optimizer_grouped_parameters = [
            {'params': [param for name, param in param_optimizer if not any(nd in name for nd in self.no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [param for name, param in param_optimizer if any(nd in name for nd in self.no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=self.lr)

        return optimizer
    
#     # def optimizers(self):
#     #     pass
    
#     # def lr_schedulers(self):
#     #     pass
    
#     # def forward(self):
#     #     pass
    
#     # def training_step(self):
#     #     pass
    
#     # def training_step_end(self, step_output):
#     #     pass
    
#     # def training_epoch_end(self, outputs):
#     #     pass
    
#     # def validation_step(self):
#     #     pass
    
#     # def validation_step_end(self, step_output):
#     #     pass
    
#     # def validation_epoch_end(self, outputs):
#     #     pass
    
#     # def test_step(self):
#     #     pass
    
#     # def test_step_end(self, step_output):
#     #     pass
    
#     # def test_epoch_end(self, outputs):
#     #     pass
    
#     # def configure_callbacks(self):
#     #     pass
    
#     def freeze_layers(self):
#         freeze_layers = []
#         if 'bart' in self.conf.model.name:
#             freeze_layers = [f'{e}.layers.{str(idx)}.' for idx in range(self.num_freeze_layers) for e in ['encoder', 'decoder']]
#         elif 't5' in self.conf.model.name:
#             freeze_layers = [f'{e}.block.{str(idx)}.layer' for idx in range(self.num_freeze_layers) for e in ['encoder', 'decoder']]
        
#         for name, param in self.model.named_parameters():
#             if param.requires_grad and any(freeze_layer in name for freeze_layer in freeze_layers):
#                 param.requires_grad = False
    
#     def configure_optimizers(self):
#         param_optimizer = [[name, param] for name, param in self.model.named_parameters() if param.requires_grad]
#         optimizer_grouped_parameters = [
#             {'params': [param for name, param in param_optimizer if not any(nd in name for nd in self.no_decay)],
#             'weight_decay': self.weight_decay},
#             {'params': [param for name, param in param_optimizer if any(nd in name for nd in self.no_decay)],
#             'weight_decay': 0.0}
#         ]

#         optimizer: torch.optim.Optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)
        
#         return optimizer
    
    
#     # def prepare_data(self):
#     #     pass
    
#     # def setup(self, stage):
#     #     pass
    
#     # def train_dataloader(self):
#     #     pass
    
#     # def val_dataloader(self):
#     #     pass
    
#     # def test_dataloader(self):
#     #     pass
    
    