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
    