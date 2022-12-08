import torch
import torch.nn as nn
from transformers import AutoModel
from .utils import Pooling, TransformerEncoderClassifier
from config.config import ModelConf


class ExAbClassifierHead(nn.Module):
    def __init__(self,
                 input_dim: int,
                 inner_dim: int,
                 num_classes: int,
                 pooler_dropout: float
        ) -> None:
        super(ExAbClassifierHead, self).__init__()
        
        self.ln = nn.Linear(in_features=input_dim, out_features=inner_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(in_features=inner_dim, out_features=num_classes)
        
    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.ln(self.dropout(hidden_states))
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        outputs = self.out_proj(hidden_states).squeeze(-1)
        outputs = outputs * mask.float()
        outputs[outputs==0] = -9e-3
        
        return torch.sigmoid(outputs)

class ExAbGenerationHead(nn.Module):
    def __init__(self, input_dim: int, vocab_size: int) -> None:
        super(ExAbGenerationHead, self).__init__()
        
        self.ln = nn.Linear(in_features=input_dim, out_features=vocab_size)
        
    def forward(self, hidden_states: torch.Tensor):
        return self.ln(hidden_states)

class ExAb(nn.Module):
    
    def __init__(self, conf: ModelConf) -> None:
        super(ExAb, self).__init__()
        
        self.conf = conf
        self.name = self.conf.name
        self.model = AutoModel.from_pretrained(self.conf.pre_trained_name)
        self.pooling = Pooling(sent_rep_tokens=self.conf.sent_rep_tokens)
        self.classifier_head = TransformerEncoderClassifier(n_classes=self.conf.n_classes,
                                                            d_model=self.model.config.hidden_size,
                                                            nhead=self.conf.nhead,
                                                            dim_feedforward=self.conf.ffn_dim,
                                                            dropout=self.conf.pooler_dropout,
                                                            num_layers=self.conf.num_layers
                                                            )
        self.generator_head = ExAbGenerationHead(self.model.config.d_model, self.model.shared.num_embeddings)
        
        
    def forward(self, 
                input_ids: torch.LongTensor,
                attention_mask: torch.LongTensor,
                decoder_input_ids: torch.LongTensor,
                decoder_attention_mask: torch.LongTensor,
                sent_rep_ids: torch.Tensor,
                sent_rep_mask: torch.Tensor):
        
        # batch_size, seq_len, embed_dim
        # print(input_ids.device, attention_mask.device, end='\n\n')
        word_vectors = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        
        # batch_size, seq_len, embed_dim
        lm_hidden_states = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     decoder_input_ids=decoder_input_ids,
                                     decoder_attention_mask=decoder_attention_mask).last_hidden_state

        # batch_size, no. sent_rep_ids, embed_dim
        sent_vecs, sent_mask = self.pooling(word_vectors, sent_rep_ids, sent_rep_mask)
        
        # batch_size, no. sent_rep_ids, embed_dim
        cls_logits = self.classifier_head(sent_vecs, sent_mask)
        # batch_size, seq_len, vocab_size
        lm_logits = self.generator_head(hidden_states=lm_hidden_states)
        
        
        return (cls_logits, lm_logits)