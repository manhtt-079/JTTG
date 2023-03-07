import torch
import torch.nn as nn
from typing import List

class TransformerEncoderClassifier(nn.Module):
    def __init__(self,
                 n_classes: int,
                 d_model: int,
                 nhead=8,
                 dim_feedforward=2048,
                 dropout=0.1,
                 num_layers=2
        ):
        super(TransformerEncoderClassifier, self).__init__()

        self.nhead = nhead
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        layer_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, norm=layer_norm)

        self.out_proj = nn.Linear(d_model, n_classes)

    def forward(self, x, mask):
        """
        Forward function. ``x`` is the input ``sent_vector`` tensor and ``mask`` avoids computations
        on padded values. Returns ``sent_scores``.
        """
        # add dimension in the middle
        attn_mask = mask.unsqueeze(1)
        attn_mask = attn_mask.expand(-1, attn_mask.size(2), -1)
        # repeat the mask for each attention head
        attn_mask = attn_mask.repeat(self.nhead, 1, 1)
        # attn_mask is shape (batch size*num_heads, target sequence length, source sequence length)
        # set all the 0's (False) to negative infinity and the 1's (True) to 0.0 because the
        # attn_mask is additive
        attn_mask = (attn_mask.float()
            .masked_fill(attn_mask == 0, float("-inf"))
            .masked_fill(attn_mask == 1, float(0.0))
        )

        x = x.transpose(0, 1)
        # x is shape (source sequence length, batch size, feature number)

        x = self.encoder(x, mask=attn_mask)
        # x is still shape (source sequence length, batch size, feature number)
        x = x.transpose(0, 1).squeeze()
        # x is shape (batch size, source sequence length, feature number)
        x = self.out_proj(x)
        sent_scores = x.squeeze(-1) * mask.float()
        sent_scores[sent_scores == 0] = -9e3
        
        return sent_scores

class Pooling(nn.Module):
    def __init__(self, sent_rep_tokens:bool=True) -> None:
        super(Pooling, self).__init__()
        self.sent_rep_tokens = sent_rep_tokens
    
    def forward(self, word_vector: torch.Tensor, sent_rep_ids: torch.Tensor, sent_rep_mask: torch.Tensor):
        output_vectors = []
        output_masks = []
        if self.sent_rep_tokens:
            sents_vec = word_vector[torch.arange(word_vector.size(0)).unsqueeze(1), sent_rep_ids]
            sents_vec = sents_vec * sent_rep_mask[:, :, None].float()
            output_vectors.append(sents_vec)
            output_masks.append(sent_rep_mask)
        
        # batch_size, number of sent_rep_ids, embed_dim
        output_vector = torch.cat(output_vectors, 1)
        output_mask = torch.cat(output_masks, 1)

        return output_vector, output_mask


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

class AutomaticWeightedLoss(nn.Module):
    def __init__(self, n_losses: int):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(n_losses, requires_grad=True)
        self.params = nn.parameter.Parameter(params)
        
    def forward(self, *losses: List[torch.Tensor]):
        total_loss = 0
        for i, loss in enumerate(losses):
            total_loss +=  0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
            
        return total_loss


class AggregationLoss(nn.Module):
    def __init__(self):
        super(AggregationLoss, self).__init__()
        self.hparam = torch.nn.parameter.Parameter(
            torch.tensor(0.5, requires_grad=True))

    def forward(self, ext_loss: torch.Tensor, abs_loss: torch.Tensor):

        return self.hparam*ext_loss + (1-self.hparam)*abs_loss
        

class EarlyStopping:
    def __init__(self, patience=5, delta=1e-5):
        self.patience = patience
        self.max_save_after_not_improve = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.is_save = True
        self.delta = delta
        self.is_best_ckpt = None

    def __call__(self, val_loss: float):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.is_best_ckpt = True
        elif score > self.best_score + self.delta:
            self.counter += 1
            if self.counter <= self.max_save_after_not_improve:
                self.is_save = True
            else:
                self.is_save = False
            if self.counter >= self.patience:
                self.early_stop = True
            self.is_best_ckpt = False
        else:
            self.best_score = score
            self.counter = 0
            self.is_save = True
            self.is_best_ckpt = True