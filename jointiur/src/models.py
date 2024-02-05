from torch import nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5EncoderModel


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(cfg.input_dim, cfg.hidden_dim1)
        self.linear2 = nn.Linear(cfg.hidden_dim1, cfg.hidden_dim2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return x


class Picker(nn.Module):
    def __init__(self, cfg):
        super(Picker, self).__init__()
        model_cfg = cfg.model[cfg.dataset.name]
        self.encoder = T5EncoderModel.from_pretrained(model_cfg.pretrained_model)
        self.encoder.resize_token_embeddings(model_cfg.token_embeddings_size)
        self.ffn = FeedForward(model_cfg.FeedForward)
        self.label_type = cfg.dataset.label_type
        if self.label_type in ['hard', 'defined']:
            output_dim = model_cfg.Linear.output_dim_hard
        elif self.label_type == 'soft':
            output_dim = model_cfg.Linear.output_dim_soft
        self.linear = nn.Linear(model_cfg.Linear.input_dim, output_dim)

    def forward(self, input_ids, src_mask):
        x = self.encoder(input_ids=input_ids, attention_mask=src_mask).last_hidden_state
        x = self.linear(self.ffn(x))
        return x

class Writer(nn.Module):
    def __init__(self, cfg):
        super(Writer, self).__init__()
        model_cfg = cfg.model[cfg.dataset.name]
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_cfg.pretrained_model)
        self.t5.resize_token_embeddings(model_cfg.token_embeddings_size)

    def forward(self, input_ids, src_mask, labels, tgt_mask):
        x = self.t5(input_ids=input_ids, attention_mask=src_mask, decoder_attention_mask=tgt_mask, labels=labels)
        return x.loss

class JointModel(nn.Module):
	def __init__(self, cfg):
		super(JointModel, self).__init__()
		model_cfg = cfg.model[cfg.dataset.name]
		self.t5 = T5ForConditionalGeneration.from_pretrained(model_cfg.pretrained_model)
		self.t5.resize_token_embeddings(model_cfg.token_embeddings_size)
		self.ffn = FeedForward(model_cfg.FeedForward)
		if cfg.dataset.label_type in ['hard', 'defined']:
			output_dim = model_cfg.Linear.output_dim_hard
		elif cfg.dataset.label_type == 'soft':
			output_dim = model_cfg.Linear.output_dim_soft
		self.linear = nn.Linear(model_cfg.Linear.input_dim, output_dim)

	def forward(self, input_ids, src_mask, labels, tgt_mask):
		encoder_outputs = self.t5.encoder(input_ids=input_ids, attention_mask=src_mask, return_dict=True)
		x = encoder_outputs.last_hidden_state
		logit_picker = self.linear(self.ffn(x))
		x = self.t5(
			encoder_outputs=encoder_outputs,
            attention_mask=src_mask,
            decoder_attention_mask=tgt_mask,
            labels=labels
			)
		return logit_picker, x.loss