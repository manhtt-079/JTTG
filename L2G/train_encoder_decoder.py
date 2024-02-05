import transformers
from transformers import AutoTokenizer, EncoderDecoderModel, BertGenerationEncoder, BertGenerationDecoder

MAIN_ENCODER = 'microsoft/MiniLM-L12-H384-uncased'

tokenizer = AutoTokenizer.from_pretrained(MAIN_ENCODER)
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

bos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

batch_size=4
encoder_max_length=512
decoder_max_length=128

def process_data_to_model_inputs(batch):
  inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=encoder_max_length)
  outputs = tokenizer(batch["highlights"], padding="max_length", truncation=True, max_length=decoder_max_length)

  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  batch["decoder_input_ids"] = outputs.input_ids
  batch["decoder_attention_mask"] = outputs.attention_mask
  batch["labels"] = outputs.input_ids.copy()

  # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`. 
  # We have to make sure that the PAD token is ignored
  batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

  return batch

encoder = BertGenerationEncoder.from_pretrained(MAIN_ENCODER, bos_token_id=bos_token_id, eos_token_id=eos_token_id)
# add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
decoder = BertGenerationDecoder.from_pretrained(MAIN_ENCODER, add_cross_attention=True, is_decoder=True, bos_token_id=bos_token_id, eos_token_id=eos_token_id)
bert2bert = EncoderDecoderModel(encoder=encoder, decoder=decoder)

