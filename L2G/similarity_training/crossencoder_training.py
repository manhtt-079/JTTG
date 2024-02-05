import random
from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util
from models import CrossEncoder, CrossEncoderEmbedding, CrossEncoderSentence, CrossEncoderLongformer
from evaluator import CECorrelationEvaluator
from sentence_transformers import InputExample
import logging
from datetime import datetime
import sys
import os
import gzip
import csv

from transformers.utils.dummy_tokenizers_objects import BartTokenizerFast
from utils import load_json, make_article_and_id_2_corpus_text, load_pickle
from tqdm import tqdm
import copy

import transformers

transformers.logging.set_verbosity_error()


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
#### /print debug information to stdout

use_95_train_data = True
top_n = 2

train_json_path = 'data/new_data_wordsegment/train_split_1.json'
dev_json_path = 'data/new_data_wordsegment/val_split_1.json'

# train_json_path = 'data/train_split_1.json'
# dev_json_path = 'data/val_split_1.json'
# CORPUS_PATH = 'data/new_legal_corpus.json'
corpus = load_json(CORPUS_PATH)
article_and_id_2_corpus_text = make_article_and_id_2_corpus_text(corpus)

text_corpus = []
text_titles = []
count_idx = 0
idx2article = {}
for law_id_idx, law_id in tqdm(enumerate(list(article_and_id_2_corpus_text.keys()))):
    for article_id_idx, article_id in enumerate(article_and_id_2_corpus_text[law_id].keys()):
        law_text = law_id + article_and_id_2_corpus_text[law_id][article_id]['title'].replace('Điều ' + article_id, '') + ' ' + article_and_id_2_corpus_text[law_id][article_id]['text']
        text_titles.append(article_and_id_2_corpus_text[law_id][article_id]['title'])
        article_and_id_2_corpus_text[law_id][article_id]['index'] = count_idx
        idx2article[count_idx] = {
            "law_id": law_id,
            "article_id": article_id
        }
        count_idx += 1
        text_corpus.append(law_text)

train_data = load_json(train_json_path)['items']
dev_data = load_json(dev_json_path)['items']

bm_top_n_train = load_pickle('data/filters/bm_top_100_train_split_1_ver_2.pkl')
bm_top_n_val = load_pickle('data/filters/bm_top_100_val_split_1_ver_2.pkl')
print('len bm_top_n_train: ', len(bm_top_n_train[0]))
print('len bm_top_n_val: ', len(bm_top_n_val[0]))
train_top_n = 50
val_top_n = 50

bm_top_n_train = [bm_top_n_train[idx][:train_top_n] for idx in range(len(bm_top_n_train))]
bm_top_n_val = [bm_top_n_val[idx][:val_top_n] for idx in range(len(bm_top_n_val))]

# print('len bm_top_n_train: ', len(bm_top_n_train[0]))
# print('len bm_top_n_val: ', len(bm_top_n_val[0]))

assert len(bm_top_n_train[0]) == train_top_n
assert len(bm_top_n_val[0]) == val_top_n
print('len text_corpus:',len(text_corpus))
#Define our Cross-Encoder
train_batch_size = 8
num_epochs = 20
model_save_path = 'output/training-CrossEncoder_stsb-distilroberta-base_'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# model_name = 'allenai/longformer-base-4096'
# tokenizer_model_name = 'allenai/longformer-base-4096'
# tokenizer_model_name = 'sentence-transformers/distiluse-base-multilingual-cased-v1'

model_name = 'microsoft/MiniLM-L12-H384-uncased'
tokenizer_model_name = 'microsoft/MiniLM-L12-H384-uncased'

# model_name = 'output/training-CrossEncoder_stsb-distilroberta-base_2021-12-06_16-51-13'
# tokenizer_model_name = 'output/training-CrossEncoder_stsb-distilroberta-base_2021-12-06_16-51-13'

# model = CrossEncoderSentence(model_name, tokenizer_model_name=tokenizer_model_name, max_length=256, num_labels=1)
model = CrossEncoder(model_name, tokenizer_model_name=tokenizer_model_name, max_length=512, num_labels=1)
# model = CrossEncoderEmbedding(model_name, tokenizer_model_name=tokenizer_model_name, max_length=128, num_labels=1)
# model = CrossEncoderLongformer(model_name, tokenizer_model_name=tokenizer_model_name, max_length=1024, num_labels=1)
# print('num_labels:', model.model.num_labels)

def get_sample_data(question_items, bm_top_n, article_and_id_2_corpus_text, text_corpus, data_type='train'):
    samples = []
    for item_idx, item in tqdm(enumerate(question_items)):
        gt = [article_and_id_2_corpus_text[article['law_id']][article['article_id']]['index'] for article in item['relevant_articles']]
        item_topn = bm_top_n[item_idx] if type(bm_top_n[item_idx]) == list else bm_top_n[item_idx].tolist()
        all_article_relevant = list(set(gt + item_topn)) if data_type == 'train' else item_topn
        for article_idx in all_article_relevant:
            article_text = text_corpus[article_idx]
            title = text_titles[article_idx]
            samples.append(InputExample(texts=[item['question'], article_text], label=1 if article_idx in gt else 0))
    return samples

def get_sample_resplit_data(question_items, bm_top_n, article_and_id_2_corpus_text, text_corpus, data_type='train'):
    samples = []
    for item_idx, item in tqdm(enumerate(question_items)):
        item_samples = []
        gt = [article_and_id_2_corpus_text[article['law_id']][article['article_id']]['index'] for article in item['relevant_articles']]
        item_topn = bm_top_n[item_idx] if type(bm_top_n[item_idx]) == list else bm_top_n[item_idx].tolist()
        all_article_relevant = list(set(gt + item_topn)) if data_type == 'train' else item_topn
        for article_idx in all_article_relevant:
            article_text = text_corpus[article_idx]
            # law_id = idx2article[article_idx]['law_id']
            # article_id = idx2article[article_idx]['article_id']
            # title = article_and_id_2_corpus_text[law_id][article_id]['title']
            # article_text = law_id + title.replace('Điều ' + article_id, '') + ' ' + get_closest(item['question'], corpus_text=article_and_id_2_corpus_text[law_id][article_id]['text'])
            item_samples.append(InputExample(texts=[item['question'], article_text], label=1 if article_idx in gt else 0))
        samples.append(item_samples)
    return samples


if use_95_train_data:
    # train all dataset train + dev 
    dev_samples_all_gt = get_sample_resplit_data(dev_data, bm_top_n_val, article_and_id_2_corpus_text, text_corpus, data_type='train')
    train_samples = get_sample_resplit_data(train_data, bm_top_n_train, article_and_id_2_corpus_text, text_corpus, data_type='train')
    full_data_samples = train_samples + dev_samples_all_gt
    dev_idx = random.choices(list(range(len(full_data_samples))), k=int(len(full_data_samples) * 5 / 100))
    dev_samples_zip = [full_data_samples[idx] for idx in dev_idx]
    train_samples_zip = [full_data_samples[idx] for idx in range(len(full_data_samples)) if idx not in dev_idx]
    train = []
    dev = []
    for sample in train_samples_zip:
        train.extend(sample)
    for sample in dev_samples_zip:
        dev.extend(sample)
    
    # We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
    train_dataloader = DataLoader(train, shuffle=True, batch_size=train_batch_size)

    # We add an evaluator, which evaluates the performance during training
    evaluator = CECorrelationEvaluator.from_input_examples(dev, name='sts-dev', top_n=top_n)
    # Configure the training
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #2% of train data for warm-up
    # warmup_steps = 0
    logger.info("Warmup-steps: {}".format(warmup_steps))
    logger.info("Batch-size: {}".format(train_batch_size))
    logger.info("Number samples of training data: {}".format(len(train)))
    logger.info("Number samples of dev data: {}".format(len(dev)))
else:
    train_samples = get_sample_data(train_data, bm_top_n_train, article_and_id_2_corpus_text, text_corpus, data_type='train')
    dev_samples = get_sample_data(dev_data, bm_top_n_val, article_and_id_2_corpus_text, text_corpus, data_type='train')

    # We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

    # We add an evaluator, which evaluates the performance during training
    evaluator = CECorrelationEvaluator.from_input_examples(dev_samples, name='sts-dev', top_n=top_n)

    # Configure the training
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #2% of train data for warm-up
    logger.info("Warmup-steps: {}".format(warmup_steps))
    logger.info("Batch-size: {}".format(train_batch_size))
    logger.info("Number samples of training data: {}".format(len(train_samples)))
    logger.info("Number samples of dev data: {}".format(len(dev_samples)))



# print("Model's state_dict:")
# for param_tensor in model.model.state_dict():
#     print(param_tensor, "\t", model.model.state_dict()[param_tensor].size())

# Train the model
model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path=model_save_path)
