import math

import torch
from losses import MultipleNegativesRankingLoss, TripletLoss
from sentence_transformers import LoggingHandler, util, InputExample
from models import SentenceTransformer
from datasets import NoDuplicatesDataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
import random
from torch.utils.data import Dataset, DataLoader
from evaluator import AccuracyTopNEvaluator
from utils import load_json
from tqdm import tqdm

import transformers

transformers.logging.set_verbosity_error()

# Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
# /print debug information to stdout


class BiEncoderDataset(Dataset):
    def __init__(self, train_samples):
        self.train_samples = train_samples

    def __getitem__(self, idx):
        return self.train_samples[idx]

    def __len__(self):
        return len(self.train_samples)


model_name = 'microsoft/MiniLM-L12-H384-uncased'
train_batch_size = 16
max_seq_length = 256
num_epochs = 20

# Save path of the model
model_save_path = 'output_biencoder/training_xlm_base' + \
    model_name.replace("/", "-")+'-' + \
    datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

model = SentenceTransformer(model_name)
model.max_seq_length = max_seq_length

train_dataset_path = 'data/biencoder/word_segment_train-95.json'
val_dataset_path = 'data/biencoder/word_segment_val-5.json'

train_dataset = load_json(train_dataset_path)
val_dataset = load_json(val_dataset_path)

def add_to_samples(sent1, sent2, label):
    if sent1 not in train_data:
        train_data[sent1] = {'contradiction': set(
        ), 'entailment': set(), 'neutral': set()}
    train_data[sent1][label].add(sent2)

train_data = {}

for item in tqdm(train_dataset):
    sent1 = item['sentence1'].strip().replace('\n', '')
    sent2 = item['sentence2'].strip().replace('\n', '')
    add_to_samples(sent1, sent2, item['label'])

train_samples = []
for sent1, others in train_data.items():
    if len(others['entailment']) > 0 and len(others['contradiction']) > 0:
        for relevent in list(others['entailment']):
            # train_samples.append(InputExample(texts=[sent1, relevent]))
            # list_not_relevants = random.choices(list(others['contradiction']), k = 5)
            for not_relevant in others['contradiction']:
                train_samples.append(InputExample(texts=[sent1, relevent, not_relevant]))

logging.info("Train texts: {}".format(len(train_data.items())))
logging.info("Train samples len: {}".format(len(train_samples)))
logging.info("Train samples: {}".format(train_samples[0]))


logging.info("Read validation dataset")

dev_samples = []
for item in tqdm(val_dataset):
    dev_samples.append(InputExample(texts=[item['sentence1'].strip().replace('\n', ''), item['sentence2'].strip().replace('\n', '')],
                       label=1.0 if item['label'] == 'entailment' else 0.0))

logging.info("Dev samples: {}".format(len(dev_samples)))

# for sample in dev_samples:
#     if (sample.label) == 1:
#         print(sample.texts[0])
#         print(sample.texts[1])
#         print('-------------------')
# 1/0
train_dataset = BiEncoderDataset(train_samples=train_samples)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)

# Our training loss
train_loss = MultipleNegativesRankingLoss(model)

# train_loss = TripletLoss(model)


# dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
#     dev_samples, batch_size=train_batch_size, name='sts-dev',
#     show_progress_bar=True)

dev_evaluator = AccuracyTopNEvaluator.from_input_examples(
    dev_samples, batch_size=train_batch_size, name='sts-dev',
    show_progress_bar=True, top_n=2)

# Configure the training
# 10% of train data for warm-up
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.05) 
logging.info("Warmup-steps: {}".format(warmup_steps))


model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=int(len(train_dataloader)*0.1),
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          use_amp=False
          )