import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
import evaluate
from typing import Dict, List, Any
from config.config import get_logger
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Trainer, TrainingArguments, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = get_logger()
metrics = evaluate.load('rouge')
def get_data(data_path: str):
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data


def compute_metric(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    tokenized_test: Dataset
):
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")

    max_target_length = 256
    dataloader = torch.utils.data.DataLoader(tokenized_test, collate_fn=data_collator, batch_size=16)

    predictions = []
    references = []
    for _, batch in enumerate(tqdm(dataloader)):
        outputs = model.generate(
            input_ids=batch['input_ids'].to(device),
            max_length=max_target_length,
            attention_mask=batch['attention_mask'].to(device)
        )
        with tokenizer.as_target_tokenizer():
            outputs = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in outputs]
            labels = np.where(batch['labels'] != -100,  batch['labels'], tokenizer.pad_token_id)
            actuals = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in labels]
            predictions.extend(outputs)
            references.extend(actuals)

    results = metrics.compute(predictions=predictions, references=references)

    return results

def loop(
    dataset: Dict[str, Any],
    model,
    tokenizer,
    output_dir: str
):
    logger.info('Loading dataset')
    df_train = pd.DataFrame(dataset['train'])[['src', 'tgt']]
    df_val = pd.DataFrame(dataset['val'])[['src', 'tgt']]
    df_test = pd.DataFrame(dataset['test'])[['src', 'tgt']]

    train_set = Dataset.from_pandas(df_train)
    val_set = Dataset.from_pandas(df_val)
    test_set = Dataset.from_pandas(df_test)

    def preprocess_function(examples):
        inputs = tokenizer(examples["src"], max_length=1024, truncation=True, padding=True)
        labels = tokenizer(examples["tgt"], max_length=256, truncation=True, padding=True)
        inputs['labels'] = labels['input_ids']
        inputs['input_ids'] = inputs['input_ids']

        return inputs

    logger.info('Tokenizing dataset')
    tokenized_train = train_set.map(preprocess_function, batched=True, remove_columns=['src','tgt'], num_proc=8)
    tokenized_val = val_set.map(preprocess_function, batched=True, remove_columns=['src', 'tgt'], num_proc=8)
    tokenized_test = test_set.map(preprocess_function, batched=True, remove_columns=['src', 'tgt'], num_proc=8)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        num_train_epochs=30,
        learning_rate=3e-5,
        warmup_ratio=0.05,
        weight_decay=0.015,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        predict_with_generate=True,
        group_by_length=True,
        save_strategy="epoch",
        save_total_limit=1,
        gradient_accumulation_steps=16,
        evaluation_strategy="epoch",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer = tokenizer,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=data_collator,
        eval_dataset=tokenized_val
    )
    logger.info('Training')
    trainer.train()

    results = compute_metric(model=model, tokenizer=tokenizer, tokenized_test=tokenized_test)
    
    return results

def main():
    import argparse
    experiments = {
        'task1': {'pretrained_model_name': 'facebook/bart-base',
                  'output_dir': '/mnt/hdd/manhtt/project_sum/checkpoint/bart-usa'},
        'task2': {'pretrained_model_name': 't5-base',
                  'output_dir': '/mnt/hdd/manhtt/project_sum/checkpoint/t5-usa'}
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/mnt/hdd/manhtt/project_sum/data/usatoday_cnn/data_new.json')
    parser.add_argument('--task', type=str, default='task1')

    args = parser.parse_args()
    data = get_data(data_path=args.data_path)
    ex = experiments[args.task]
    for d in data:
        logger.info(f'Load tokenizer and model checkpoint: {ex["pretrained_model_name"]}')
        tokenizer = AutoTokenizer.from_pretrained(ex['pretrained_model_name'])
        model = AutoModelForSeq2SeqLM.from_pretrained(ex['pretrained_model_name'])
        results = loop(dataset=d, model=model, tokenizer=tokenizer, output_dir=ex['output_dir'])
        logger.info(f"Rouge score: {results}")
        del model, tokenizer

if __name__=='__main__':
    main()