import json
import warnings
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from datasets import Dataset
import evaluate
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
metrics = evaluate.load('rouge')
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")

@dataclass
class DatasetConf:
    train_path: str = '/home/int2-user/project_sum/data/gov-report/train.csv'
    val_path: str = '/home/int2-user/project_sum/data/gov-report/val.csv'
    test_path: str = '/home/int2-user/project_sum/data/gov-report/test.csv'


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

def process(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    test_dataset,
    output_dir: str
):
    print('Loading dataset')
    df_train = pd.read_csv(train_dataset)[['document', 'summary']]
    df_val = pd.read_csv(val_dataset)[['document', 'summary']]
    df_test = pd.read_csv(test_dataset)[['document', 'summary']]

    train_set = Dataset.from_pandas(df_train)
    val_set = Dataset.from_pandas(df_val)
    test_set = Dataset.from_pandas(df_test)

    def preprocess_function(examples):
        inputs = tokenizer(examples["document"], max_length=1024, truncation=True, padding=True)
        labels = tokenizer(examples["summary"], max_length=256, truncation=True, padding=True)
        inputs['labels'] = labels['input_ids']
        inputs['input_ids'] = inputs['input_ids']

        return inputs

    print('Tokenizing dataset')
    tokenized_train = train_set.map(preprocess_function, batched=True, remove_columns=['document', 'summary'], num_proc=8)
    tokenized_val = val_set.map(preprocess_function, batched=True, remove_columns=['document', 'summary'], num_proc=8)
    tokenized_test = test_set.map(preprocess_function, batched=True, remove_columns=['document', 'summary'], num_proc=8)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        do_train=True,
        logging_strategy='steps',
        log_level='debug',
        logging_steps=500,
        do_eval=True,
        num_train_epochs=10,
        learning_rate=1e-4,
        warmup_ratio=0.05,
        weight_decay=0.015,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        predict_with_generate=True,
        group_by_length=True,
        save_strategy="epoch",
        save_total_limit=1,
        gradient_accumulation_steps=8,
        evaluation_strategy="epoch",
        report_to='none'
    )
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer = tokenizer,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=data_collator,
        eval_dataset=tokenized_val
    )
    print('Training')
    trainer.train()

    results = compute_metric(model=model, tokenizer=tokenizer, tokenized_test=tokenized_test)
    
    return results

def main(task: str):
    experiments = {
        'task1': {'pretrained_model_name': 'facebook/bart-base',
                  'output_dir': '/home/int2-user/project_sum/checkpoint/gov'},
        'task2': {'pretrained_model_name': 't5-base',
                  'output_dir': '/home/int2-user/project_sum/checkpoint/gov'}
    }
    dataset_conf = DatasetConf()
    ex = experiments[task]
    
    print(f'Load tokenizer and model checkpoint: {ex["pretrained_model_name"]}')
    tokenizer = AutoTokenizer.from_pretrained(ex['pretrained_model_name'])
    model = AutoModelForSeq2SeqLM.from_pretrained(ex['pretrained_model_name'])
    rouge_score = process(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_conf.train_path,
        val_dataset=dataset_conf.val_path,
        test_dataset=dataset_conf.test_path,
        output_dir=ex['output_dir']
    )
    
    return rouge_score

if __name__=='__main__':
    main('task1')