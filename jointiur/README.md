# JET

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.8+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.8+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.5+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra 1.1-89b8cd?style=for-the-badge&labelColor=gray"></a>

This is the repository for implementation of paper [Enhance Incomplete Utterance Restoration by Joint Learning Token Extraction and Text Generation](https://arxiv.org/abs/2204.03958 "Paper Link"), published in NAACL 2022.

![model_overview](./asset/model_overview.jpg)

This paper introduces a model for incomplete utterance restoration (IUR). Different from prior studies that only work on extraction or abstraction datasets, we design a simple but effective model, working for both scenarios of IUR. Our design simulates the nature of IUR, where omitted tokens from the context contribute to restoration. From this, we construct a Picker that identifies the omitted tokens. To support the picker, we design two label creation methods (soft and hard labels), which can work in cases of no annotation of the omitted tokens. The restoration is done by using a Generator with the help of the Picker on joint learning. Promising results on four benchmark datasets in extraction and abstraction scenarios show that our model is better than the pretrained T5 and non-generative language model methods in both rich and limited training data settings.

# 1. Setup

## Install required libraries

`pip install -r requirements.txt`

## Download fasttext pretrained weight

Put the fasttext pretrained weights trained on wiki for english and chinese to `./wv/` and unzip.
Pretrained weights are available from following links:

- [english](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip)
- [chinese](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.zh.zip)

## Unfold Data

Four public dataset is available for IUR tasks: CANARD, Restoraton200k, Rewrite20k and TASK.

Unzip the raw.zip and put directory 'raw' into `./data/`
`unzip raw.zip`

```
data
├── raw
│   ├── CANARD_Release
│   ├── Restoration-200K
│   ├── Rewrite20k
│   └── TASK
└── raw.zip
```

# 2. Preprocessing

## Run preprocessing to unify dataset

`python src/data_transform.py`

`data_transform.py` unifies the data format for CANARD, Restoration200k, TASK and Rewrite.
It alos builds the limited dataset with ratio=0.1 (10%)
After running script, directories with name {dataset}_ratio={ratio} are constructed. Each directory contains `train.json`, `dev.json`, `test.json`

```
data/processed
├── CANARD_ratio=0.1
├── CANARD_ratio=1
├── Restoration200k_ratio=0.1
├── Restoration200k_ratio=1
├── Rewrite20k_ratio=0.1
├── Rewrite20k_ratio=1
├── TASK_ratio=0.1
└── TASK_ratio=1
```

# 3. Training and Evaluation

## Run training and evaluation

`python src/run.py model={model} dataset={dataset} dataset.label_type={label_type} dataset.ratio={ratio}`

where you can choose paramters as following

- `model`: type of model. picker or writer or jointmodel.
- `dataset`: type of dataset. CANARD, Restoration200k, TASK or Rewrite
- `dataset.label_type`: type of label. soft or hard or defined.
    soft is soft label, hard is hard label, defined is only for Restoration200k which use token originally defined in Restoration200k.
- `dataset.ratio`: 1 or 0.1. 1 use full dataset, 0.1

## Check the performance

The result is stored in `./result/` with the name of folder `{model}_{dataset}_ratio={ratio}_{label_type}`
