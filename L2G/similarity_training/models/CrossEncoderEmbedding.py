import copy
import gc
from rank_bm25 import BM25Okapi
from transformers import RobertaModel, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, AutoModel, RobertaPreTrainedModel
import numpy as np
import logging
import os
from typing import Dict, Type, Callable, List
import transformers
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.evaluation import SentenceEvaluator
from underthesea.pipeline.sent_tokenize import sent_tokenize
from underthesea.pipeline.word_tokenize import word_tokenize

logger = logging.getLogger(__name__)

class CrossEmbeddingPhoBertModel(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = RobertaModel(config=self.config)
        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        classifier_dropout = (
            self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(self.config.hidden_size, 1)
        self.init_weights()

    # def resize_token_embeddings(self, len_tokenizer):
    #     self.model.resize_token_embeddings(len_tokenizer)

    def forward(self, batch_features, sample_struc_dict, len_texts_batch, activation_fct, _target_device):
        model_predictions = None
        # model_predictions = torch.zeros((len_texts_batch, self.config.hidden_size), device=_target_device)
        
        # arr_idx = 0
        for f_idx, features in enumerate(batch_features):
            for name in features:
                features[name] = features[name].to(_target_device)
            model_prediction = self.model(**features, return_dict=True)
            output = model_prediction[0][:, 0, :]
            if model_predictions == None:
                model_predictions = output
            else:
                model_predictions = torch.cat((model_predictions, output), dim=0)
        # print('model_predictions: ', model_predictions)
        # print('model_predictions shape: ', model_predictions.shape)
        # final_idx = 0
        # final_model_prediction = torch.zeros((len(list(sample_struc_dict.keys())), self.config.hidden_size), device=_target_device)
        final_model_prediction = None
        for sample_idx in range(len(list(sample_struc_dict.keys()))):
            idx_relevant = torch.tensor(sample_struc_dict[sample_idx]).to(_target_device)
            sample_predict = torch.index_select(model_predictions, 0, idx_relevant)
            sample_predict = torch.mean(sample_predict, dim=0)
            sample_predict = sample_predict.reshape((1, self.config.hidden_size))
            if final_model_prediction == None:
                final_model_prediction = sample_predict
            else:
                final_model_prediction = torch.cat((final_model_prediction, sample_predict), dim=0)
        # print('final_model_prediction: ', final_model_prediction)
        # print('final_model_prediction shape: ', final_model_prediction.shape)
        x = self.dropout(final_model_prediction)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        logits = activation_fct(x)
        # print()
        return logits

class CrossEncoderEmbedding():
    def __init__(self, model_name:str, tokenizer_model_name:str, num_labels:int = None, max_length:int = None, device:str = None, tokenizer_args:Dict = {},
                 default_activation_function = None):
        """
        A CrossEncoder takes exactly two sentences / texts as input and either predicts
        a score or label for this sentence pair. It can for example predict the similarity of the sentence pair
        on a scale of 0 ... 1.
        It does not yield a sentence embedding and does not work for individually sentences.
        :param model_name: Any model name from Huggingface Models Repository that can be loaded with AutoModel. We provide several pre-trained CrossEncoder models that can be used for common tasks
        :param num_labels: Number of labels of the classifier. If 1, the CrossEncoder is a regression model that outputs a continous score 0...1. If > 1, it output several scores that can be soft-maxed to get probability scores for the different classes.
        :param max_length: Max length for input sequences. Longer sequences will be truncated. If None, max length of the model will be used
        :param device: Device that should be used for the model. If None, it will use CUDA if available.
        :param tokenizer_args: Arguments passed to AutoTokenizer
        :param default_activation_function: Callable (like nn.Sigmoid) about the default activation function that should be used on-top of model.predict(). If None. nn.Sigmoid() will be used if num_labels=1, else nn.Identity()
        """

        self.config = AutoConfig.from_pretrained(model_name)
        classifier_trained = True
        if self.config.architectures is not None:
            classifier_trained = any([arch.endswith('ForSequenceClassification') for arch in self.config.architectures])

        if num_labels is None and not classifier_trained:
            num_labels = 1

        if num_labels is not None:
            self.config.num_labels = num_labels

        # self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.config)
        self.model = CrossEmbeddingPhoBertModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name, **tokenizer_args)
        self.max_length = max_length

        # self.model.resize_token_embeddings(len(self.tokenizer))

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Use pytorch device: {}".format(device))

        self._target_device = torch.device(device)

        if default_activation_function is not None:
            self.default_activation_function = default_activation_function
            try:
                self.config.sbert_ce_default_activation_function = util.fullname(self.default_activation_function)
            except Exception as e:
                logger.warning("Was not able to update config about the default_activation_function: {}".format(str(e)) )
        elif hasattr(self.config, 'sbert_ce_default_activation_function') and self.config.sbert_ce_default_activation_function is not None:
            self.default_activation_function = util.import_from_string(self.config.sbert_ce_default_activation_function)()
        else:
            self.default_activation_function = nn.Sigmoid() if self.config.num_labels == 1 else nn.Identity()

    def smart_batching_collate(self, batch):
        texts = [[] for _ in range(len(batch[0].texts))]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text.strip())

            labels.append(example.label)

        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)
        labels = torch.tensor(labels, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(self._target_device)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized, labels

    def smart_batching_sentences_collate(self, batch):
        sample_struc_dict = {}
        texts = [[] for _ in range(len(batch[0].texts))]
        labels = []
        all_tokenized = []

        for e_idx, example in enumerate(batch):
            sample_struc_dict[e_idx] = []
            query, context = example.texts
            context_sents = context.split('\n')
            # if len(context_sents) <= 10:
            #     for sent in context_sents:
            #         texts[0].append(query)
            #         texts[1].append(sent)
            #         sample_struc_dict[e_idx].append(len(texts[1]) - 1)
            # else:
            #     tokens = context.split(' ')
            #     token_batch = len(tokens) // 10
            #     if len(tokens) / 10 != 0: token_batch+=1
            #     tokens_range = 10
            #     for part in range(tokens_range):
            #         a = part*token_batch
            #         b = (part+1)*token_batch
            #         b = min(b,len(tokens))
            #         texts[0].append(query)
            #         texts[1].append(' '.join(tokens[a: b]))
            #         sample_struc_dict[e_idx].append(len(texts[1]) - 1)
            if len(context_sents) > 1:
                context_sents = context_sents[:1]
            for sent in context_sents:
                texts[0].append(query)
                texts[1].append(sent)
                sample_struc_dict[e_idx].append(len(texts[1]) - 1)
            labels.append(example.label)
        CHUNK = 4
        # print('Finding similar titles...')
        len_batch = len(texts[1])
        CTS = len_batch//CHUNK
        if len_batch%CHUNK!=0: CTS += 1
        for j in range( CTS ):
            a = j*CHUNK
            b = (j+1)*CHUNK
            b = min(b,len_batch)
            batch_texts = []
            batch_texts.append(texts[0][a:b])
            batch_texts.append(texts[1][a:b])
            tokenized = self.tokenizer(*batch_texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)
            all_tokenized.append(tokenized)

        labels = torch.tensor(labels, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(self._target_device)

        return all_tokenized, labels, sample_struc_dict, len(texts[1])

    def smart_batching_collate_text_only(self, batch):
        texts = [[] for _ in range(len(batch[0]))]

        for example in batch:
            for idx, text in enumerate(example):
                texts[idx].append(text.strip())

        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized

    def smart_batching_sentences_collate_text_only(self, batch):
        sample_struc_dict = {}
        texts = [[] for _ in range(len(batch[0]))]
        all_tokenized = []

        for e_idx, example in enumerate(batch):
            sample_struc_dict[e_idx] = []
            query, context = example
            context_sents = context.split('\n')
            if len(context_sents) > 1:
                context_sents = context_sents[:1]
            for sent in context_sents:
                texts[0].append(query)
                texts[1].append(sent)
                sample_struc_dict[e_idx].append(len(texts[1]) - 1)
        CHUNK = 4
        len_batch = len(texts[1])
        CTS = len_batch//CHUNK
        if len_batch%CHUNK!=0: CTS += 1
        for j in range( CTS ):
            a = j*CHUNK
            b = (j+1)*CHUNK
            b = min(b,len_batch)
            batch_texts = []
            batch_texts.append(texts[0][a:b])
            batch_texts.append(texts[1][a:b])
            tokenized = self.tokenizer(*batch_texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)
            all_tokenized.append(tokenized)
        return all_tokenized, sample_struc_dict, len(texts[1])

    def fit(self,
            train_dataloader: DataLoader,
            evaluator: SentenceEvaluator = None,
            epochs: int = 1,
            loss_fct = None,
            activation_fct = nn.Identity(),
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = transformers.AdamW,
            optimizer_params: Dict[str, object] = {'lr': 2e-6},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True
            ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.
        :param train_dataloader: DataLoader with training InputExamples
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param loss_fct: Which loss function to use for training. If None, will use nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()
        :param activation_fct: Activation function applied on top of logits output of model.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        :param show_progress_bar: If True, output a tqdm progress bar
        """
        # train_dataloader.collate_fn = self.smart_batching_collate
        train_dataloader.collate_fn = self.smart_batching_sentences_collate

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.model.to(self._target_device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        self.best_score = -9999999
        num_train_steps = int(len(train_dataloader) * epochs)

        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

        if loss_fct is None:
            loss_fct = nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()


        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0
            if epoch == 0:
                self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, callback)

            self.model.zero_grad()
            self.model.train()

            for batch_features, labels, sample_struc_dict, len_texts_batch in tqdm(train_dataloader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                # print('featues: ', features)
                # print('len features shape: ', batch_features[0]['input_ids'].shape)
                # print('len features:', len(batch_features))
                # print('sample_struc_dict: ', sample_struc_dict)
                if use_amp:
                    # with autocast():
                        # model_predictions = self.model(**features, return_dict=True)
                        # print('model predict: ', model_predictions)
                        # logits = activation_fct(model_predictions.logits)
                        # if self.config.num_labels == 1:
                        #     logits = logits.view(-1)
                        # loss_value = loss_fct(logits, labels)

                    # scale_before_step = scaler.get_scale()
                    # scaler.scale(loss_value).backward()
                    # scaler.unscale_(optimizer)
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    # scaler.step(optimizer)
                    # scaler.update()

                    # skip_scheduler = scaler.get_scale() != scale_before_step
                    pass
                else:
                    logits = self.model(batch_features, sample_struc_dict, len_texts_batch, activation_fct, _target_device=self._target_device)
                    if self.config.num_labels == 1:
                        logits = logits.view(-1)
                    loss_value = loss_fct(logits, labels)
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1

                if evaluator is not None and evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, callback)

                    self.model.zero_grad()
                    self.model.train()

            if evaluator is not None:
                self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)
            
            # self.save(output_path)



    def predict(self, sentences: List[List[str]],
               batch_size: int = 32,
               show_progress_bar: bool = None,
               num_workers: int = 0,
               activation_fct = None,
               apply_softmax = False,
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False
               ):
        """
        Performs predicts with the CrossEncoder on the given sentence pairs.
        :param sentences: A list of sentence pairs [[Sent1, Sent2], [Sent3, Sent4]]
        :param batch_size: Batch size for encoding
        :param show_progress_bar: Output progress bar
        :param num_workers: Number of workers for tokenization
        :param activation_fct: Activation function applied on the logits output of the CrossEncoder. If None, nn.Sigmoid() will be used if num_labels=1, else nn.Identity
        :param convert_to_numpy: Convert the output to a numpy matrix.
        :param apply_softmax: If there are more than 2 dimensions and apply_softmax=True, applies softmax on the logits output
        :param convert_to_tensor:  Conver the output to a tensor.
        :return: Predictions for the passed sentence pairs
        """
        input_was_string = False
        if isinstance(sentences[0], str):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        inp_dataloader = DataLoader(sentences, batch_size=batch_size, collate_fn=self.smart_batching_sentences_collate_text_only, num_workers=num_workers, shuffle=False)

        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)

        iterator = inp_dataloader
        if show_progress_bar:
            iterator = tqdm(inp_dataloader, desc="Batches")

        if activation_fct is None:
            activation_fct = self.default_activation_function

        pred_scores = []
        self.model.eval()
        self.model.to(self._target_device)
        all_score = []
        with torch.no_grad():
            for batch_features, sample_struc_dict, len_texts_batch in iterator:
                batch_preds = []
                logits = self.model(batch_features, sample_struc_dict, len_texts_batch, activation_fct, _target_device=self._target_device)
                if apply_softmax and len(logits[0]) > 1:
                    logits = torch.nn.functional.softmax(logits, dim=1)
                pred_scores.extend(logits)

                # for idx in list(sample_struc_dict.keys()):
                #     print('idx: ', idx)
                #     for sample_idx in sample_struc_dict[idx]:
                #         batch_idx = sample_idx // 4
                #         print(self.tokenizer.convert_ids_to_tokens(batch_features[batch_idx]['input_ids'][sample_idx%4]))
                # 1/0
                        
        if self.config.num_labels == 1:
            pred_scores = [score[0] for score in pred_scores]

        if convert_to_tensor:
            pred_scores = torch.stack(pred_scores)
        elif convert_to_numpy:
            pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])

        if input_was_string:
            pred_scores = pred_scores[0]

        return pred_scores


    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback):
        """Runs evaluation during the training"""
        if evaluator is not None:
            score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)

    def save(self, path):
        """
        Saves all model and tokenizer to path
        """
        if path is None:
            return

        logger.info("Save model to {}".format(path))
        # torch.save(self.model.state_dict(), path)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self.config.save_pretrained(path)

    def save_pretrained(self, path):
        """
        Same function as save
        """
        return self.save(path)