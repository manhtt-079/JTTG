from sentence_transformers.evaluation import SentenceEvaluator, SimilarityFunction
import logging
import os
import csv
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
import numpy as np
from typing import List
import torch
from sentence_transformers.readers import InputExample
from sentence_transformers.util import pytorch_cos_sim



logger = logging.getLogger(__name__)

class AccuracyTopNEvaluator(SentenceEvaluator):
    def __init__(self, sentences1: List[str], sentences2: List[str], scores: List[float], top_n: int = 5, batch_size: int = 16, main_similarity: SimilarityFunction = None, name: str = '', show_progress_bar: bool = False, write_csv: bool = True):
        """
        Constructs an evaluator based for the dataset
        The labels need to indicate the similarity between the sentences.
        :param sentences1:  List with the first sentence in a pair
        :param sentences2: List with the second sentence in a pair
        :param scores: Similarity score between sentences1[i] and sentences2[i]
        :param write_csv: Write results to a CSV file
        """
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.scores = scores
        self.write_csv = write_csv
        self.top_n = top_n

        self.all_queries = list(set(self.sentences1))
        self.queryidx2corpus = {}
        for sent_idx, sent in enumerate(self.all_queries):
            self.queryidx2corpus[sent_idx] = [c_idx for c_idx, corpus_sent in enumerate(self.sentences2) if self.sentences1[c_idx] == sent]

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.scores)

        self.main_similarity = main_similarity
        self.name = name

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file = "similarity_evaluation"+("_"+name if name else '')+"_results.csv"
        self.csv_headers = ["epoch", "steps", "cosine_pearson", "cosine_spearman", "euclidean_pearson", "euclidean_spearman", "manhattan_pearson", "manhattan_spearman", "dot_pearson", "dot_spearman"]

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], top_n, **kwargs):
        sentences1 = []
        sentences2 = []
        scores = []

        for example in examples:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            scores.append(example.label)
        return cls(sentences1, sentences2, scores, top_n, **kwargs)


    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("EmbeddingSimilarityEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

        embeddings1 = model.encode(self.sentences1, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        embeddings2 = model.encode(self.sentences2, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        query_embeddings = model.encode(self.all_queries, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        labels = self.scores

        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
        dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]


        eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
        eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

        eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
        eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

        eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
        eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

        eval_pearson_dot, _ = pearsonr(labels, dot_products)
        eval_spearman_dot, _ = spearmanr(labels, dot_products)
        all_gt = 0
        all_gt_count = 0
        tp = 0
        for q_idx, query in enumerate(self.all_queries):
            query_embedding = query_embeddings[q_idx]
            query_labels = [labels[idx] for idx in self.queryidx2corpus[q_idx]]
            corpus_embeddings = [embeddings2[idx] for idx in self.queryidx2corpus[q_idx]]
            cos_scores = pytorch_cos_sim(torch.tensor([query_embedding]), torch.tensor(corpus_embeddings))[0]
            final_top_k = torch.topk(torch.tensor(cos_scores), k=self.top_n)
            ind_top_k = final_top_k[1].numpy()
            list_gt_idx = []
            for l_idx, l in enumerate(query_labels):
                if int(l) == 1:
                    all_gt_count += 1
                    all_gt += 1
                    list_gt_idx.append(l_idx)
            for idx in ind_top_k:
                if idx in list_gt_idx:
                    tp += 1
        # hard code
        all_gt = 662
        # ----------------
        logger.info("Predict correct {} samples per all {} samples:".format(tp, all_gt))
        logger.info("Actual gt in data: {} samples:".format(all_gt_count))
        logger.info("Accuracy Top {} :\t{:.4f}".format(self.top_n, tp/all_gt))
        logger.info("Cosine-Similarity :\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_cosine, eval_spearman_cosine))
        logger.info("Manhattan-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_manhattan, eval_spearman_manhattan))
        logger.info("Euclidean-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_euclidean, eval_spearman_euclidean))
        logger.info("Dot-Product-Similarity:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_dot, eval_spearman_dot))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline='', mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, eval_pearson_cosine, eval_spearman_cosine, eval_pearson_euclidean,
                                 eval_spearman_euclidean, eval_pearson_manhattan, eval_spearman_manhattan, eval_pearson_dot, eval_spearman_dot])


        if self.main_similarity == SimilarityFunction.COSINE:
            return eval_spearman_cosine
        elif self.main_similarity == SimilarityFunction.EUCLIDEAN:
            return eval_spearman_euclidean
        elif self.main_similarity == SimilarityFunction.MANHATTAN:
            return eval_spearman_manhattan
        elif self.main_similarity == SimilarityFunction.DOT_PRODUCT:
            return eval_spearman_dot
        elif self.main_similarity is None:
            return max(eval_spearman_cosine, eval_spearman_manhattan, eval_spearman_euclidean, eval_spearman_dot)
        else:
            raise ValueError("Unknown main_similarity value")

        # return tp/all_gt


class CECorrelationEvaluator:
    """
    This evaluator can be used with the CrossEncoder class. Given sentence pairs and continuous scores,
    it compute the pearson & spearman correlation between the predicted score for the sentence pair
    and the gold score.
    """
    def __init__(self, sentence_pairs: List[List[str]], sentences1: List[str], sentences2: List[str], scores: List[float], top_n=2, show_progress_bar=True, name: str=''):
        self.sentence_pairs = sentence_pairs
        self.scores = scores
        self.name = name
        self.top_n = top_n
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.show_progress_bar = show_progress_bar
        self.all_queries = list(set(self.sentences1))
        self.queryidx2corpus = {}
        for sent_idx, sent in enumerate(self.all_queries):
            self.queryidx2corpus[sent_idx] = [c_idx for c_idx, corpus_sent in enumerate(self.sentences2) if self.sentences1[c_idx] == sent]

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.scores)
        self.csv_file = "CECorrelationEvaluator" + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps", "Pearson_Correlation", "Spearman_Correlation"]

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], top_n=2, show_progress_bar=True, **kwargs):
        sentence_pairs = []
        scores = []
        sentences1 = []
        sentences2 = []

        for example in examples:
            sentence_pairs.append(example.texts)
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            scores.append(example.label)
        return cls(sentence_pairs, sentences1, sentences2, scores, top_n, show_progress_bar, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logging.info("CECorrelationEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)
        pred_scores = model.predict(self.sentence_pairs, convert_to_numpy=True, show_progress_bar=self.show_progress_bar)
        all_gt_count = 0
        tp = 0
        for q_idx, query in enumerate(self.all_queries):
            query_labels = [self.scores[idx] for idx in self.queryidx2corpus[q_idx]]
            pred_cross_scores = [pred_scores[idx] for idx in self.queryidx2corpus[q_idx]]
            final_top_k = torch.topk(torch.tensor(pred_cross_scores), k=self.top_n)
            ind_top_k = final_top_k[1].numpy()
            list_gt_idx = []
            for l_idx, l in enumerate(query_labels):
                if int(l) == 1:
                    all_gt_count += 1
                    list_gt_idx.append(l_idx)
            for idx in ind_top_k:
                if idx in list_gt_idx:
                    tp += 1
        # hard code
        # all_gt = 662
        # ----------------
        logger.info("Predict correct {} samples per all {} samples:".format(tp, all_gt_count))
        logger.info("Actual gt in data: {} samples:".format(all_gt_count))
        logger.info("Accuracy Top {} :\t{:.4f}".format(self.top_n, tp/all_gt_count))
        eval_pearson, _ = pearsonr(self.scores, pred_scores)
        eval_spearman, _ = spearmanr(self.scores, pred_scores)

        logging.info("Correlation:\tPearson: {:.4f}\tSpearman: {:.4f}".format(eval_pearson, eval_spearman))

        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, eval_pearson, eval_spearman])

        return eval_spearman