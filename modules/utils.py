import logging
import random
from typing import Tuple
import numpy as np
import torch
from sklearn import metrics

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(name)s/%(funcName)s - %(levelname)s - %(message)s',
                        level=logging.DEBUG,
                        datefmt="%m/%d/%Y %I:%M:%S %p")
    logger = logging.getLogger(__name__)
    return logger


logger = init_logger()

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    logger.info(f'using seed: {seed}')

def compute_metrics(y_true: np.array, y_pred: np.array):
    confusion_mt = metrics.confusion_matrix(y_true, y_pred)
    # num_labels = confusion_mt.sum(axis=1)
    tp = np.diag(confusion_mt)
    fn = confusion_mt.sum(axis=1)-tp
    fp = confusion_mt.sum(axis=0)-tp

    p = np.nan_to_num(tp/(tp+fp))
    r = np.nan_to_num(tp/(tp+fn))
    f1 = 2*(p*r)/(p+r)

    f1: np.array = np.nan_to_num(f1)
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc: float = metrics.auc(fpr, tpr)
    
    acc: float = sum(tp)/confusion_mt.sum()
    # macro_f1 = np.mean(f1)
    # w_avg_f1 = sum(f1*num_labels)/num_labels.sum()


    return acc, f1[-1], auc