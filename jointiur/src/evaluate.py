import os
import json
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torchnlp.metrics import get_moses_multi_bleu
import matplotlib.pyplot as plt

from dataloader import load_dataiter
from score import Scorer, BasicTokenizer
from models import Picker, Writer, JointModel
from utils import load_tokenizer, get_model_name



def conf_mat(span_tags, pred_tags, save_dir, writer):
    labels = [0,1,2]
    display_labels = ['O','B','I']
    cm = confusion_matrix(span_tags, pred_tags, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
    disp.plot(cmap=plt.cm.Blues)
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    writer.log_artifact(cm_path)


def clss_report(span_tags,pred_tags,save_dir,writer):
    labels = [0,1,2]
    display_labels = ['O','B','I']
    report = classification_report(span_tags, pred_tags, labels=labels, target_names=display_labels, output_dict=True)
    cls_json_path = os.path.join(save_dir, 'classification_report.json')
    with open(cls_json_path, 'wt', encoding='utf8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    writer.log_artifact(cls_json_path)
    report_df = pd.DataFrame(report).transpose()
    cls_csv_path = os.path.join(save_dir, 'classification_report.csv')
    report_df.to_csv(cls_csv_path, index=True)
    writer.log_artifact(cls_csv_path)


def record_pred(orig_utterances, labels_writer, preds_writer, save_dir, writer):
    results = [{'original': o, 'reference': r, 'prediction': p} for o, r, p in zip(orig_utterances,labels_writer, preds_writer)]
    res_path =os.path.join(save_dir, 'predictions.json')
    with open(res_path, 'wt', encoding='utf8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    writer.log_artifact(res_path)


def restoration_scores(labels_writer, preds_writer, orig_utterances, model_name, save_dir, writer):
    scorer = Scorer()
    #re-format for acc calculation.
    basic_tokenizer = BasicTokenizer()
    _labels_writer = [' '.join(basic_tokenizer.tokenize(l)) for l in labels_writer]
    _preds_writer = [' '.join(basic_tokenizer.tokenize(p)) for p in preds_writer]
    _orig_utterances = [' '.join(basic_tokenizer.tokenize(o)) for o in orig_utterances]
    bleu_1, bleu_2, bleu_3, bleu_4 = scorer.corpus_bleu_score(references=_labels_writer, predictions=_preds_writer)
    bleu = get_moses_multi_bleu(_preds_writer, _labels_writer, lowercase=True)
    rouge_1, rouge_2, rouge_l = scorer.rouge_score(references=_labels_writer, predictions=_preds_writer)
    p1, r1, f1 = scorer.resolution_score(xs=_preds_writer, refs=_labels_writer, oris=_orig_utterances, ngram=1)
    p2, r2, f2 = scorer.resolution_score(xs=_preds_writer, refs=_labels_writer, oris=_orig_utterances, ngram=2)
    p3, r3, f3 = scorer.resolution_score(xs=_preds_writer, refs=_labels_writer, oris=_orig_utterances, ngram=3)
    df = pd.DataFrame(
                data={
                    'bleu': bleu,
                    'bleu_1': bleu_1,
                    'bleu_2': bleu_2,
                    'bleu_3': bleu_3,
                    'bleu_4': bleu_4,
                    'rouge_1': rouge_1,
                    'rouge_2': rouge_2,
                    'p1': p1,
                    'r1': r1,
                    'f1': f1,
                    'p2': p2,
                    'r2': r2,
                    'f2': f2,
                    'p3': p3,
                    'r3': r3,
                    'f3': f3,
            },
            index=[model_name]
            )
    df_path = os.path.join(save_dir, 'scores.csv')
    df.to_csv(df_path)
    writer.log_metric('bleu_1', bleu_1)
    writer.log_metric('bleu_2', bleu_2)
    writer.log_metric('bleu_3', bleu_3)
    writer.log_metric('bleu_4', bleu_4)
    writer.log_metric('bleu', bleu)
    writer.log_metric('rouge_1', rouge_1)
    writer.log_metric('rouge_2', rouge_2)
    writer.log_metric('rouge_l', rouge_l)
    writer.log_metric('p1', p1)
    writer.log_metric('r1', r1)
    writer.log_metric('f1', f1)
    writer.log_metric('p2', p2)
    writer.log_metric('r2', r2)
    writer.log_metric('f2', f2)
    writer.log_metric('p3', p3)
    writer.log_metric('r3', r3)
    writer.log_metric('f3', f3)



def predict_picker(model, input_ids, src_mask, label_picker):
    logit_picker = model(input_ids, src_mask).permute(0,2,1)
    _, pred_picker = torch.max(logit_picker, 1)
    pred_picker = pred_picker.view(-1).tolist()
    label_picker = label_picker.view(-1).tolist()
    return pred_picker, label_picker


def predict_writer(cfg, model, input_ids, src_mask, labels, tokenizer):#TODO: take cfg
    generated_ids = model.t5.generate(
        input_ids=input_ids,
        attention_mask=src_mask,
        max_length=100,
        num_beams=cfg.training.writer.num_beams,
        early_stopping=True
    )
    pred_writer = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
    label_writer = [tokenizer.decode(l, skip_special_tokens=True, clean_up_tokenization_spaces=True)for l in labels]
    return pred_writer, label_writer

def predict_jointmodel(cfg, model, input_ids, src_mask, labels, tokenizer, label_picker, label_type):
    encoder_outputs = model.t5.encoder(input_ids=input_ids, attention_mask=src_mask, return_dict=True)
    x = encoder_outputs.last_hidden_state
    logit_picker = model.linear(model.ffn(x)).permute(0,2,1)
    generated_ids = model.t5.generate(
        encoder_outputs=encoder_outputs,
        attention_mask=src_mask,
        max_length=100,
        num_beams=cfg.training.jointmodel.num_beams,
        early_stopping=True
    )
    pred_writer = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
    label_writer = [tokenizer.decode(l, skip_special_tokens=True, clean_up_tokenization_spaces=True)for l in labels]
    if label_type=='soft':
        return pred_writer, label_writer

    _, pred_picker = torch.max(logit_picker, 1)
    pred_picker = pred_picker.view(-1).tolist()
    label_picker = label_picker.view(-1).tolist()

    return pred_writer, label_writer, pred_picker, label_picker


def eval_picker(cfg, writer):
    if cfg.dataset.label_type=='soft':
        writer.set_terminated()
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    model = Picker(cfg)
    model_name = get_model_name(cfg)
    fname = f'{model_name}.pt'
    modelpath = os.path.join(cfg.work_dir, cfg.model.ckpts_root, fname)
    checkpoint = torch.load(modelpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    test_iter = load_dataiter(cfg, folds=['train'])
    span_tags, pred_tags = [],[]
    with torch.no_grad():
        for batch in test_iter:
            input_ids = batch['source_ids'].to(device)
            src_mask = batch['source_mask'].to(device)
            label_picker = batch['span_tag'].to(device)
            pred_picker, label_picker = predict_picker(model, input_ids, src_mask, label_picker)
            pred_tags.extend(pred_picker)
            span_tags.extend(label_picker)


    save_dir = os.path.join(cfg.save_dir,model_name)
    os.makedirs(save_dir, exist_ok=True)

    #confusion matrix
    conf_mat(span_tags, pred_tags, save_dir, writer)
    #classification report
    clss_report(span_tags,pred_tags, save_dir, writer)
    writer.set_terminated()



def eval_writer(cfg, writer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    model = Writer(cfg)
    model_name = get_model_name(cfg)
    fname = f'{model_name}.pt'
    modelpath = os.path.join(cfg.work_dir, cfg.model.ckpts_root, fname)
    checkpoint = torch.load(modelpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    tokenizer = load_tokenizer(cfg.dataset)
    test_iter = load_dataiter(cfg, folds=['test'])

    labels_writer, preds_writer, orig_utterances = [],[],[]

    with torch.no_grad():
        for batch in test_iter:
            input_ids = batch['source_ids'].to(device)
            src_mask = batch['source_mask'].to(device)
            labels = batch['target_ids'].to(device)
            query = batch['query']
            pred_writer, label_writer = predict_writer(cfg, model, input_ids, src_mask, labels, tokenizer)
            preds_writer.extend(pred_writer)
            labels_writer.extend(label_writer)
            orig_utterances.extend(query)

    save_dir = os.path.join(cfg.save_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)
    #record of writer's results.
    record_pred(orig_utterances, labels_writer, preds_writer, save_dir, writer)
    restoration_scores(labels_writer, preds_writer, orig_utterances, model_name, save_dir, writer)
    writer.set_terminated()


def eval_jointmodel(cfg, writer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    model = JointModel(cfg)
    model_name = get_model_name(cfg)
    fname = f'{model_name}.pt'
    modelpath = os.path.join(cfg.work_dir, cfg.model.ckpts_root, fname)
    print(modelpath)
    checkpoint = torch.load(modelpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    test_iter = load_dataiter(cfg, folds=['test'])
    tokenizer = load_tokenizer(cfg.dataset)

    span_tags, labels_writer, pred_tags, preds_writer, orig_utterances = [],[],[],[],[]
    label_type = cfg.dataset.label_type
    with torch.no_grad():
        for batch in test_iter:
            input_ids = batch['source_ids'].to(device)
            src_mask = batch['source_mask'].to(device)
            labels = batch['target_ids'].to(device)
            label_picker = batch['span_tag'].to(device)
            query = batch['query']
            if label_type != 'soft':
                pred_writer, label_writer, pred_picker, label_picker\
                                = predict_jointmodel(cfg, model, input_ids, src_mask, labels, tokenizer, label_picker, label_type)

                pred_tags.extend(pred_picker)
                span_tags.extend(label_picker)
                preds_writer.extend(pred_writer)
                labels_writer.extend(label_writer)
                orig_utterances.extend(query)
            else:
                pred_writer, label_writer\
                                = predict_jointmodel(cfg, model, input_ids, src_mask, labels, tokenizer, label_picker, label_type)
                preds_writer.extend(pred_writer)
                labels_writer.extend(label_writer)
                orig_utterances.extend(query)

    save_dir = os.path.join(cfg.save_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)

    if label_type != 'soft':
        #confusion matrix
        conf_mat(span_tags, pred_tags, save_dir, writer)
        #classification report
        clss_report(span_tags,pred_tags, save_dir, writer)

    #record of writer's results.
    record_pred(orig_utterances, labels_writer, preds_writer, save_dir, writer)
    restoration_scores(labels_writer, preds_writer, orig_utterances, model_name, save_dir, writer)
    writer.set_terminated()




if __name__ == "__main__":
    pass
