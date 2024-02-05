import os
import torch
from torch import optim as O
from torch import nn
import numpy as np
import time
from hydra import utils
import mlflow
from mlflow import log_metric, log_param, log_artifacts
import itertools
from transformers import AdamW, T5Tokenizer
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score


from dataloader import load_dataiter
from models import Picker, Writer, JointModel
from utils import MyCriterion, load_tokenizer, get_model_name



def train_picker(cfg, writer):
    def batch2values(batch, device):
        input_ids = batch['source_ids'].to(device)
        src_mask = batch['source_mask'].to(device)
        span_tag = batch['span_tag'].to(device)
        return input_ids, src_mask, span_tag

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Used Device：", device)

    model = Picker(cfg)
    model.to(device)
    torch.backends.cudnn.benchmark = cfg.training.picker.benchmark
    train_iter, dev_iter = load_dataiter(cfg, folds=['train','dev'])
    #weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.training.writer.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.training.writer.lr, eps=cfg.training.writer.eps)

    criterion = MyCriterion(cfg)
    best_dev_loss = float('inf')
    picker_loss = 0.0
    iterations, n_trials, n_updates = 0, 0, 0
    model_name = get_model_name(cfg)
    fname = f'{model_name}.pt'
    savepath = os.path.join(cfg.work_dir, cfg.model.ckpts_root, fname)

    if os.path.exists(savepath) and cfg.training.picker.load_premodel:
        checkpoint = torch.load(savepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_dev_loss = checkpoint['dev_picker_loss']

    log_dir = os.path.join(cfg.log_dir,model_name)
    tb = SummaryWriter(log_dir=log_dir)

    start_time = time.time()
    model.train()
    for epoch in range(cfg.training.picker.num_epochs):
        for batch in train_iter:
            optimizer.zero_grad()
            input_ids, src_mask, span_tag = batch2values(batch, device)
            logit_picker = model(input_ids=input_ids, src_mask=src_mask)
            loss_picker = criterion(logit_picker, span_tag)

            batch_size = input_ids.size(0)
            picker_loss += loss_picker.item()*batch_size
            n_trials += batch_size
            iterations += 1

            loss_picker = loss_picker / cfg.training.writer.accumulation_steps
            loss_picker.backward()
            if iterations % cfg.training.picker.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                n_updates += 1
                if iterations % cfg.training.picker.dev_interval == 0:
                    model.eval()
                    dev_picker_loss = 0.0
                    span_tags, pred_tags = [], []
                    with torch.no_grad():
                        for batch in dev_iter:
                            input_ids, src_mask, span_tag = batch2values(batch, device)
                            logit_picker = model(input_ids=input_ids, src_mask=src_mask)
                            loss_picker = criterion(logit_picker, span_tag)
                            batch_size = input_ids.size(0)
                            dev_picker_loss += loss_picker.item()*batch_size

                            span_tags.extend(span_tag.view(-1).tolist())
                            _, pred_tag = torch.max(logit_picker, 1)
                            pred_tag = pred_tag.view(-1).tolist()
                            pred_tags.extend(pred_tag)


                    picker_loss /= n_trials
                    dev_picker_loss /= len(dev_iter.dataset)

                    elapsed = time.time() - start_time
                    print('Epoch [{}/{}], Step [{}/{}], Elapsed time: {:.3f} min.'
                            .format(epoch+1, cfg.training.picker.num_epochs, iterations, cfg.training.picker.num_epochs*(len(train_iter)), elapsed/60))
                    print('[Train] Picker Loss: {:.4f}, '
                            .format(picker_loss))
                    print('[Dev] Picker Loss: {:.4f}'
                            .format(dev_picker_loss))

                    tb.add_scalar('train_picker_loss', picker_loss, n_updates)
                    tb.add_scalar('dev_picker_loss', dev_picker_loss, n_updates)
                    writer.log_metric('train_picker_loss', picker_loss, step=n_updates)
                    writer.log_metric('dev_picker_loss', dev_picker_loss, step=n_updates)

                    if best_dev_loss > dev_picker_loss:
                        print('The best model is updaded.')
                        best_dev_loss = dev_picker_loss
                        torch.save({
                            'model_state_dict':model.state_dict(),
                            'dev_picker_loss': dev_picker_loss,
                            'train_picker_loss': picker_loss,
                            }, savepath)

                    n_trials = 0
                    picker_loss, dev_picker_loss = 0.0, 0.0
                    model.train()


def train_writer(cfg, writer):
    def batch2values(batch, device):
        input_ids = batch['source_ids'].to(device)
        src_mask = batch['source_mask'].to(device)
        tgt_mask = batch['target_mask'].to(device)
        labels = batch['target_ids'].to(device)
        labels[labels==tokenizer.pad_token_id] = -100 #skip loss caluculation for pad token.
        span_tag = batch['span_tag'].to(device)
        return input_ids, src_mask, tgt_mask, labels, span_tag

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Used Device：", device)

    model = Writer(cfg)
    model.to(device)
    torch.backends.cudnn.benchmark = cfg.training.writer.benchmark
    train_iter, dev_iter = load_dataiter(cfg, folds=['train','dev'])
    tokenizer = load_tokenizer(cfg.dataset)
    #weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.training.writer.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.training.writer.lr, eps=cfg.training.writer.eps)

    best_dev_loss = float('inf')
    writer_loss = 0.0
    iterations, n_trials, n_updates = 0, 0, 0
    model_name = get_model_name(cfg)
    fname = f'{model_name}.pt'
    savepath = os.path.join(cfg.work_dir, cfg.model.ckpts_root, fname)

    if os.path.exists(savepath) and cfg.training.writer.load_premodel:
        checkpoint = torch.load(savepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_dev_loss = checkpoint['dev_writer_loss']

    log_dir = os.path.join(cfg.log_dir,model_name)
    tb = SummaryWriter(log_dir=log_dir)

    start_time = time.time()
    model.train()
    optimizer.zero_grad()
    for epoch in range(cfg.training.writer.num_epochs):
        for batch in train_iter:
            optimizer.zero_grad()
            input_ids, src_mask, tgt_mask, labels, span_tag = batch2values(batch, device)
            loss = model(input_ids=input_ids, src_mask=src_mask, labels=labels, tgt_mask=tgt_mask)

            batch_size = input_ids.size(0)
            writer_loss += loss.item()*batch_size
            n_trials += batch_size
            iterations += 1

            loss = loss / cfg.training.writer.accumulation_steps
            loss.backward()
            if iterations % cfg.training.writer.accumulation_steps == 0:
                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad()
                n_updates += 1
                if iterations % cfg.training.writer.dev_interval == 0:
                    model.eval()
                    dev_writer_loss = 0.0
                    with torch.no_grad():
                        for batch in dev_iter:
                            input_ids, src_mask, tgt_mask, labels, span_tag = batch2values(batch, device)
                            loss = model(input_ids=input_ids, src_mask=src_mask, labels=labels, tgt_mask=tgt_mask)
                            batch_size = input_ids.size(0)
                            dev_writer_loss += loss.item()*batch_size


                    dev_writer_loss /= len(dev_iter.dataset)
                    writer_loss /= n_trials

                    dev_ppl = np.exp(dev_writer_loss)
                    train_ppl = np.exp(writer_loss)

                    elapsed = time.time() - start_time
                    print('Epoch [{}/{}], Step [{}/{}], Elapsed time: {:.3f} min.'
                            .format(epoch+1, cfg.training.writer.num_epochs, iterations, cfg.training.writer.num_epochs*(len(train_iter)), elapsed/60))
                    print('[Train] Writer Loss: {:.4f}, Writer PPL: {:.4f}'
                            .format(writer_loss, train_ppl))
                    print('[Dev] Writer Loss: {:.4f}, Dev PPL: {:.4f}'
                            .format(dev_writer_loss, dev_ppl))

                    tb.add_scalar('dev_writer_loss', dev_writer_loss, n_updates)
                    tb.add_scalar('train_writer_loss', writer_loss, n_updates)
                    tb.add_scalar('train_ppl', train_ppl, n_updates)
                    tb.add_scalar('dev_ppl', dev_ppl, n_updates)

                    writer.log_metric('dev_writer_loss', dev_writer_loss, step=n_updates)
                    writer.log_metric('train_writer_loss', writer_loss, step=n_updates)
                    writer.log_metric('train_ppl', dev_writer_loss, step=n_updates)
                    writer.log_metric('dev_ppl', dev_ppl, step=n_updates)

                    if best_dev_loss > dev_writer_loss:
                        print('The best model is updaded.')
                        best_dev_loss = dev_writer_loss
                        torch.save({
                            'model_state_dict':model.state_dict(),
                            'dev_writer_loss': dev_writer_loss,
                            'train_writer_loss': writer_loss,
                            'train_ppl': train_ppl,
                            'dev_ppl': dev_ppl,
                            }, savepath)

                    n_trials = 0
                    writer_loss, dev_writer_loss = 0.0, 0.0
                    model.train()



def train_jointmodel(cfg, writer): #TODO: input cfg.training.jointmodel in stead of cfg in the future when JointModel is fixed.
    def batch2values(batch, device):
        input_ids = batch['source_ids'].to(device)
        src_mask = batch['source_mask'].to(device)
        tgt_mask = batch['target_mask'].to(device)
        labels = batch['target_ids'].to(device)
        labels[labels==tokenizer.pad_token_id] = -100 #skip loss caluculation for pad token.
        span_tag = batch['span_tag'].to(device)
        return input_ids, src_mask, tgt_mask, labels, span_tag

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Used Device：", device)

    model = JointModel(cfg)
    model.to(device)
    torch.backends.cudnn.benchmark = cfg.training.jointmodel.benchmark
    train_iter, dev_iter = load_dataiter(cfg, folds=['train','dev'])
    tokenizer = load_tokenizer(cfg.dataset)

    # weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.training.jointmodel.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.training.jointmodel.lr, eps=cfg.training.jointmodel.eps)

    criterion = MyCriterion(cfg)

    best_dev_loss = float('inf')
    picker_loss, writer_loss, total_loss = 0.0, 0.0, 0.0
    iterations, n_total, n_updates = 0, 0, 0
    model_name = get_model_name(cfg)
    fname = f'{model_name}.pt'
    savepath = os.path.join(cfg.work_dir, cfg.model.ckpts_root, fname)
    print('=================================')
    print(savepath)
    print('=================================')

    if os.path.exists(savepath) and cfg.training.jointmodel.load_premodel:
        checkpoint = torch.load(savepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_dev_loss = checkpoint['dev_writer_loss']

    log_dir = os.path.join(cfg.log_dir, model_name)
    tb = SummaryWriter(log_dir=log_dir)

    start_time = time.time()
    model.train()
    optimizer.zero_grad()
    for epoch in range(cfg.training.jointmodel.num_epochs):
        for batch in train_iter:
            input_ids, src_mask, tgt_mask, labels, span_tag = batch2values(batch, device)
            logit_picker, loss_writer = model(input_ids=input_ids, src_mask=src_mask, labels=labels, tgt_mask=tgt_mask)
            loss_picker = criterion(logit_picker, span_tag)
            loss = loss_writer + cfg.training.jointmodel.loss_weight*loss_picker
            batch_size = input_ids.size(0)
            picker_loss += loss_picker.item()*batch_size
            writer_loss += loss_writer.item()*batch_size
            total_loss += loss.item()*batch_size
            n_total += batch_size
            iterations += 1

            loss = loss / cfg.training.jointmodel.accumulation_steps
            loss.backward()

            if iterations % cfg.training.jointmodel.accumulation_steps == 0:
                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad()
                n_updates += 1
                if iterations % cfg.training.jointmodel.dev_interval == 0:
                    model.eval()
                    dev_picker_loss, dev_writer_loss, dev_total_loss = 0.0, 0.0, 0.0
                    span_tags, pred_tags = [], []
                    with torch.no_grad():
                        for batch in dev_iter:
                            input_ids, src_mask, tgt_mask, labels, span_tag = batch2values(batch, device)

                            logit_picker, loss_writer = model(input_ids=input_ids, src_mask=src_mask, labels=labels, tgt_mask=tgt_mask)
                            loss_picker = criterion(logit_picker, span_tag)
                            loss = loss_writer + cfg.training.jointmodel.loss_weight*loss_picker

                            batch_size = input_ids.size(0)
                            dev_picker_loss += loss_picker.item()*batch_size
                            dev_writer_loss += loss_writer.item()*batch_size
                            dev_total_loss += loss.item()*batch_size

                            span_tags.extend(span_tag.view(-1).tolist())
                            _, pred_tag = torch.max(logit_picker, 1)
                            pred_tag = pred_tag.view(-1).tolist()
                            pred_tags.extend(pred_tag)

                    dev_picker_loss /= len(dev_iter.dataset)
                    dev_writer_loss /= len(dev_iter.dataset)
                    dev_total_loss /= len(dev_iter.dataset)
                    picker_loss /= n_total
                    writer_loss /= n_total
                    total_loss /= n_total
                    dev_ppl = np.exp(dev_writer_loss)
                    train_ppl = np.exp(writer_loss)


                    elapsed = time.time() - start_time
                    print('Epoch [{}/{}], Step [{}/{}], Elapsed time: {:.3f} min.'
                            .format(epoch+1, cfg.training.jointmodel.num_epochs, iterations, cfg.training.jointmodel.num_epochs*(len(train_iter)), elapsed/60))
                    print('[Train] Total Loss: {:.4f}, Picker Loss: {:.4f}, Writer Loss: {:.4f}. PPL: {:.4f}'
                            .format(total_loss, picker_loss, writer_loss, train_ppl))
                    print('[Dev] Total Loss: {:.4f}, Picker Loss: {:.4f}, Writer Loss: {:.4f}, PPL: {:.4f}'
                            .format(dev_total_loss, dev_picker_loss, dev_writer_loss, dev_ppl))

                    tb.add_scalar('dev_writer_loss', dev_writer_loss, n_updates)
                    tb.add_scalar('train_writer_loss', writer_loss, n_updates)
                    tb.add_scalar('train_ppl', train_ppl, n_updates)
                    tb.add_scalar('dev_ppl', dev_ppl, n_updates)
                    tb.add_scalar('dev_picker_loss', dev_picker_loss, n_updates)
                    tb.add_scalar('train_picker_loss', picker_loss, n_updates)
                    tb.add_scalar('dev_total_loss', dev_total_loss, n_updates)
                    tb.add_scalar('train_total_loss', total_loss, n_updates)

                    writer.log_metric('dev_writer_loss', dev_writer_loss, step=n_updates)
                    writer.log_metric('train_writer_loss', writer_loss, step=n_updates)
                    writer.log_metric('train_ppl', train_ppl, step=n_updates)
                    writer.log_metric('dev_ppl', dev_ppl, step=n_updates)
                    writer.log_metric('dev_picker_loss', dev_picker_loss, step=n_updates)
                    writer.log_metric('train_picker_loss', picker_loss, step=n_updates)
                    writer.log_metric('dev_total_loss', dev_total_loss, step=n_updates)
                    writer.log_metric('train_total_loss', total_loss, step=n_updates)

                    if best_dev_loss > dev_writer_loss: #use dev_writer_loss in stead of dev_total_loss
                        best_dev_loss = dev_writer_loss
                        torch.save({
                            'model_state_dict':model.state_dict(),
                            'dev_writer_loss': dev_writer_loss,
                            'train_writer_loss': writer_loss,
                            'train_ppl': train_ppl,
                            'dev_ppl': dev_ppl,
                            'dev_picker_loss': dev_picker_loss,
                            'train_picker_loss': picker_loss,
                            'dev_total_loss': dev_total_loss,
                            'train_total_loss': total_loss
                            }, savepath)
                        print('The best model is updaded.')

                    n_total = 0
                    dev_picker_loss, dev_writer_loss, dev_total_loss = 0.0, 0.0, 0.0
                    picker_loss, writer_loss, total_loss = 0.0, 0.0, 0.0
                    model.train()
















if __name__ == "__main__":
    pass
