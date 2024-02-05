import argparse
import json
from pathlib import Path
import random
import shutil
import os



fold_map = {
    'train':'train',
    'valid':'dev',
    'test':'test',
    }


def Restoration200k():
    dataroot = Path('./data/')
    fs_src = sorted((dataroot/'raw'/'Restoration-200K').glob('**/*.sr'), key=lambda x:x.name)
    fs_tgt = sorted((dataroot/'raw'/'Restoration-200K').glob('**/*.tr'), key=lambda x:x.name)
    for f_src, f_tgt in zip(fs_src, fs_tgt):
        data = []
        data_src, data_tgt = open(f_src, encoding='utf-8').readlines(), open(f_tgt, encoding='utf-8').readlines()
        data_src = [d for d in data_src if d!=' || |\n'] #remove wrong data which exist in test data only.
        for d_src, d_tgt in zip(data_src, data_tgt):
            if 'S-N S-N S-N S-N S-N S-N S-N' in d_src:
                continue
            d_src = d_src.split('||') # [dialog, query|token] or [dialog, query]
            dialog = d_src[0].replace(' ','').split('<split>')
            # if '|' not in d_src[1]: # skip strange 1 sample. (S-N S-N S-N S-N S-N S-N S-N)
            #     continue
            query = d_src[1].split('|')[0].replace(' ','')
            token = d_src[1].split('|')[1].replace(' ','').strip()
            token = [token] if token!='' else []
            gold = d_tgt.replace(' ','').strip()
            data.append({
                'dialog': dialog,
                'query': query,
                'gold': gold,
                'token': token
                })

        fname = fold_map[f_src.stem] + '.json'
        save_dir = dataroot/f'processed/Restoration200k_ratio=1'
        save_dir.mkdir(exist_ok=True, parents=True)
        with open(save_dir/f'{fname}', 'wt', encoding='utf8') as fp:
            json.dump(data, fp, ensure_ascii=False, indent=4)



def CANARD():
    dataroot = Path('./data/')
    fs = [dataroot/f'raw/CANARD_Release/{fold}.json' for fold in ['test', 'train', 'dev']]
    save_dir = dataroot/'processed/CANARD_ratio=1'
    save_dir.mkdir(exist_ok=True, parents=True)
    for f in fs:
        data = json.load(open(f))
        for d in data:
            #rename key names
            d['dialog'] = d.pop('History')
            d['query'] = d.pop('Question')
            d['gold'] = d.pop('Rewrite')
            d['token'] = []
        with open(save_dir/f'{f.name}', 'wt', encoding='utf8') as fp:
            json.dump(data, fp, ensure_ascii=False, indent=4)

def TASK():
    dataroot = Path('./data/')
    path_raw = dataroot/'raw'/'TASK'/'CamRest676_annotated.json'
    f = open(path_raw, encoding='utf8')
    ff = json.load(f)

    dial = []
    label = []
    for i in ff:
        dial_temp = []
        label_temp = []
        for j in i['dial']:
            dial_temp.append(j['usr']['transcript'])
            dial_temp.append(j['sys']['sent'])
            label_temp.append(j['usr']['transcript_complete'])
        dial.append(dial_temp)
        label.append(label_temp)

    dials = []
    labels = []
    for i, k in zip(dial, label):
        count = 0
        temp_dial = None
        temp_label =None
        for j in range(0, len(i), 2):
            if j==0:
                temp_dial = [i[0]]
            else:
                temp_dial = i[:j+1]
            temp_label = k[count]
            count+=1
            dials.append(temp_dial)
            labels.append(temp_label)

    dataset = []
    for i,j in zip(dials, labels):
        temp = {}
        if len(i)==1:
            temp['dialog'] = i[0]
            temp['query'] = i[0]
        else:
            temp['dialog'] = i[:-1]
            temp['query'] = i[-1]
        temp['gold'] = j
        temp['token'] = []
        dataset.append(temp)

    train, test, dev = dataset[:2200], dataset[2200:], dataset[2200:]

    save_dir = Path('./data/processed/TASK_ratio=1')
    save_dir.mkdir(exist_ok=True, parents=True)

    with open(save_dir/'train.json', 'wt', encoding='utf8') as fp:
        json.dump(train, fp, ensure_ascii=False, indent=4)

    with open(save_dir/'test.json', 'wt', encoding='utf8') as fp:
        json.dump(test, fp, ensure_ascii=False, indent=4)

    with open(save_dir/'dev.json', 'wt', encoding='utf8') as fp:
        json.dump(dev, fp, ensure_ascii=False, indent=4)


def Rewrite20k():
    dataroot = Path('./data/')
    path_raw = dataroot/'raw'/'Rewrite20k'/'rewrite.txt'
    save_dir = dataroot/'processed'/'Rewrite20k_ratio=1'
    save_dir.mkdir(exist_ok=True, parents=True)
    outs = []
    rawdata = open(path_raw, encoding='utf8').readlines()
    data_dict = {
        'train': rawdata[18000:],
        'dev': rawdata[:18000],
        'test': rawdata[:18000],
    }

    for fold, data in data_dict.items():
        for d in data:
            out = {}
            out['dialog'] = d.split('\t\t')[:2]
            out['query'] = d.split('\t\t')[2]
            out['gold'] = d.split('\t\t')[-1].strip()
            out['token'] = []
            outs.append(out)

        with open(save_dir/f'{fold}.json', 'wt', encoding='utf8') as fp:
            json.dump(outs, fp, ensure_ascii=False, indent=4)


def gen_limited_data(dataset):
    SEED = 42
    ratio=0.1
    random.seed(SEED)
    droot = Path('./data/processed')
    new_dir = droot/f'{dataset}_ratio={ratio}'
    if not os.path.exists(new_dir):
        shutil.copytree(droot/f'{dataset}_ratio=1', new_dir)
    random.seed(SEED)
    path = new_dir/'train.json'
    d_train = json.load(open(path))
    sample_size = round(len(d_train)*ratio)
    d_train = random.sample(d_train, sample_size)
    with open(path, 'wt', encoding='utf8') as f:
        json.dump(d_train, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datasets',
        nargs='*',
        choices=['Restoration200k','CANARD','TASK','Rewrite20k'],
        default=['Restoration200k','CANARD','TASK','Rewrite20k'])
    args = parser.parse_args()
    for dataset in args.datasets:
        eval(dataset)()
        gen_limited_data(dataset)
        print(f'finished transform for {dataset}')