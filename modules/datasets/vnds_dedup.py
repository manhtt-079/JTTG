import re
import os
import unicodedata
import pandas as pd
from loguru import logger
from typing import Dict, List
from tqdm import tqdm
from dataclasses import dataclass

@dataclass
class Config:
    train_raw_dir: str = '.data/vnds/train_tokenized/'
    val_raw_dir: str = '.data/vnds/val_tokenized/'
    test_raw_dir: str = '.data/vnds/test_tokenized/'

    output_dir: str = './data/vnds/'
    text_col: str = 'text'
    abstract_col: str = 'abstract'
    

VI_RE_MAP: Dict[str, str] = {"òa": "oà", "Òa": "Oà", "ÒA": "OÀ", "óa": "oá", "Óa": "Oá", "ÓA": "OÁ", "ỏa": "oả", "Ỏa": "Oả", "ỎA": "OẢ", 
                 "õa": "oã", "Õa": "Oã", "ÕA": "OÃ", "ọa": "oạ", "Ọa": "Oạ", "ỌA": "OẠ", "òe": "oè", "Òe": "Oè", "ÒE": "OÈ",
                 "óe": "oé", "Óe": "Oé", "ÓE": "OÉ", "ỏe": "oẻ", "Ỏe": "Oẻ", "ỎE": "OẺ", "õe": "oẽ", "Õe": "Oẽ", "ÕE": "OẼ", 
                 "ọe": "oẹ", "Ọe": "Oẹ", "ỌE": "OẸ", "ùy": "uỳ", "Ùy": "Uỳ", "ÙY": "UỲ", "úy": "uý", "Úy": "Uý", "ÚY": "UÝ", 
                 "ủy": "uỷ", "Ủy": "Uỷ", "ỦY": "UỶ", "ũy": "uỹ", "Ũy": "Uỹ", "ŨY": "UỸ", "ụy": "uỵ", "Ụy": "Uỵ", "ỤY": "UỴ"}
    

def clean_text(text):
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(pattern='|'.join(VI_RE_MAP.keys()), repl=lambda m: VI_RE_MAP.get(m.group()), string=text)
    text = re.sub('_', ' ', text)
    text = re.sub(r"\s{1,}", ' ', text)
    
    return text.strip()

def get_file_paths(dir: str):
    return [os.path.join(dir, p) for p in os.listdir(dir)]


def clean_body(text: str) -> str:
    """Remove the last sentence if it contains reporter'name

    Args:
        text (str): body of article

    Returns:
        str: article was filtered
    """
    text = text.strip().split('\n')
    words = text[-1].split(' ')

    if len(words) <= 7:
        return '\n'.join(text[:-1])

    return '\n'.join(text)

def get_data(file_paths: List[str], text_col: str, abstract_col: str):
    data = {
        text_col: [],
        abstract_col: []
    }
    for p in tqdm(file_paths, desc='Get data', total=len(file_paths), ncols=100, nrows=5):
        with open(p) as f:
            text = f.read().split('\n\n')
            body = clean_body(text[2])
            
            abstract = clean_text(text[1])
            text = clean_text(body)
            
            data[abstract_col].append(abstract)
            data[text_col].append(text)
    return data



def main():
    config = Config()
    train_paths = get_file_paths(dir=config.train_raw_dir)
    val_paths = get_file_paths(dir=config.val_raw_dir)
    test_paths = get_file_paths(dir=config.test_raw_dir)
    
    logger.info('Getting train data')
    train_data = get_data(file_paths=train_paths, text_col=config.text_col, abstract_col=config.abstract_col)
    logger.info('Getting val data')
    val_data = get_data(file_paths=val_paths, text_col=config.text_col, abstract_col=config.abstract_col)
    logger.info('Getting test data')
    test_data = get_data(file_paths=test_paths, text_col=config.text_col, abstract_col=config.abstract_col)
    
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)
    is_train = [True]*len(train_df) + [False]*len(val_df) + [False]*len(test_df)
    is_val = [False]*len(train_df) + [True]*len(val_df) + [False]*len(test_df)
    is_test = [False]*len(train_df) + [False]*len(val_df) + [True]*len(test_df)
    
    df = pd.concat([train_df, val_df, test_df])
    df['is_train'] = is_train
    df['is_val'] = is_val
    df['is_test'] = is_test
    
    logger.info('Deduplicating data')
    df_tt = df[(df.is_train) | (df.is_test)].copy(deep=True)
    df_tt.drop_duplicates(subset=[config.text_col, config.abstract_col], keep='last', inplace=True)
    
    df_tv = pd.concat([df_tt[df_tt.is_train], df[df.is_val]])
    df_tv.drop_duplicates(subset=[config.text_col, config.abstract_col], keep='last', inplace=True)
    
    
    df = pd.concat([df_tv[df_tv.is_train], df_tv[df_tv.is_val], df_tt[df_tt.is_test]])
    
    logger.info(f'Saving data to: {os.path.join(config.output_dir, "vnds.csv")}')
    df.to_csv(os.path.join(config.output_dir, 'vnds.csv'), index=False)
        

if __name__=='__main__':
    main()