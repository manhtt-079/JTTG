import json
from json.tool import main
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import stopwordsiso
import jieba
from functools import partial
from pathlib import Path
import re
from tqdm import tqdm
from itertools import groupby
from itertools import chain

from utils import get_logger



logger = get_logger('master')

additioanl_skip_tokens = ["'s"]
tokenizer = {
    'en': nltk.word_tokenize,
    'zh': jieba.cut
}
tagger = {
    'en': nltk.pos_tag,
    'zh': lambda x: x #TODO: Update
}
stop_words = {
    'en':stopwords.words('english') + additioanl_skip_tokens,
    'zh':list(stopwordsiso.stopwords('zh')) + additioanl_skip_tokens
}
stemmer = {
    'en': PorterStemmer().stem,
    'zh': lambda x:x
}
lemmatizer = {
    'en': partial(WordNetLemmatizer().lemmatize, pos='v'), #pos='v' to convert past to present form for verbs.
    'zh': lambda x:x
}

#For english
TARGET_TAGS = ['NNP','NNPS']
tag_trans = lambda tag: 'TARGET' if tag in TARGET_TAGS else tag
#TODO: For chinese


def get_clue_tokens(datadir, lang):
    """
    extract important tokens for utterance rewriting in huristic strategy.
    """
    def _is_stopwords(token):
        return (token in stop_words[lang]) or (re.fullmatch(r'[^a-zA-Z0-9]+', token))

    def _tokenize(text):
        tokens = list(tokenizer[lang](text))
        tokens = _join(tokens) #group proper noun as 1 token.
        return tokens

    def _join(tokens):
        tagged = tagger[lang](tokens)
        tagged = [(token, tag_trans(tag)) for token, tag in tagged]
        groups = groupby(tagged, key=lambda x: x[1])
        tokens = []
        for tag, words in groups:
            ws = [w for w, _ in words]
            if tag=='TARGET':
                tokens.append(" ".join(ws))
            else:
                tokens.extend(ws)
        return tokens
    def _normalize(tokens):
        sm, lm = stemmer[lang], lemmatizer[lang]
        tokens = [sm(lm(token)) for token in tokens] #lemmatize + stemming as normalization.
        return tokens

    datadir = Path(datadir)
    logger.info(f'processing to get clue token for {datadir.name}.')
    for path in Path(datadir).glob('*.json'):
        data = json.load(open(path))
        for d in tqdm(data):
            d_tokens = sum([_tokenize(i) for i in d['dialog']], []) #sum(list, []) flatten the list
            q_tokens, g_tokens = _tokenize(d['query']), _tokenize(d['gold'])
            d_norms, q_norms, g_norms = _normalize(d_tokens), _normalize(q_tokens), _normalize(g_tokens)
            clue_norms = set(g_norms) - set(q_norms)
            clue_norms = [c for c in clue_norms if not _is_stopwords(c)] #remove stop words
            idxs = [i for i, v in enumerate(d_norms) if v in clue_norms]
            clues = [d_tokens[i] for i in idxs]
            d['d_tokens'] = d_tokens
            d['clue'] = clues
            d['clue_idx'] = idxs
        with open(path, 'wt', encoding='utf8') as fp:
            json.dump(data, fp, ensure_ascii=False, indent=4)
    logger.info('done.')


d={
        "dialog": [
            "Ursula K. Le Guin",
            "Influences",
            "what influenced her?",
            "Le Guin was influenced by fantasy writers,",
            "who were they?",
            "J. R. R. Tolkien, by science fiction writers,",
            "how did they influence her?",
            "her influences, she replied: Once I learned to read, I read everything. I read all the famous fantasies",
            "which other fantasy writer influenced her?",
            "including Philip K. Dick (who was in her high school class,"
        ],
        "query": "how did Philip influence her?",
        "gold": "how did Philip influence Le Guin?",
}




if __name__ == '__main__':
    get_clue_tokens('../data/processed/CANARD_test/', 'en')

    from itertools import chain
    list(chain.from_iterable(x))
    from itertools.chain import from_iterable
    x = [['x',2],[3,4,5],[6,7]]
    sum(x,[])
    [*(i) for i in x]