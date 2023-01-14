"""
Computes title, abstract, and article-level token counts + term frequencies
"""
import pandas as pd
import re
from util.nlp import get_stop_words

snek = snakemake

# get list of stop words
STOP_WORDS = get_stop_words(snek.config['processing'] == 'lemmatized')

# minimum token length to include
MIN_TOKEN_LEN = snek.config['tokenization']['min_length']

# target text? (title, abstract, or both)
TEXT_SOURCE = snek.config['text_source']

# load list of n-gram tokens to detect
with open(snek.config["ngram_input"], "rt", encoding="utf-8") as fp:
    ngrams = fp.read().split("\n")

# match all alphanumeric tokens;
regex = re.compile(r"[\w\d_]+", re.UNICODE)

dat = pd.read_feather(snek.input[0]).set_index('id')

rows = []

for article_id, article in dat.iterrows():
    # extract target text
    if TEXT_SOURCE == "title":
        text = article.title.lower()
    elif TEXT_SOURCE == "abstract":
        text = article.abstract.lower()
    else:
        text = article.title.lower() + " " + article.abstract.lower()

    # collapse n-gram tokens using underscores
    for ngram in ngrams:
        if ngram in text:
            text = text.replace(ngram, ngram.replace(" ", "_"))

    # get a list of tokens as they appear in the target text
    tokens = [match.group() for match in regex.finditer(text)]

    # exclude stop words
    tokens = [x for x in tokens if x not in STOP_WORDS]

    # exclude tokens that are below the minimum required len
    tokens = [x for x in tokens if len(x) >= MIN_TOKEN_LEN] 
    
    # total number of tokens remaining
    num_tokens = len(tokens)

    # compute counts of each token
    token_counts = pd.Series(tokens, dtype=str).value_counts()

    # iterate over each unique token and compute article-level stats
    for token in sorted(list(set(tokens))):
        # token counts
        token_count = int(token_counts[token]) if token in token_counts else 0

        # term frequency
        tf = (token_count / num_tokens) if num_tokens > 0 else 0

        rows.append({
            'id': article_id,
            'token': token,
            'n': token_count,
            'tf': tf,
        })

# combine into a single dataframe
res = pd.DataFrame(rows)

# set token dtype to "category" to reduce size
res.token = res.token.astype('category')

# store pubmed article ids as numeric
if snek.config["corpus"] == "pubmed":
    res.id = res.id.astype(int)

res.reset_index(drop=True).to_parquet(snek.output[0])
