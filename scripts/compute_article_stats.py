"""
Computes title, abstract, and article-level token counts + term frequencies
"""
import pandas as pd
import re
from util.nlp import get_stop_words

# get list of stop words
STOP_WORDS = get_stop_words(snakemake.config['processing'] == 'lemmatized')

# minimum token length to include
MIN_TOKEN_LEN = snakemake.config['tokenization']['min_length']

# target text? (title, abstract, or both)
TARGET = snakemake.config['target']

# match all alphanumeric tokens;
regex = re.compile(r"[\w\d]+", re.UNICODE)

dat = pd.read_feather(snakemake.input[0]).set_index('id')

rows = []

for article_id, article in dat.iterrows():
    # extract target text
    if TARGET == "title":
        text = article.title.lower()
    elif TARGET == "abstract":
        text = article.abstract.lower()
    else:
        text = article.title.lower() + " " + article.abstract.lower()

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
            'id': int(article_id),
            'token': token,
            'count': token_count,
            'tf': tf,
        })

# combine into a single dataframe
res = pd.DataFrame(rows)

# filter low count tokens
#  token_counts = res.groupby('token').total_count.agg(sum)

#  to_keep = token_counts[token_counts >= snakemake.config["filtering"]["batch_min_count"]].index
#  res = res[res.token.isin(to_keep)]

# set token type to "category" to reduce size
res.token = res.token.astype('category')

res.reset_index(drop=True).to_parquet(snakemake.output[0])
