"""
Computes token co-occurrence scores for tokens with frequencies above a specified threshold
"""
import random
import pandas as pd

snek = snakemake

# config
RANDOM_SEED:int = snek.config["random_seed"]
MIN_TOKEN_FREQ:int = snek.config["cooccurrence"]["token_min_freq"]
SUBSAMPLE_NUM_ARTICLES:int = snek.config["cooccurrence"]["subsample_num_articles"]

# load token & article dataframes
articles = pd.read_parquet(snek.input[0])
tokens = pd.read_parquet(snek.input[1])

# drop columns not needed
articles = articles.drop(['tf', 'n'], axis=1)

# get a list of tokens that meet the minimum threshold requirements
target_tokens = tokens.token[tokens.n >= MIN_TOKEN_FREQ].values

# limit articles df to the target token subset
articles = articles[articles.token.isin(target_tokens)]

# choose random subset of articles to use for estimating token co-occurrence
random.seed(RANDOM_SEED)

article_ids = list(set(articles.id))

# subsample articles to reduce memory usage, if enabled
if len(article_ids) > SUBSAMPLE_NUM_ARTICLES:
    article_ids = random.sample(article_ids, SUBSAMPLE_NUM_ARTICLES)

    articles = articles[articles.id.isin(article_ids)]

# drop unused category levels
articles.token = articles.token.cat.remove_unused_categories()

# create dense binary co-occurrence matrix
articles["presence"] = True
mat = articles.pivot_table(index="id", columns="token", values="presence", fill_value=False)

# fix type for resulting wide matrix (columns get converted to floats as a result of missing values)
mat = mat.astype(bool)

# save result
mat.reset_index().to_feather(snek.output[0])
