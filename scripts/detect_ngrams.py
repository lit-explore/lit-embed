"""
Script to check corpus against a set of possible n-grams
"""
import pandas as pd

snek = snakemake

# load corpus
#  corpus = pd.read_feather(snek.input[0])

# load article batch
articles = pd.read_feather(snek.input[0])

# load correlated token pairs table
token_pairs = pd.read_feather(snek.input[1])

# list of possible n-grams
ngrams = list((token_pairs.token1 + " " + token_pairs.token2).values)
ngrams = list(set(ngrams + list((token_pairs.token2 + " " + token_pairs.token1).values)))
ngrams = sorted(ngrams)

# create a dict to keep count of all potential ngrams
counters = dict(zip(ngrams, [0] * len(ngrams)))

# iterate over articles and count n-gram matches
TEXT_SOURCE:str = snek.config["text_source"]

for article_id, article in articles.iterrows():
    # extract target text
    if TEXT_SOURCE == "title":
        text = article.title.lower()
    elif TEXT_SOURCE == "abstract":
        text = article.abstract.lower()
    else:
        text = article.title.lower() + " " + article.abstract.lower()

    for ngram in ngrams:
        if ngram in text:
            counters[ngram] += 1

res = pd.DataFrame.from_dict(counters, orient="index").reset_index()
res.columns = ["ngram", "freq"]

res.to_feather(snek.output[0])
