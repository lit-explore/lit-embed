"""
Creates alternate article embeddings based on token statistics, etc.
"""
import os
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer

# assign snakemake to a known variable to prevent excessive mypy, etc. warnings
snek = snakemake

# filtering parameters
MIN_ARTICLE_RATIO:float = snek.config['filtering']['min_article_ratio']
MAX_ARTICLE_RATIO:float = snek.config['filtering']['max_article_ratio']

# embedding dimensions?
DIM:int = snek.config['embedding_dim']

# load article corpus
dat = pd.read_feather(snek.input[0])

dat.title.fillna("", inplace=True)
dat.abstract.fillna("", inplace=True)

# load list of n-gram tokens to detect
with open(snek.config["ngram_input"], "rt", encoding="utf-8") as fp:
    ngrams = fp.read().split("\n")

# total number of articles
N = dat.shape[0]

# get list of article ids
article_ids = dat.id.values.tolist()

TEXT_SOURCE:str = snek.config["text_source"]

# combine lowercase title + abstract to form corpus
corpus = []

for index, article in dat.iterrows():
    # extract target text
    if TEXT_SOURCE == "title":
        doc = article.title.lower()
    elif TEXT_SOURCE == "abstract":
        doc = article.abstract.lower()
    else:
        doc = article.title.lower() + " " + article.abstract.lower()

    # collapse n-gram tokens using underscores, as before
    for ngram in ngrams:
        if ngram in doc:
            doc = doc.replace(ngram, ngram.replace(" ", "_"))

    corpus.append(doc)

del dat

# load token-level stats
tokens = pd.read_parquet(snek.input[1])

# remove tokens above/below cutoffs
min_cutoff = MIN_ARTICLE_RATIO * N
max_cutoff = MAX_ARTICLE_RATIO * N

tokens = tokens[(tokens.num_articles >= min_cutoff) & (tokens.num_articles <= max_cutoff)]

# get top tokens based on simple word frequency
frequency_tokens = tokens.sort_values('n', ascending=False).head(DIM).token.values

# get top tokens based on mean tf-idf and residual idf
tfidf_tokens = tokens.sort_values('mean_tfidf', ascending=False).head(DIM).token.values
ridf_tokens = tokens.sort_values('adj_ridf', ascending=False).head(DIM).token.values

# get top "ensemble" tokens
tokens = tokens.sort_values('mean_tfidf', ascending=False)
tokens['tfidf_rank'] = list(range(1, tokens.shape[0] + 1))

tokens = tokens.sort_values('adj_ridf', ascending=False)
tokens['ridf_rank'] = list(range(1, tokens.shape[0] + 1))

tokens['min_rank'] = tokens[['tfidf_rank', 'ridf_rank']].min(axis=1)
ensemble_tokens = tokens.sort_values('min_rank').head(DIM).token.values

# create token frequency count matrix
vocab = dict(zip(frequency_tokens, range(DIM)))

vectorizer = CountVectorizer(vocabulary=vocab)
mat = vectorizer.fit_transform(corpus)

missed_articles = (mat.sum(axis=1) == 0).sum()
print("# articles with no embedding tokens (Word Frequency): ", missed_articles)

scipy.sparse.save_npz(snakemake.output[0], mat)

# create tf-idf count matrix
vocab = dict(zip(tfidf_tokens, range(DIM)))

vectorizer = CountVectorizer(vocabulary=vocab)
mat = vectorizer.fit_transform(corpus)

missed_articles = (mat.sum(axis=1) == 0).sum()
print("# articles with no embedding tokens (TF-IDF): ", missed_articles)

scipy.sparse.save_npz(snakemake.output[1], mat)

# create residual idf count matrix
vocab = dict(zip(ridf_tokens, range(DIM)))

vectorizer = CountVectorizer(vocabulary=vocab)
mat = vectorizer.fit_transform(corpus)

missed_articles = (mat.sum(axis=1) == 0).sum()
print("# articles with no embedding tokens (residual IDF): ", missed_articles)

scipy.sparse.save_npz(snakemake.output[2], mat)

# create ensemble count matrix
vocab = dict(zip(ensemble_tokens, range(DIM)))

vectorizer = CountVectorizer(vocabulary=vocab)
mat = vectorizer.fit_transform(corpus)

missed_articles = (mat.sum(axis=1) == 0).sum()
print("# articles with no embedding tokens (ensemble): ", missed_articles)

scipy.sparse.save_npz(snakemake.output[3], mat)

# store tokens used for article embeddings
df =  pd.DataFrame({
    "rank": range(1, DIM + 1),
    "frequency": frequency_tokens,
    "tfidf": tfidf_tokens,
    "ridf": ridf_tokens,
    "ensemble": ensemble_tokens
})

df.to_feather(snakemake.output[4])

# store embedding article ids
with open(snakemake.output[5], "w") as fp:
    fp.write("\n".join([str(x) for x in article_ids]))
