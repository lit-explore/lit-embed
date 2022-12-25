"""
Creates an article embedding based on high TF-IDF terms.
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
infile = os.path.join(snek.config["input_dir"], "corpus", snek.config["processing"] + ".feather")
dat = pd.read_feather(infile)

# total number of articles
N = dat.shape[0]

# get list of article ids
article_ids = dat.id.values.tolist()

# combine lowercase title + abstract to form corpus
corpus = []

for index, row in dat.iterrows():
    text = (row.title + " " + row.abstract).lower()    
    corpus.append(text)

del dat

# load token-level stats
tokens = pd.read_parquet(snek.input[0])

# remove tokens above/below cutoffs
min_cutoff = MIN_ARTICLE_RATIO * N
max_cutoff = MAX_ARTICLE_RATIO * N

tokens = tokens[(tokens.num_articles >= min_cutoff) & (tokens.num_articles <= max_cutoff)]

# get top tokens based on simple word frequency
frequency_tokens = tokens.sort_values('count', ascending=False).head(DIM).token.values

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
