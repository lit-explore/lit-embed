"""
Computes global token statistics

Stats computed:

1. Total count
2. Article count
3. Inverse Document Frequency (IDF)
4. Residual IDF (RIDF)
5. Adjusted RIDF

Adjusted RIDF:

By itself, RIDF favors tokens which are mentioned a large amount of times in a very small number of
articles or even a single article.
The effect of this is that those tokens with the highest RIDF scores tend to be tokens found in a
single article, and mentioned more than once.
By reintroducing a weight based on the total number of articles mentioning a token, tokens with such
few article mentions are effectively penalized in the score.
"""
import numpy as np
import pandas as pd
from util.nlp import ridf

df = pd.read_parquet(snakemake.input[0]).set_index("id")

# total number of articles?
N = len(set(df.index))

# total number of token occurrences across all articles?
res = df.groupby('token').sum()

# number of articles including in each token? (document frequency)
res['num_articles'] = df.groupby('token').agg('count')['tf']

# compute token idf
res['idf'] = np.log((1 + N)/ (1 + res['num_articles']))

# convert categorical index to str for faster indexing (~1000x speed-up in testing..)
res = res.reindex(list(res.index))

# add idf to article-level dataframe
df['idf'] = res.loc[df.token, ].idf.values

# compute tf-idf for each token/article
df['tfidf'] = df['tf'] * df['idf']

print("Computing mean TF-IDF..")

# compute token-level mean tf-idf for each token across all articles
res['mean_tfidf'] = df[['token', 'tfidf']].groupby('token').sum() / N

# store article-level tfidf (not used atm, so leaving out to keep things simple..)
# df.head[['token', 'tfidf']].reset_index().to_parquet(snakemake.output[1])

# delete article-level df to free up memory
del df

# compute residual IDF scores for each token
ridfs = []

for token, row in res.iterrows():
    ridfs.append(ridf(row["count"], row["num_articles"], N))

res['ridf'] = ridfs

# Compute adjusted RIDF

# scale to [0, 1]
scaled_ridfs = res.ridf + np.abs(res.ridf.min())
scaled_ridfs = scaled_ridfs / scaled_ridfs.max()

# compute a scaled version of log article counts
article_count_score = np.log(res.num_articles + 1)
article_count_score = article_count_score / article_count_score.max()

# compute adjusted RIDF score and scale to [0, 1]
adj_ridfs = scaled_ridfs * article_count_score
adj_ridfs = adj_ridfs / adj_ridfs.max()

res['adj_ridf'] = adj_ridfs

# sort by frequency and write out
res = res.sort_values('count', ascending=False)

res.reset_index().to_parquet(snakemake.output[0])
