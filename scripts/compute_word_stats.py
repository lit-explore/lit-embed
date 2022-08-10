#!/bin/python
"""
Computes word usage statistics across all articles

1. Total count of a word across all articles
2. Number of articles mentioning a word at least once
3. Inverse Document Frequency (IDF)
4. Residual IDF (RIDF)
5. Moderated RIDF
"""
import numpy as np
import pandas as pd
from util.nlp import ridf

df = pd.read_feather(snakemake.input[0]).set_index("id")

df['total_count'] = df['title_count'] + df['abstract_count']

# total number of word occurrences across all articles? (by title, abstract, and both)
res = df.groupby('token').sum()

# number of articles including in each word? (document frequency)
res['num_articles'] = df.groupby('token').agg('count').title_count

# total number of articles?
N = len(set(df.index))

# lists to store IDF and residual IDF scores
idfs = []
ridfs = []

for token, row in res.iterrows():
    idfs.append(np.log2(N / row.num_articles))
    ridfs.append(ridf(row.total_count, row.num_articles, N))

res['idf'] = idfs
res['ridf'] = ridfs

# compute modified version of RIDF, which is adjusted for the number of articles
# mentioning a word.
# by itself, RIDF favors words which are mentioned a large amount of times in a very
# small number of articles or even a single article.
# the effect of this is that those words with the highest RIDF scores tend to be words
# found in a single article, and mentioned more than once.
# by reintroducing a weight based on the log of the total number of articles mentioning a
# word, words with such few article mentions are effectively penalized in the score.

# scale to [0, 1]
scaled_ridfs = res.ridf + np.abs(res.ridf.min())
scaled_ridfs = scaled_ridfs / scaled_ridfs.max()

# compute a scaled version of log article counts
article_count_score = np.log10(res.num_articles + 1)
article_count_score = article_count_score / article_count_score.max()

# compute modified RIDF score and scale to [0, 1]
mridfs = scaled_ridfs * article_count_score

mridfs = mridfs + np.abs(mridfs.min())
mridfs = mridfs / mridfs.max()

res['mridf'] = mridfs

res.sort_values('mridf', ascending=False)

# sort by frequency and write out
res = res.sort_values('total_count', ascending=False)

res.reset_index().to_feather(snakemake.output[0])

