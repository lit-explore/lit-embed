#!/bin/python
"""
Computes word usage statistics across all articles
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

# compute idf (~rarity) and residual idf (~informativeness)
idfs = []
ridfs = []

for token, row in res.iterrows():
    idfs.append(np.log2(N / row.num_articles))
    ridfs.append(ridf(row.total_count, row.num_articles, N))

res['idf'] = idfs
res['ridf'] = ridfs

# sort by frequency and write out
res = res.sort_values('total_count', ascending=False)

res.reset_index().to_feather(snakemake.output[0])

