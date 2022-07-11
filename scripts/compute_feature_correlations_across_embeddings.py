#!/bin/env python
"""
Computes correlations of all pairs of embeddings from two different embedding matrices.
"""
import pandas as pd
from scipy.stats import pearsonr

df1 = pd.read_feather(snakemake.input[0]).set_index('article_id')
df2 = pd.read_feather(snakemake.input[1]).set_index('article_id')

shared_ids = list(set(df1.index).intersection(df2.index))

df1 = df1.loc[shared_ids]
df2 = df2.loc[shared_ids]

# corrwith doesn't scale
# df2.columns = df1.columns
# cor_mat = df1.corrwith(df2, axis=1)

# compare article correlations across embeddings
cors = []

for i in range(df1.shape[1]):
    for j in range(df1.shape[1]):
        cors.append(pearsonr(df1.iloc[:, i], df2.iloc[:, j])[0])

res = pd.DataFrame({
    "embedding1": df1.columns,
    "embedding2": df2.columns,
    "pearson_cor": cors
})

res.to_feather(snakemake.output[0])
