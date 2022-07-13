#!/bin/env python
"""
Computes correlations of all pairs of embeddings from two different embedding matrices.
"""
import pandas as pd
from scipy.stats import pearsonr

df1 = pd.read_feather(snakemake.input[0]).set_index('article_id')
df2 = pd.read_feather(snakemake.input[1]).set_index('article_id')

# normalize article order
shared_ids = list(set(df1.index).intersection(df2.index))

df1 = df1.loc[shared_ids]
df2 = df2.loc[shared_ids]

if (df1.shape[1] != df2.shape[1]):
    print("Error: Embeddings have different numbers of columns!")
    breakpoint()

# corrwith doesn't scale
# df2.columns = df1.columns
# cor_mat = df1.corrwith(df2, axis=1)

# compare article correlations across embeddings
rows = []

for i in range(df1.shape[1]):
    for j in range(df2.shape[1]):
        rows.append({
            "embedding1": df1.columns[i],
            "embedding2": df2.columns[j],
            "pearson_cor": pearsonr(df1.iloc[:, i], df2.iloc[:, j])[0]
        })

res = pd.DataFrame(rows)
res.to_feather(snakemake.output[0])
