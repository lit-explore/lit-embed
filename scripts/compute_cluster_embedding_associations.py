#!/bin/env python
"""
Computes associations between BERT embedding clusters and mRIDF selected terms
"""
import pandas as pd

embedding_mat = pd.read_feather(snakemake.input[0]).set_index('article_id')
clusters = pd.read_feather(snakemake.input[1]).set_index('cluster')

clusters = clusters[clusters.article_id.isin(embedding_mat.index)]

rows = []

cluster_ids = sorted(list(set(clusters.index.values)))

for cluster in cluster_ids:
    articles = clusters.loc[cluster].article_id.values

    submat = embedding_mat.loc[articles]

    rows.append(submat.sum())

res = pd.DataFrame(rows)

res.insert(0, "cluster_id", cluster_ids)

# save results
res.to_feather(snakemake.output[0])
