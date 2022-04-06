"""
Computes average embedding vectors for each article cluster
"""
import pandas as pd

embedding = pd.read_feather(snakemake.input[0]).set_index('article_id')
clusters = pd.read_feather(snakemake.input[1]).set_index('article_id')

# sanity check
if not (all(embedding.index == clusters.index)):
    raise Exception("Embedding and cluster indices don't match!")

# iterate over clusters and compute average embeddings
cluster_ids = sorted(list(set(clusters.cluster)))

rows = []

for cluster_id in cluster_ids:
    article_ids = clusters[clusters.cluster == cluster_id].index

    rows.append(embedding.loc[article_ids].mean())

# combine averages into dataframe
df = pd.DataFrame(rows)
df.insert(0, "cluster_id", cluster_ids)

df = df.set_index('cluster_id')

# get a table of the most significant terms associated with each cluster
summary_tbl = df.idxmax(axis=1).reset_index()
summary_tbl.columns = ["cluster_id", "top_term"]

# save results
df.reset_index().to_feather(snakemake.output[0])
summary_tbl.to_feather(snakemake.output[1])
