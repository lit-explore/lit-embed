"""
UMAP dimension reduction
"""
import random
import umap
import pandas as pd

random.seed(snakemake.config['random_seed'])

# load data
dat = pd.read_feather(snakemake.input[0]).set_index('article_id')
clusters = pd.read_feather(snakemake.input[1]).set_index('article_id')

# subsample articles
if snakemake.config['umap']['subsample'] < dat.shape[0]:
    ind = random.sample(range(dat.shape[0]), snakemake.config['umap']['subsample'])
    dat = dat.iloc[ind, :]
    clusters = clusters.iloc[ind, :]

# remove any articles with zero variance
mask = dat.var(axis=1) > 0

num_zero_var = dat.shape[0] - mask.sum()

if num_zero_var > 0:
    print(f"Removing {num_zero_var} articles with zero variance prior to UMAP projection.")
    dat = dat[mask]

# generate projection
reducer = umap.UMAP(
        n_neighbors=snakemake.config['umap']['n_neighbors'],
        min_dist=snakemake.config['umap']['min_dist'],
        metric=snakemake.config['umap']['metric'],
        densmap=snakemake.config['umap']['densmap'],
        random_state=snakemake.config['random_seed']
)

embedding = reducer.fit_transform(dat)

# store result
df = pd.DataFrame(embedding, columns=("UMAP1", "UMAP2"), index=dat.index)
df["cluster"] = clusters.cluster.astype("category")

df.reset_index().rename(columns={"index": "article_id"}).to_feather(snakemake.output[0])

