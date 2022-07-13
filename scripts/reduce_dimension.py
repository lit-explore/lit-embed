"""
Dimension reduction
"""
import random
import umap
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

random.seed(snakemake.config['random_seed'])

# load data
dat = pd.read_feather(snakemake.input[0]).set_index('article_id')

# determine orientation of projection (articles/topics)
target = snakemake.wildcards['target']
dim_method = snakemake.wildcards['projection']

if target == "articles":
    # subsample articles?
    if snakemake.config[dim_method]['articles']['num'] < dat.shape[0]:
        ind = random.sample(range(dat.shape[0]), snakemake.config[dim_method]['articles']['num'])
        dat = dat.iloc[ind, :]
else:
    # subsample topics?
    if snakemake.config[dim_method]['embedding_columns']['num'] < dat.shape[1]:
        ind = random.sample(range(dat.shape[1]),
                snakemake.config[dim_method]['embedding_columns']['num'])
        dat = dat.iloc[:, ind]

# remove any articles with zero variance (done for both article/topic projections);
mask = dat.apply(np.var, axis=1) > 0

num_zero_var = dat.shape[0] - mask.sum()

if num_zero_var > 0:
    print(f"Removing {num_zero_var} articles with zero variance prior to projection.")
    dat = dat[mask]

# generate projection
if dim_method == 'umap':
    reducer = umap.UMAP(
        n_neighbors=snakemake.config['umap'][target]['n_neighbors'],
        min_dist=snakemake.config['umap'][target]['min_dist'],
        metric=snakemake.config['umap'][target]['metric'],
        densmap=snakemake.config['umap'][target]['densmap'],
        random_state=snakemake.config['random_seed']
    )
elif dim_method == 'tsne':
    reducer = TSNE(n_components=2, perplexity=snakemake.config['tsne']['articles']['perplexity'], init='random', 
                metric=snakemake.config['tsne']['articles']['metric'],
                n_jobs=-1, learning_rate='auto', square_distances=True,
                random_state=snakemake.config['random_seed'])

    # for t-sne + topics, limit number of articles to avoid running out of memory
    if target == 'embedding_columns':

        max_articles = snakemake.config[dim_method]['embedding_columns']['max_articles']

        if max_articles < dat.shape[0]:
            ind = random.sample(range(dat.shape[0]), max_articles)
            dat = dat.iloc[ind, :]

if target == "articles":
    embedding = reducer.fit_transform(dat)
    embedding_index = dat.index
    index_name = "article_id"
else: 
    embedding = reducer.fit_transform(dat.T)
    embedding_index = dat.columns
    index_name = "topic"

# store result
colnames = [dim_method.upper() + str(i) for i in range(1, 3)]

df = pd.DataFrame(embedding, columns=colnames, index=embedding_index)
df.reset_index().rename(columns={"index": index_name}).to_feather(snakemake.output[0])

