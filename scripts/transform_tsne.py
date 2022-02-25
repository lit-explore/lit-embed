"""
t-SNE dimension reduction
"""
import random
import pandas as pd
from sklearn.manifold import TSNE

# set random seed used for sub-sampling
random.seed(snakemake.config['random_seed'])

dat = pd.read_feather(snakemake.input[0]).set_index('article_id')
clusters = pd.read_feather(snakemake.input[1]).set_index('article_id')

# subsample to speed things up
if snakemake.config['tsne']['subsample'] < dat.shape[0]:
    ind = random.sample(range(dat.shape[0]), snakemake.config['tsne']['subsample'])
    dat = dat.iloc[ind]
    clusters = clusters.iloc[ind]

num_articles = dat.shape[0]

tsne = TSNE(n_components=2, perplexity=snakemake.config['tsne']['perplexity'], init='random', 
            metric=snakemake.config['tsne']['metric'],
            n_jobs=-1, learning_rate='auto', square_distances=True,
            random_state=snakemake.config['random_seed'])

embedding = tsne.fit_transform(dat)

# store result
df = pd.DataFrame(embedding, columns=['TSNE1', 'TSNE2'], index=dat.index)
df["cluster"] = clusters.cluster.astype("category")

df.reset_index().rename(columns={"index": "article_id"}).to_feather(snakemake.output[0])

