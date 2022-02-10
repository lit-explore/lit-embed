"""
Generates a t-SNE plot from the article TF-IDF matrix + cluster assignments

- [ ] add config opt to enable sub-sampling when num articles is very large
"""
import random
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

sns.set(rc={"figure.figsize":(16, 12)})

# set random seed used for sub-sampling
random.seed(snakemake.config['random_seed'])

dat = pd.read_feather(snakemake.input[0]).set_index('article_id')
clusters = pd.read_feather(snakemake.input[1]).set_index('article_id')

# subsample to speed things up
ind = random.sample(range(dat.shape[0]), snakemake.config['tsne']['subsample'])

dat = dat.iloc[ind]
clusters = clusters.iloc[ind]

tsne = TSNE(n_components=2, perplexity=30.0, init='random', metric='cosine', 
            n_jobs=-1, learning_rate='auto', square_distances=True,
            random_state=snakemake.config['random_seed'])

tsne_dat = pd.DataFrame(tsne.fit_transform(dat), columns=['TSNE1', 'TSNE2'])
tsne_dat.index = dat.index

# add cluster information
tsne_dat = pd.concat([tsne_dat, clusters], axis=1)

sns.scatterplot(data=tsne_dat, x="TSNE1", y="TSNE2", hue="cluster", s=14)

plt.savefig(snakemake.output[0])
