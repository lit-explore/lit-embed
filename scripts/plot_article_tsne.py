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
num_articles = snakemake.config['tsne']['subsample']

ind = random.sample(range(dat.shape[0]), num_articles)

dat = dat.iloc[ind]
clusters = clusters.iloc[ind]

tsne = TSNE(n_components=2, perplexity=30.0, init='random', metric='cosine', 
            n_jobs=-1, learning_rate='auto', square_distances=True,
            random_state=snakemake.config['random_seed'])

tsne_dat = pd.DataFrame(tsne.fit_transform(dat), columns=['TSNE1', 'TSNE2'])
tsne_dat.index = dat.index

# add cluster information
tsne_dat = pd.concat([tsne_dat, clusters], axis=1)

if snakemake.wildcards['source'] == 'arxiv':
    source = 'arXiv'
else:
    source = 'Pubmed'

plt_title = f"{source} Article {snakemake.params['title']} t-SNE (n={num_articles})"

sns.scatterplot(data=tsne_dat, x="TSNE1", y="TSNE2", hue="cluster", s=14).set(title=plt_title)

plt.savefig(snakemake.output[0])
