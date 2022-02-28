"""
Generates a scatterplot using seaborn
"""
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

random.seed(snakemake.config['random_seed'])

sns.set(rc={
    "figure.figsize": (
        snakemake.config['plots']['scatterplot']['width'],
        snakemake.config['plots']['scatterplot']['height'],
    )
})

# load data
target = snakemake.wildcards['target']

if target == 'articles':
    index = 'article_id'
else:
    index = 'topic'

# add cluster assignments
dat = pd.read_feather(snakemake.input[0]).set_index(index)
clusters = pd.read_feather(snakemake.input[1]).set_index(index)
dat["cluster"] = clusters.cluster.astype("category")

# subsample data
max_points = snakemake.config['plots']['scatterplot']['max_points']

if max_points < dat.shape[0]:
    ind = random.sample(range(dat.shape[0]), max_points)
    dat = dat.iloc[ind]

num_items = dat.shape[0]

# determine plot title to use
if "source" in snakemake.wildcards.keys():
    if snakemake.wildcards['source'] == 'arxiv':
        source = 'arXiv'
    else:
        source = 'Pubmed'
    processing = snakemake.wildcards['processing']
else:
    source = 'Pubmed'
    processing = snakemake.wildcards['agg_func']

name = snakemake.params['name']
projection = snakemake.wildcards['projection']

plt_title = f"{source} {target.capitalize()} {name} {projection} ({processing}, n={num_items})"

size = snakemake.config['plots']['scatterplot']['size']
sns.scatterplot(data=dat, x=dat.columns[0], y=dat.columns[1], hue="cluster", s=size).set(title=plt_title)

plt.savefig(snakemake.output[0])
