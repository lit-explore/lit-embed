"""
Generates a scatterplot using seaborn
"""
import os
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
target = snakemake.wildcards.target

if target == 'articles':
    index = 'article_id'
else:
    index = 'embedding_column'

# add cluster assignments
try:
    dat = pd.read_feather(snakemake.input[0]).set_index(index)
    clusters = pd.read_feather(snakemake.input[1]).set_index(index)

    dat['cluster'] = clusters.loc[dat.index].cluster
except:
    breakpoint()

# subsample data
max_points = snakemake.config['plots']['scatterplot']['max_points']

if max_points < dat.shape[0]:
    ind = random.sample(range(dat.shape[0]), max_points)
    dat = dat.iloc[ind]

num_items = dat.shape[0]

# data source?
if snakemake.wildcards.source == 'arxiv':
    source = 'arXiv'
else:
    source = 'PubMed'

# determine plot title to use
if os.path.basename(snakemake.input[0]).startswith("bert"):
    if snakemake.wildcards.source == 'pubmed':
        name = "BioBERT"
    else:
        name = "SciBERT"
elif "idf_type" in snakemake.wildcards.keys():
    name = snakemake.wildcards.idf_type
else:
    name = "?"

projection = snakemake.wildcards['projection']

plt_title = f"{source} {target.capitalize()} {name} {projection} ({snakemake.config['processing']}, n={num_items})"

size = snakemake.config['plots']['scatterplot']['size']

sns.scatterplot(data=dat, x=dat.columns[0], y=dat.columns[1], hue="cluster", s=size).set(title=plt_title)

plt.savefig(snakemake.output[0])
