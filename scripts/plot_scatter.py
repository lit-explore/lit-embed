"""
Generates a scatterplot using seaborn
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(rc={"figure.figsize":(16, 12)})

dat = pd.read_feather(snakemake.input[0]).set_index('article_id')

num_articles = dat.shape[0]

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

plt_title = f"{source} Article {name} {projection} ({processing}, n={num_articles})"

sns.scatterplot(data=dat, x=dat.columns[0], y=dat.columns[1], hue="cluster", s=14).set(title=plt_title)

plt.savefig(snakemake.output[0])
