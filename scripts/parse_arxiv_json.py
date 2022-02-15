"""
Parses arxiv metadata JSON and extracts article ids and combined title + abstracts
"""
import random
import ujson
import pandas as pd

# load arxiv articles
ids = []
dois = []
titles = []
abstracts = []

with open(snakemake.input[0]) as fp:
    lines = fp.readlines()

# if dev-mode is enabled, subsample articles
if snakemake.config['dev_mode']['enabled']:
    random.seed(snakemake.config['random_seed'])

    random.shuffle(lines)
    lines = lines[:snakemake.config['dev_mode']['num_articles']]

for line in lines:
    article = ujson.loads(line)

    ids.append(article['id'])
    dois.append(article['doi'])
    titles.append(article['title'])
    abstracts.append(article['abstract'])

dat = pd.DataFrame({"id":ids, "doi": dois, "title": titles, "abstract": abstracts })

dat.to_feather(snakemake.output[0])
