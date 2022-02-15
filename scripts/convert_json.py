"""
Parses arxiv metadata JSON and extracts article ids and combined title + abstracts
"""
import ujson
import pandas as pd

with open(snakemake.input[0], 'r') as fp:
    lines = fp.readlines()

# load arxiv articles
ids = []
corpus = []

with open(snakemake.input[0]) as fp:
    lines = fp.readlines()

for line in lines:
    article = ujson.loads(line)

    ids.append(article['id'])
    corpus.append(article['title'].lower() + " " + article['abstract'].lower())

dat = pd.DataFrame({"id":ids, "text":corpus})

dat.to_feather(snakemake.output[0])
