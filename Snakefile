"""
lit-embed: Scientific literature embedding pipeline
https://github.com/lit-explore/lit-embed
"""
import os
import pandas as pd
from os.path import join

# directory with batched article data
batch_dir = join(config["input_dir"], config["processing"])

# get list of article batch ids
batches = [os.path.splitext(x)[0] for x in os.listdir(batch_dir)]

# limit to requested range
batch_range = range(config['batches']['start'], config['batches']['end'] + 1)
batches_allowed = [f"{x:04}" for x in batch_range]

batches = [x for x in batches if x in batches_allowed]

if len(batches) == 0:
    raise Exception("No input batches found!")

# rule all:
#     input:
#         join(config["out_dir"], "embeddings/tfidf.feather"),

rule create_tfidf_embeddings:
    input:
        join(config["out_dir"], "stats/tokens.parquet"),
        join(config["out_dir"], "stats/articles.parquet")
    output:
        join(config["out_dir"], "embeddings/tfidf.feather"),
    script: "scripts/create_tfidf_embeddings.py"

rule compute_token_stats:
    input:
        join(config["out_dir"], "stats/articles.parquet")
    output:
        join(config["out_dir"], "stats/tokens.parquet"),
    script:
        "scripts/compute_token_stats.py"

rule combine_article_stats:
    input:
        expand(join(config["out_dir"], "stats/articles/{batch}.parquet"), batch=batches)
    output:
        join(config["out_dir"], "stats/articles.parquet")
    script:
        "scripts/combine_article_stats.py"

rule compute_article_stats:
    input:
        join(batch_dir, "{batch}.feather")
    output:
        join(config["out_dir"], "stats/articles/{batch}.parquet")
    script:
        "scripts/compute_article_stats.py"

# vi:syntax=snakemake
