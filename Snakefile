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

# rule all:
#     input:
#         join(config["out_dir"], "embeddings/tfidf.feather"),

# rule compute_tfidf_matrix:
#     input:
#         join(config["input_dir"], "corpus", config["processing"] + ".csv")
#     output:
#         join(config["out_dir"], "embeddings/tfidf.feather"),
#         join(config["out_dir"], "embeddings/tfidf-sparse-mat.npz"),
#         join(config["out_dir"], "embeddings/tfidf-stats.feather"),
#     script: "scripts/compute_tfidf_matrix.py"
#
# rule compute_word_stats:
#     input:
#         join(config["out_dir"], "stats/word-counts.feather"),
#     output:
#         join(config["out_dir"], "stats/word-stats.feather"),
#     script:
#         "scripts/compute_word_stats.py"

# expand(join(config["out_dir"], "stats/batches/word-counts-{batch}.feather"), batch=batches)

rule combine_article_stats:
    input:
        expand(join(config["out_dir"], "stats/articles/{batch}.feather"), batch=batches)
    output:
        join(config["out_dir"], "stats/articles.feather")
    script:
        "scripts/combine_article_stats.py"

rule compute_article_stats:
    input:
        join(batch_dir, "{batch}.feather")
    output:
        join(config["out_dir"], "stats/articles/{batch}.feather")
    script:
        "scripts/compute_article_stats.py"

# vi:syntax=snakemake
