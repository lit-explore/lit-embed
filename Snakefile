"""
lit-embed: Scientific literature embedding pipeline
https://github.com/lit-explore/lit-embed
"""
import os
import pandas as pd
from os.path import join

# get list of article batch ids
batch_dir = join(config["input_dir"], config["processing"])
batches = [os.path.splitext(x)[0] for x in os.listdir(batch_dir)]

# limit to requested range
batch_range = range(config['batches']['start'], config['batches']['end'] + 1)
batches_allowed = [f"{x:04}" for x in batch_range]

batches = [x for x in batches if x in batches_allowed]

if len(batches) == 0:
    raise Exception("No input batches found!")

rule all:
    input:
        join(config["out_dir"], "embeddings/ensemble.npz"),
        join(config["out_dir"], "ngrams/counts.feather"),
        join(config["out_dir"], "embeddings/bert.parquet"),

rule combine_bert_embeddings:
    input:
        expand(join(config["out_dir"], "bert/articles/{batch}.parquet"), batch=batches)
    output:
        join(config["out_dir"], "embeddings/bert.parquet"),
    script:
        "scripts/combine_bert_embeddings.py"

rule create_bert_embeddings:
    input:
        join(batch_dir, "{batch}.feather")
    output:
        join(config["out_dir"], "bert/articles/{batch}.parquet")
    resources:
        load=60
    script:
        "scripts/create_bert_embeddings.py"

rule create_stat_embeddings:
    input:
        join(config["input_dir"], "corpus", config["processing"] + ".feather"),
        join(config["out_dir"], "stats/tokens.parquet"),
    output:
        join(config["out_dir"], "embeddings/frequency.npz"),
        join(config["out_dir"], "embeddings/tfidf.npz"),
        join(config["out_dir"], "embeddings/ridf.npz"),
        join(config["out_dir"], "embeddings/ensemble.npz"),
        join(config["out_dir"], "embeddings/embedding_tokens.feather"),
        join(config["out_dir"], "embeddings/article_ids.txt"),
    script: "scripts/create_stat_embeddings.py"

rule combine_ngrams:
    input:
        expand(join(config["out_dir"], "ngrams/counts/{batch}.feather"), batch=batches)
    output:
        join(config["out_dir"], "ngrams/counts.feather"),
        join(config["out_dir"], "ngrams/ngrams.txt")
    script:
        "scripts/combine_ngrams.py"

rule detect_ngrams:
    input:
        join(batch_dir, "{batch}.feather"),
        join(config["out_dir"], "stats/correlated-token-pairs.feather"),
    output:
        join(config["out_dir"], "ngrams/counts/{batch}.feather")
    script:
        "scripts/detect_ngrams.py"

rule detect_correlated_token_pairs:
    input:
        join(config["out_dir"], "stats/token-correlations.feather")
    output:
        join(config["out_dir"], "stats/correlated-token-pairs.feather")
    script:
        "scripts/detect_correlated_token_pairs.py"

rule compute_token_correlations:
    input:
        join(config["out_dir"], "stats/co-occurrence.feather")
    output:
        join(config["out_dir"], "stats/token-correlations.feather")
    script:
        "scripts/compute_token_correlations.py"

rule compute_token_cooccurence:
    input:
        join(config["out_dir"], "stats/articles.parquet"),
        join(config["out_dir"], "stats/tokens.parquet")
    output:
        join(config["out_dir"], "stats/co-occurrence.feather")
    script:
        "scripts/compute_token_cooccurrence.py"

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
