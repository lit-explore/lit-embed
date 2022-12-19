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

# URL and target directory for BERT model
if config["target"] == "pubmed":
    model_dir = join(config["out_dir"], "models", "biobert-v1.1")
    model_url = "https://huggingface.co/allenai/scibert_scivocab_uncased"
else:
    model_dir = join(config["out_dir"], "models", "scibert_scivocab_uncased")
    model_url = "https://huggingface.co/dmis-lab/biobert-v1.1"

rule all:
    input:
        join(config["out_dir"], "embeddings/bert.feather"),
        join(config["out_dir"], "embeddings/tfidf.feather"),

rule compute_tfidf_matrix:
    input:
        join(config["input_dir"], "corpus", config["processing"] + ".csv")
    output:
        join(config["out_dir"], "embeddings/tfidf.feather"),
        join(config["out_dir"], "embeddings/tfidf-sparse-mat.npz"),
        join(config["out_dir"], "embeddings/tfidf-stats.feather"),
    script: "scripts/compute_tfidf_matrix.py"

rule combine_bert_embeddings:
    input:
        expand(join(config["out_dir"], "embeddings/batches/bert/{batch}.feather"), batch=batches),
    output:
        join(config["out_dir"], "embeddings/bert.feather")
    script:
        "scripts/combine_bert_embeddings.py"

rule create_bert_embeddings:
    input:
        join(batch_dir, "{batch}.feather"),
        join(model_dir, "pytorch_model.bin")
    output:
        join(config["out_dir"], "embeddings/batches/bert/{batch}.feather"),
    script:
        "scripts/create_bert_embeddings.py"

rule compute_word_stats:
    input:
        join(config["out_dir"], "stats/word-counts.feather"),
    output:
        join(config["out_dir"], "word-stats/{embedding}.feather"),
    script:
        "scripts/compute_word_stats.py"

rule combine_word_counts:
    input:
        expand(join(config["out_dir"], "stats/batches/word-counts-{batch}.feather"), batch=batches)
    output:
        join(config["out_dir"], "stats/word-counts.feather")
    script:
        "scripts/combine_word_counts.py"

rule compute_word_counts:
    input:
        join(batch_dir, "{batch}.feather")
    output:
        join(config["out_dir"], "stats/batches/word-counts-{batch}.feather")
    script:
        "scripts/create_word_count_matrix.py"

rule download_bert_model:
    params:
        url=model_url
    output:
        join(model_dir, "pytorch_model.bin")
    shell:
        """
        cd `dirname {output}`
        cd ..
        git lfs install
        git clone {params.url}
        """

# vi:syntax=snakemake
