"""
Science literature embedding pipeline
"""
import os

configfile: "config/config.yml"

rule all:
    input:
        os.path.join(config['out_dir'], "models/biobert-v1.1/pytorch_model.bin"),
        os.path.join(config['out_dir'], "data/arxiv/arxiv-word-counts.feather"),
        os.path.join(config['out_dir'], "data/arxiv/arxiv-word-freq.feather"),
        os.path.join(config['out_dir'], "data/arxiv/arxiv-tfidf.feather"),
        os.path.join(config['out_dir'], "data/arxiv/arxiv-tfidf-clusters.feather"),
        os.path.join(config['out_dir'], "data/arxiv/arxiv-biobert-mean.feather"),
        os.path.join(config['out_dir'], "fig/arxiv/arxiv-tfidf-tsne.png"),
        os.path.join(config['out_dir'], "fig/arxiv/arxiv-biobert-mean-tsne.png"),

# if "dev_mode" is enabled, subsample articles
if config['dev_mode']['enabled']:
    arxiv_input = os.path.join(config['out_dir'], "data/arxiv/arxiv-metadata-oai-snapshot-subset.json")
else:
    arxiv_input = os.path.join(config['out_dir'], "data/arxiv/arxiv-metadata-oai-snapshot.json")

rule create_biobert_embeddings:
    input:
        arxiv_input,
        os.path.join(config['out_dir'], "models/biobert-v1.1/pytorch_model.bin")
    output:
        os.path.join(config['out_dir'], "data/arxiv/arxiv-biobert-mean.feather"),
    script:
        "scripts/create_biobert_embeddings.py"

rule plot_article_tfidf_tsne:
    input:
        os.path.join(config['out_dir'], "data/arxiv/arxiv-tfidf.feather"),
        os.path.join(config['out_dir'], "data/arxiv/arxiv-tfidf-clusters.feather"),
    output:
        os.path.join(config['out_dir'], "fig/arxiv/arxiv-tfidf-tsne.png"),
    script:
        "scripts/plot_article_tsne.py"

rule plot_article_biobert_tsne:
    input:
        os.path.join(config['out_dir'], "data/arxiv/arxiv-biobert-mean.feather"),
        os.path.join(config['out_dir'], "data/arxiv/arxiv-tfidf-clusters.feather"),
    output:
        os.path.join(config['out_dir'], "fig/arxiv/arxiv-biobert-mean-tsne.png"),
    script:
        "scripts/plot_article_tsne.py"

rule compute_tfidf_clusters:
    input:
        os.path.join(config['out_dir'], "data/arxiv/arxiv-tfidf.feather"),
    output:
        os.path.join(config['out_dir'], "data/arxiv/arxiv-tfidf-clusters.feather"),
    script: "scripts/compute_tfidf_clusters.py"

rule compute_tfidf_matrix:
    input:
        arxiv_input
    output:
        os.path.join(config['out_dir'], "data/arxiv/arxiv-tfidf.feather"),
    script: "scripts/compute_tfidf_matrix.py"

rule compute_word_counts:
    input:
        os.path.join(config['out_dir'], "data/arxiv/arxiv-word-freq.feather"),
    output:
        os.path.join(config['out_dir'], "data/arxiv/arxiv-word-counts.feather")
    script:
        "scripts/compute_word_counts.py"

rule compute_word_freq_matrix:
    input:
        arxiv_input
    output:
        os.path.join(config['out_dir'], "data/arxiv/arxiv-word-freq.feather"),
    script: "scripts/compute_word_freq_matrix.py"

if config['dev_mode']['enabled']:
    rule select_arxiv_articles:
        input:
            os.path.join(config['out_dir'], "data/arxiv/arxiv-metadata-oai-snapshot.json")
        output:
            os.path.join(config['out_dir'], "data/arxiv/arxiv-metadata-oai-snapshot-subset.json")
        script:
            "scripts/subsample_arxiv_articles.py"

rule download_biobert:
    output:
        os.path.join(config['out_dir'], "models/biobert-v1.1/pytorch_model.bin")
    shell:
        """
        cd `dirname {output}`
        cd ..
        git clone https://huggingface.co/dmis-lab/biobert-v1.1
        """

rule download_arxiv_metadata: 
    output:
        os.path.join(config['out_dir'], "data/arxiv/arxiv-metadata-oai-snapshot.json")
    shell:
        """
        cd `dirname {output}`
        kaggle datasets download -d "Cornell-University/arxiv"
        unzip arxiv.zip
        rm arxiv.zip
        """

