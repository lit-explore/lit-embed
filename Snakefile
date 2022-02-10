"""
Science literature embedding pipeline
"""
import os

configfile: "config/config.yml"

rule all:
    input:
        os.path.join(config['out_dir'], "models/biobert-v1.1/pytorch_model.bin"),
        expand(os.path.join(config['out_dir'], "data/arxiv/topics/{topic}.json"), topic=config['topic_subsets']['topics']),
        os.path.join(config['out_dir'], "data/arxiv/arxiv-word-counts.csv"),
        os.path.join(config['out_dir'], "data/arxiv/arxiv-tfidf.feather"),
        os.path.join(config['out_dir'], "data/arxiv/arxiv-tfidf-clusters.feather"),
        os.path.join(config['out_dir'], "fig/arxiv/arxiv-tfidf-tsne.png"),
        expand(os.path.join(config['out_dir'], "embeddings/arxiv/biobert/{topic}.npz"), topic=config['topic_subsets']['topics'])

rule create_biobert_embeddings:
    input:
        os.path.join(config['out_dir'], "data/arxiv/topics/{topic}.json"),
        os.path.join(config['out_dir'], "models/biobert-v1.1/pytorch_model.bin")
    output:
        os.path.join(config['out_dir'], "embeddings/arxiv/biobert/{topic}.npz")
    script:
        "scripts/create_biobert_embeddings.py"

# if "dev_mode" is enabled, subsample articles
if config['dev_mode']['enabled']:
    arxiv_input = os.path.join(config['out_dir'], "data/arxiv/arxiv-metadata-oai-snapshot-subset.json")
else:
    arxiv_input = os.path.join(config['out_dir'], "data/arxiv/arxiv-metadata-oai-snapshot.json")

rule plot_tfidf_tsne:
    input:
        os.path.join(config['out_dir'], "data/arxiv/arxiv-tfidf.feather"),
        os.path.join(config['out_dir'], "data/arxiv/arxiv-tfidf-clusters.feather"),
    output:
        os.path.join(config['out_dir'], "fig/arxiv/arxiv-tfidf-tsne.png"),
    script:
        "scripts/plot_tfidf_tsne.py"

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
        arxiv_input
    output:
        os.path.join(config['out_dir'], "data/arxiv/arxiv-word-counts.csv")
    script:
        "scripts/compute_word_counts.py"

rule create_topic_subsets:
    input:
        arxiv_input 
    output:
        os.path.join(config['out_dir'], "data/arxiv/topics/{topic}.json")
    script:
        "scripts/create_topic_subsets.py"

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

