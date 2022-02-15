"""
Science literature embedding pipeline
"""
import os

configfile: "config/config.yml"

# pubmed annual file numbers (2022)
pubmed_annual = [f"{n:04}" for n in range(1, 1115)]

# pubmed daily update file numbers
daily_start = config['pubmed']['updates_start']
daily_end = config['pubmed']['updates_end']

pubmed_daily = [f"{n:04}" for n in range(daily_start, daily_end + 1)]

# if "dev_mode" is enabled, subsample articles
if config['dev_mode']['enabled']:
    arxiv_input = os.path.join(config['out_dir'], "data/arxiv/arxiv-articles-subset.feather")
    arxiv_lemmatized = os.path.join(config['out_dir'], "data/arxiv/arxiv-articles-lemma-subset.feather")
else:
    pubmed_input = os.path.join(config['out_dir'], "data/pubmed/pubmed-articles.feather"),
    arxiv_input = os.path.join(config['out_dir'], "data/arxiv/arxiv-articles.feather")
    arxiv_lemmatized = os.path.join(config['out_dir'], "data/arxiv/arxiv-articles-lemma.feather")

rule all:
    input:
        arxiv_input,
        arxiv_lemmatized,
        os.path.join(config['out_dir'], "models/biobert-v1.1/pytorch_model.bin"),
        os.path.join(config['out_dir'], "fig/arxiv/arxiv-lemma-tfidf-tsne.png"),
        # pubmed_input,
        # os.path.join(config['out_dir'], "data/arxiv/arxiv-tfidf.feather"),
        # os.path.join(config['out_dir'], "data/arxiv/arxiv-tfidf-clusters.feather"),
        # os.path.join(config['out_dir'], "data/arxiv/arxiv-word-counts.feather"),
        # os.path.join(config['out_dir'], "data/arxiv/arxiv-word-freq.feather"),
        # os.path.join(config['out_dir'], "data/arxiv/arxiv-biobert-mean.feather"),
        # os.path.join(config['out_dir'], "fig/arxiv/arxiv-biobert-mean-tsne.png"),


rule create_lemmatized_corpus:
    input:
        arxiv_input
    output:
        arxiv_lemmatized
    script:
        "scripts/lemmatize_text.py"

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
        os.path.join(config['out_dir'], "data/arxiv/arxiv-lemma-tfidf.feather"),
        os.path.join(config['out_dir'], "data/arxiv/arxiv-lemma-tfidf-clusters.feather"),
    output:
        os.path.join(config['out_dir'], "fig/arxiv/arxiv-lemma-tfidf-tsne.png"),
    params:
        title="arXiv Article TF-IDF t-SNE"
    script:
        "scripts/plot_article_tsne.py"

rule plot_article_biobert_tsne:
    input:
        os.path.join(config['out_dir'], "data/arxiv/arxiv-biobert-mean.feather"),
        os.path.join(config['out_dir'], "data/arxiv/arxiv-tfidf-clusters.feather"),
    output:
        os.path.join(config['out_dir'], "fig/arxiv/arxiv-biobert-mean-tsne.png"),
    params:
        title="arXiv Article BioBERT Mean Embedding t-SNE"
    script:
        "scripts/plot_article_tsne.py"

rule compute_tfidf_clusters:
    input:
        os.path.join(config['out_dir'], "data/arxiv/arxiv-lemma-tfidf.feather"),
    output:
        os.path.join(config['out_dir'], "data/arxiv/arxiv-lemma-tfidf-clusters.feather"),
    script: "scripts/compute_tfidf_clusters.py"

rule compute_tfidf_matrix:
    input:
       arxiv_lemmatized 
    output:
        os.path.join(config['out_dir'], "data/arxiv/arxiv-lemma-tfidf.feather"),
    script: "scripts/compute_tfidf_matrix.py"

rule compute_arxiv_word_counts:
    input:
        os.path.join(config['out_dir'], "data/arxiv/arxiv-word-freq.feather"),
    output:
        os.path.join(config['out_dir'], "data/arxiv/arxiv-word-counts.feather")
    script:
        "scripts/compute_word_counts.py"

rule compute_arxiv_word_freq_matrix:
    input:
        arxiv_input
    output:
        os.path.join(config['out_dir'], "data/arxiv/arxiv-word-freq.feather"),
    script: "scripts/compute_word_freq_matrix.py"

if config['dev_mode']['enabled']:
    rule select_arxiv_articles:
        input:
            os.path.join(config['out_dir'], "data/arxiv/arxiv-articles.feather")
        output:
            os.path.join(config['out_dir'], "data/arxiv/arxiv-articles-subset.feather")
        script:
            "scripts/subsample_articles.py"

    rule select_pubmed_articles:
        input:
            os.path.join(config['out_dir'], "data/pubmed/pubmed-articles.feather")
        output:
            os.path.join(config['out_dir'], "data/pubmed/pubmed-articles-subset.feather")
        script:
            "scripts/subsample_articles.py"

rule convert_json:
    input:
        os.path.join(config['out_dir'], "data/arxiv/arxiv-metadata-oai-snapshot.json")
    output:
        os.path.join(config['out_dir'], "data/arxiv/arxiv-articles.feather")
    script:
        "scripts/convert_json.py"
    
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

# rule download_daily_updates:
#     output:
#         os.path.join(config['out_dir'], "data/pubmed/updatefiles/pubmed22n{pubmed_update_num}.xml.gz")
#     shell:
#         """
#         cd `dirname {output}`
#
#         for i in $(seq -w 0001 1114); do
#             echo $i
#             wget "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed22n${i}.xml.gz"
#         done
#         """

rule combine_pubmed_data:
    input:
        expand(os.path.join(config['out_dir'], "data/pubmed/baseline/feather/pubmed22n{pubmed_baseline_num}.feather"), pubmed_baseline_num=pubmed_annual)
    output:
        os.path.join(config['out_dir'], "data/pubmed/pubmed-articles.feather")

rule convert_pubmed_baseline_data:
    input: 
        os.path.join(config['out_dir'], "data/pubmed/baseline/xml/pubmed22n{pubmed_baseline_num}.xml.gz")
    output:
        os.path.join(config['out_dir'], "data/pubmed/baseline/feather/pubmed22n{pubmed_baseline_num}.feather")
    script:
        "scripts/parse_pubmed_xml.py"

rule download_pubmed_baseline_data:
    output:
        os.path.join(config['out_dir'], "data/pubmed/baseline/xml/pubmed22n{pubmed_baseline_num}.xml.gz")
    shell:
        """
        cd `dirname {output}`
        wget "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed22n${i}.xml.gz"
        """
