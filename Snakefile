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

model_dir = os.path.join(config['out_dir'], 'models')
input_dir = os.path.join(config['out_dir'], 'input')

# if "dev_mode" is enabled, operate on a subset of articles
if config['dev_mode']['enabled']:
    data_dir = os.path.join(config['out_dir'], 'subset')
else:
    data_dir = os.path.join(config['out_dir'], 'full')

data_sources = ['pubmed', 'arxiv']

rule all:
    input:
        expand(os.path.join(data_dir, "data/{source}/articles-lemmatized.feather"), source=data_sources),
        expand(os.path.join(data_dir, "fig/{source}/article-tfidf-tsne.png"), source=data_sources),
        expand(os.path.join(data_dir, "fig/{source}/article-tfidf-lemmatized-tsne.png"), source=data_sources),
        expand(os.path.join(data_dir, "data/{source}/word-counts.feather"), source=data_sources),
        os.path.join(model_dir, "biobert-v1.1/pytorch_model.bin"),
        # os.path.join(data_dir, "data/arxiv/biobert-mean.feather"),
        # os.path.join(data_dir, "fig/arxiv/biobert-mean-tsne.png"),

rule create_lemmatized_corpus:
    input:
       os.path.join(data_dir, "data/{source}/articles.feather") 
    output:
       os.path.join(data_dir, "data/{source}/articles-lemmatized.feather") 
    script:
        "scripts/lemmatize_text.py"

rule create_biobert_embeddings:
    input:
        os.path.join(data_dir, "data/{source}/articles.feather"),
        os.path.join(model_dir, "biobert-v1.1/pytorch_model.bin")
    output:
        os.path.join(data_dir, "data/{source}/biobert-mean.feather"),
    script:
        "scripts/create_biobert_embeddings.py"

rule plot_article_tfidf_lemmatized_tsne:
    input:
        os.path.join(data_dir, "data/{source}/tfidf-lemmatized.feather"),
        os.path.join(data_dir, "data/{source}/tfidf-lemmatized-clusters.feather"),
    output:
        os.path.join(data_dir, "fig/{source}/article-tfidf-lemmatized-tsne.png"),
    params:
        title="Lemmatized TF-IDF"
    script:
        "scripts/plot_article_tsne.py"

rule plot_article_tfidf_tsne:
    input:
        os.path.join(data_dir, "data/{source}/tfidf.feather"),
        os.path.join(data_dir, "data/{source}/tfidf-clusters.feather"),
    output:
        os.path.join(data_dir, "fig/{source}/article-tfidf-tsne.png"),
    params:
        title="TF-IDF"
    script:
        "scripts/plot_article_tsne.py"

rule plot_article_biobert_tsne:
    input:
        os.path.join(data_dir, "data/{source}/biobert-mean.feather"),
        os.path.join(data_dir, "data/{source}/tfidf-clusters.feather"),
    output:
        os.path.join(data_dir, "fig/{source}/article-biobert-tsne.png"),
    params:
        title="BioBERT Mean Embedding"
    script:
        "scripts/plot_article_tsne.py"

rule compute_tfidf_lemmatized_clusters:
    input:
        os.path.join(data_dir, "data/{source}/tfidf-lemmatized.feather"),
    output:
        os.path.join(data_dir, "data/{source}/tfidf-lemmatized-clusters.feather"),
    script: "scripts/compute_tfidf_clusters.py"

rule compute_tfidf_clusters:
    input:
        os.path.join(data_dir, "data/{source}/tfidf.feather"),
    output:
        os.path.join(data_dir, "data/{source}/tfidf-clusters.feather"),
    script: "scripts/compute_tfidf_clusters.py"

rule compute_tfidf_lemmatized_matrix:
    input:
       os.path.join(data_dir, "data/{source}/articles-lemmatized.feather") 
    output:
        os.path.join(data_dir, "data/{source}/tfidf-lemmatized.feather"),
    script: "scripts/compute_tfidf_matrix.py"

rule compute_tfidf_matrix:
    input:
       os.path.join(data_dir, "data/{source}/articles.feather") 
    output:
        os.path.join(data_dir, "data/{source}/tfidf.feather"),
    script: "scripts/compute_tfidf_matrix.py"

rule compute_word_counts:
    input:
        os.path.join(data_dir, "data/{source}/word-freq.feather"),
    output:
        os.path.join(data_dir, "data/{source}/word-counts.feather")
    script:
        "scripts/compute_word_counts.py"

rule compute_word_freq_matrix:
    input:
       os.path.join(data_dir, "data/{source}/articles.feather") 
    output:
        os.path.join(data_dir, "data/{source}/word-freq.feather"),
    script: "scripts/compute_word_freq_matrix.py"

rule parse_arxiv_json:
    input:
        os.path.join(input_dir, "arxiv/arxiv-metadata-oai-snapshot.json")
    output:
        os.path.join(data_dir, "data/arxiv/articles.feather")
    script:
        "scripts/parse_arxiv_json.py"
    
rule download_arxiv_data: 
    output:
        os.path.join(input_dir, "arxiv/arxiv-metadata-oai-snapshot.json")
    shell:
        """
        cd `dirname {output}`
        kaggle datasets download -d "Cornell-University/arxiv"
        unzip arxiv.zip
        rm arxiv.zip
        """

# rule download_pubmed_daily_data:
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
        expand(os.path.join(input_dir, "pubmed/baseline/feather/pubmed22n{pubmed_baseline_num}.feather"), pubmed_baseline_num=pubmed_annual)
    output:
        os.path.join(data_dir, "data/pubmed/articles.feather")
    script:
        "scripts/combine_pubmed_articles.py"

rule convert_pubmed_baseline_data:
    input: 
        os.path.join(input_dir, "pubmed/baseline/xml/pubmed22n{pubmed_baseline_num}.xml.gz")
    output:
        os.path.join(input_dir, "pubmed/baseline/feather/pubmed22n{pubmed_baseline_num}.feather")
    script:
        "scripts/parse_pubmed_xml.py"

rule download_pubmed_baseline_data:
    output:
        os.path.join(input_dir, "pubmed/baseline/xml/pubmed22n{pubmed_baseline_num}.xml.gz")
    shell:
        """
        cd `dirname {output}`
        wget "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed22n${i}.xml.gz"
        """

rule download_biobert:
    output:
        os.path.join(model_dir, "biobert-v1.1/pytorch_model.bin")
    shell:
        """
        cd `dirname {output}`
        cd ..
        git clone https://huggingface.co/dmis-lab/biobert-v1.1
        """

