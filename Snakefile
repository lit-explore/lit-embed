"""
Science literature embedding pipeline
"""
import os

configfile: "config/config.yml"

# arxiv article subset ids
arxiv_num = [f"{n:04}" for n in range(1, config['arxiv']['num_chunks'] + 1)]

# pubmed annual file numbers (2022)
# pubmed_annual = [f"{n:04}" for n in range(1, 1115)]
# TESTING
pubmed_annual = [f"{n:04}" for n in range(1, 5)]

# pubmed daily update file numbers (not yet incorporated..)
# daily_start = config['pubmed']['updates_start']
# daily_end = config['pubmed']['updates_end']
#
# pubmed_daily = [f"{n:04}" for n in range(daily_start, daily_end + 1)]

model_dir = os.path.join(config['out_dir'], 'models')
input_dir = os.path.join(config['out_dir'], 'input')

# if "dev_mode" is enabled, operate on a subset of articles
if config['dev_mode']['enabled']:
    output_dir = os.path.join(config['out_dir'], 'output', 'subsampled', str(config['dev_mode']['num_articles']))
else:
    output_dir = os.path.join(config['out_dir'], 'output', 'full')

# wildcard values
data_sources = ['pubmed', 'arxiv']
processing_versions = ['baseline', 'lemmatized']
agg_funcs = ['mean', 'median']

rule all:
    input:
        expand(os.path.join(output_dir, "fig/{source}/article-tfidf-{processing}-tsne.png"), source=data_sources, processing=processing_versions),
        expand(os.path.join(output_dir, "data/{source}/tfidf-{processing}-umap.feather"), source=data_sources, processing=processing_versions),
        expand(os.path.join(output_dir, "fig/pubmed/article-biobert-{agg_func}-tsne.png"), agg_func=agg_funcs)

rule project_article_tfdf_umap:
    input:
        os.path.join(output_dir, "data/{source}/tfidf-{processing}.feather"),
        os.path.join(output_dir, "data/{source}/tfidf-{processing}-clusters.feather"),
    output:
        os.path.join(output_dir, "data/{source}/tfidf-{processing}-umap.feather"),
    script:
        "scripts/transform_umap.py"

rule plot_biobert_tsne:
    input:
        os.path.join(output_dir, "data/pubmed/biobert-{agg_func}.feather"),
        os.path.join(output_dir, "data/pubmed/biobert-{agg_func}-clusters.feather")
    output:
        os.path.join(output_dir, "fig/pubmed/article-biobert-{agg_func}-tsne.png"),
    params:
        title="BioBERT"
    script:
        "scripts/plot_article_tsne.py"

rule plot_article_tfidf_tsne:
    input:
        os.path.join(output_dir, "data/{source}/tfidf-{processing}.feather"),
        os.path.join(output_dir, "data/{source}/tfidf-{processing}-clusters.feather"),
    output:
        os.path.join(output_dir, "fig/{source}/article-tfidf-{processing}-tsne.png"),
    params:
        title="TF-IDF"
    script:
        "scripts/plot_article_tsne.py"

rule compute_tfidf_clusters:
    input:
        os.path.join(output_dir, "data/{source}/tfidf-{processing}.feather"),
    output:
        os.path.join(output_dir, "data/{source}/tfidf-{processing}-clusters.feather"),
    script: "scripts/compute_article_clusters.py"

rule compute_biobert_clusters:
    input:
        os.path.join(output_dir, "data/pubmed/biobert-{agg_func}.feather")
    output:
        os.path.join(output_dir, "data/pubmed/biobert-{agg_func}-clusters.feather"),
    script: "scripts/compute_article_clusters.py"

rule compute_tfidf_matrix:
    input:
       os.path.join(output_dir, "data/{source}/articles-{processing}.csv") 
    output:
        os.path.join(output_dir, "data/{source}/tfidf-{processing}.feather"),
    script: "scripts/compute_tfidf_matrix.py"

rule compute_word_counts:
    input:
        os.path.join(output_dir, "data/{source}/word-freq-{processing}.feather"),
    output:
        os.path.join(output_dir, "data/{source}/word-counts-{processing}.feather")
    script:
        "scripts/compute_word_counts.py"

rule compute_word_freq_matrix:
    input:
       os.path.join(output_dir, "data/{source}/articles-{processing}.csv") 
    output:
        os.path.join(output_dir, "data/{source}/word-freq-{processing}.feather"),
    script: "scripts/compute_word_freq_matrix.py"

rule combine_arxiv_lemmatized_articles:
    input:
        expand(os.path.join(input_dir, "arxiv/lemmatized/{arxiv_num}.feather"), arxiv_num=arxiv_num)
    output:
        os.path.join(output_dir, "data/arxiv/articles-lemmatized.csv")
    script:
        "scripts/combine_articles.py"

# combine articles and sub-sample, if enabled
rule combine_arxiv_articles:
    input:
        expand(os.path.join(input_dir, "arxiv/orig/{arxiv_num}.feather"), arxiv_num=arxiv_num)
    output:
        os.path.join(output_dir, "data/arxiv/articles-baseline.csv")
    script:
        "scripts/combine_articles.py"

rule combine_pubmed_lemmatized_articles:
    input:
        expand(os.path.join(input_dir, "pubmed/lemmatized/{pubmed_num}.feather"), pubmed_num=pubmed_annual)
    output:
        os.path.join(output_dir, "data/pubmed/articles-lemmatized.csv")
    script:
        "scripts/combine_articles.py"

# combine articles and sub-sample, if enabled
rule combine_pubmed_articles:
    input:
        expand(os.path.join(input_dir, "pubmed/orig/{pubmed_num}.feather"), pubmed_num=pubmed_annual)
    output:
        os.path.join(output_dir, "data/pubmed/articles-baseline.csv")
    script:
        "scripts/combine_articles.py"

rule combine_embeddings:
    input:
        expand(os.path.join(output_dir, "data/pubmed/biobert/{{agg_func}}/{pubmed_num}.feather"), pubmed_num=pubmed_annual),
    output:
        os.path.join(output_dir, "data/pubmed/biobert-{agg_func}.feather")
    script:
        "scripts/combine_embeddings.py"

rule create_embeddings:
    input:
        os.path.join(input_dir, "pubmed/orig/{pubmed_num}.feather"),
        os.path.join(model_dir, "biobert-v1.1/pytorch_model.bin")
    output:
        os.path.join(output_dir, "data/pubmed/biobert/mean/{pubmed_num}.feather"),
        os.path.join(output_dir, "data/pubmed/biobert/median/{pubmed_num}.feather"),
    script:
        "scripts/create_biobert_embeddings.py"

rule create_lemmatized_arxiv_corpus:
    input:
        os.path.join(input_dir, "arxiv/orig/{arxiv_num}.feather")
    output:
        os.path.join(input_dir, "arxiv/lemmatized/{arxiv_num}.feather")
    script:
        "scripts/lemmatize_text.py"

rule create_lemmatized_pubmed_corpus:
    input:
        os.path.join(input_dir, "pubmed/orig/{pubmed_num}.feather")
    output:
        os.path.join(input_dir, "pubmed/lemmatized/{pubmed_num}.feather")
    script:
        "scripts/lemmatize_text.py"

rule parse_pubmed_xml:
    input: 
        os.path.join(input_dir, "pubmed/raw/pubmed22n{pubmed_num}.xml.gz")
    output:
        os.path.join(input_dir, "pubmed/orig/{pubmed_num}.feather")
    script:
        "scripts/parse_pubmed_xml.py"

rule parse_arxiv_json:
    input:
        os.path.join(input_dir, "arxiv/raw/arxiv-metadata-oai-snapshot.json")
    output:
        os.path.join(input_dir, "arxiv/orig/{arxiv_num}.feather")
    script:
        "scripts/parse_arxiv_json.py"

rule download_biobert:
    output:
        os.path.join(model_dir, "biobert-v1.1/pytorch_model.bin")
    shell:
        """
        cd `dirname {output}`
        cd ..
        git clone https://huggingface.co/dmis-lab/biobert-v1.1
        """

rule download_pubmed_baseline_data:
    output:
        os.path.join(input_dir, "pubmed/raw/pubmed22n{pubmed_num}.xml.gz")
    shell:
        """
        cd `dirname {output}`
        wget "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed22n${pubmed_num}.xml.gz"
        """

rule download_arxiv_data: 
    output:
        os.path.join(input_dir, "arxiv/raw/arxiv-metadata-oai-snapshot.json")
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

