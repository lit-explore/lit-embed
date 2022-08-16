"""
lit-embed: Scientific literature embedding pipeline
"""
import os
import pandas as pd
from os.path import join

configfile: "config/config.yml"

# arxiv article subset ids
arxiv_num = [f"{n:04}" for n in range(1, config['arxiv']['num_chunks'] + 1)]

# pubmed annual file numbers
pubmed_annual = [f"{n:04}" for n in range(config['pubmed']['annual_start'], 
                                          config['pubmed']['annual_end'] + 1)]

# pubmed update file numbers
pubmed_updates = [f"{n:04}" for n in range(config['pubmed']['updates_start'], 
                                          config['pubmed']['updates_end'] + 1)]

# exclude pubmed22n0654.xml.gz (missing abstracts for all entries)
if config['exclude_articles']['missing_abstract']:
    pubmed_annual = [x for x in pubmed_annual if x != "0654"]

pubmed_all = pubmed_annual + pubmed_updates

model_dir = join(config['out_dir'], 'models')
input_dir = join(config['out_dir'], 'input')

# if "dev_mode" is enabled, operate on a subset of articles
if config['dev_mode']['enabled']:
    output_dir = join(config['out_dir'], 'output', 'subsampled', str(config['dev_mode']['num_articles']))
else:
    output_dir = join(config['out_dir'], 'output', 'full')

# wildcard values
idf_types = ['tfidf', 'mridf']
data_sources = ['pubmed', 'arxiv']
agg_funcs = ['mean', 'median', 'max']
proc_levels = ['baseline', 'lemmatized']
projection_types = ['tsne', 'umap']
targets = ['articles', 'embedding_columns']


wildcard_constraints:
    idf_type="|".join(idf_types)

rule all:
    input:
        # embeddings
        expand(join(output_dir, "data/{source}/embeddings/{idf_type}-{processing}.feather"), source=data_sources, idf_type=idf_types, processing=proc_levels),
        expand(join(output_dir, "data/{source}/embeddings/bert-{agg_func}.feather"), source=data_sources, agg_func=agg_funcs),

        # embedding performance
        expand(join(output_dir, "data/pubmed/performance/{idf_type}-{processing}.feather"), idf_type=idf_types, processing=proc_levels),
        expand(join(output_dir, "data/pubmed/performance/bert-{agg_func}.feather"), agg_func=agg_funcs),

        # word stats
        expand(join(output_dir, "data/{source}/word-stats/{processing}.feather"), source=data_sources, processing=proc_levels),

        # cluster/word associations
        expand(join(output_dir, "data/{source}/clusters/articles/associations/{idf_type}-{processing}-cluster-mridf-word-associations.feather"), source=data_sources, idf_type=idf_types, processing=proc_levels),
        expand(join(output_dir, "data/{source}/clusters/articles/associations/bert-{agg_func}-cluster-mridf-word-associations.feather"), source=data_sources, agg_func=agg_funcs),

        # plots
        expand(join(output_dir, "fig/{source}/{projection}/{target}/{idf_type}-{processing}-scatterplot.png"), source=data_sources, idf_type=idf_types, processing=proc_levels, target=targets, projection=projection_types),
        expand(join(output_dir, "fig/{source}/{projection}/{target}/bert-{agg_func}-scatterplot.png"), source=data_sources, projection=projection_types, target=targets, agg_func=agg_funcs),

        # co-citation
        join(output_dir, "data/pubmed/citations/citations.feather"),
        join(output_dir, "data/pubmed/citations/citations-stats.feather"),
        join(output_dir, "data/pubmed/citations/test/co-citation1.feather"),
        join(output_dir, "data/pubmed/citations/test/co-citation2.feather"),
        join(output_dir, "data/pubmed/citations/test/co-citation3.feather")

rule datashader:
    input:
        expand(join(output_dir, "fig/{source}/umap/articles/{idf_type}-{processing}-datashader.png"), source=data_sources, idf_type=idf_types, processing=proc_levels),
        expand(join(output_dir, "fig/{source}/umap/articles/bert-{agg_func}-datashader.png"), source=data_sources, agg_func=agg_funcs)

rule compute_tfidf_embedding_cluster_mridf_word_associations:
    input:
       join(output_dir, "data/{source}/embeddings/mridf-lemmatized.feather"),
       join(output_dir, "data/{source}/clusters/articles/tfidf-{processing}-clusters.feather"),
    output:
       join(output_dir, "data/{source}/clusters/articles/associations/tfidf-{processing}-cluster-mridf-word-associations.feather"),
    script: "scripts/compute_cluster_embedding_associations.py"

rule compute_bert_embedding_cluster_mridf_word_associations:
    input:
       join(output_dir, "data/{source}/embeddings/mridf-lemmatized.feather"),
       join(output_dir, "data/{source}/clusters/articles/bert-{agg_func}-clusters.feather"),
    output:
       join(output_dir, "data/{source}/clusters/articles/associations/bert-{agg_func}-cluster-mridf-word-associations.feather")
    script: "scripts/compute_cluster_embedding_associations.py"

rule compute_idf_cluster_average_embeddings:
    input:
        join(output_dir, "data/{source}/embeddings/{idf_type}-{processing}.feather"),
        join(output_dir, "data/{source}/clusters/articles/{idf_type}-{processing}-clusters.feather"),
    output:
        join(output_dir, "data/{source}/embeddings/cluster-mean-embeddings/{idf_type}-{processing}-mean-embedding.feather"),
    script:
        "scripts/compute_cluster_average_embeddings.py"

# rule compute_idf_bert_embedding_feature_correlations:
#     input:
#         join(output_dir, "data/{source}/embeddings/{idf_type}-{processing}.feather"),
#         join(output_dir, "data/{source}/embeddings/bert-{agg_func}.feather"),
#     output:
#         join(output_dir, "data/{source}/correlation/embedding_columns/{idf_type}-{processing}-bert-{agg_func}.feather")
#     script:
#         "scripts/compute_cross_embedding_feature_correaltions.py"

rule evaluate_bert_embedding_performance:
    input:
        join(output_dir, "data/pubmed/embeddings/bert-{agg_func}.feather"),
        join(output_dir, "data/pubmed/citations/test/co-citation1.feather"),
        join(output_dir, "data/pubmed/citations/test/co-citation2.feather"),
        join(output_dir, "data/pubmed/citations/test/co-citation3.feather")
    output:
        join(output_dir, "data/pubmed/performance/bert-{agg_func}.feather"),
    script:
        "scripts/compute_article_embedding_correlations.py"

rule evaluate_idf_embedding_performance:
    input:
        join(output_dir, "data/pubmed/embeddings/{idf_type}-{processing}.feather"),
        join(output_dir, "data/pubmed/citations/test/co-citation1.feather"),
        join(output_dir, "data/pubmed/citations/test/co-citation2.feather"),
        join(output_dir, "data/pubmed/citations/test/co-citation3.feather")
    output:
        join(output_dir, "data/pubmed/performance/{idf_type}-{processing}.feather"),
    script:
        "scripts/compute_article_embedding_correlations.py"

rule create_pubmed_test_sets:
    input:
        join(output_dir, "data/pubmed/citations/citations.feather"),
        join(output_dir, "data/pubmed/citations/citations-stats.feather"),
        join(output_dir, "data/pubmed/corpus/articles-baseline.csv")
    output:
        join(output_dir, "data/pubmed/citations/test/co-citation1.feather"),
        join(output_dir, "data/pubmed/citations/test/co-citation2.feather"),
        join(output_dir, "data/pubmed/citations/test/co-citation3.feather")
    script:
        "scripts/create_pubmed_test_sets.py"

rule plot_idf_datashader:
    input:
        join(output_dir, "data/{source}/projections/umap/articles/{idf_type}-{processing}.feather"),
        join(output_dir, "data/{source}/clusters/articles/{idf_type}-{processing}-clusters.feather"),
    output:
        join(output_dir, "fig/{source}/umap/articles/{idf_type}-{processing}-datashader.png"),
    conda:
        "envs/datashader.yml"
    script:
        "scripts/plot_datashader.py"

rule plot_bert_datashader:
    input:
        join(output_dir, "data/{source}/projections/umap/articles/bert-{agg_func}.feather"),
        join(output_dir, "data/{source}/clusters/articles/bert-{agg_func}-clusters.feather"),
    output:
        join(output_dir, "fig/{source}/umap/articles/bert-{agg_func}-datashader.png"),
    conda:
        "envs/datashader.yml"
    script:
        "scripts/plot_datashader.py"

rule plot_idf_scatterplot:
    input:
        join(output_dir, "data/{source}/projections/{projection}/{target}/{idf_type}-{processing}.feather"),
        join(output_dir, "data/{source}/clusters/{target}/{idf_type}-{processing}-clusters.feather"),
    output:
        join(output_dir, "fig/{source}/{projection}/{target}/{idf_type}-{processing}-scatterplot.png"),
    script:
        "scripts/plot_scatter.py"

rule plot_bert_scatterplot:
    input:
        join(output_dir, "data/{source}/projections/{projection}/{target}/bert-{agg_func}.feather"),
        join(output_dir, "data/{source}/clusters/{target}/bert-{agg_func}-clusters.feather"),
    output:
        join(output_dir, "fig/{source}/{projection}/{target}/bert-{agg_func}-scatterplot.png"),
    script:
        "scripts/plot_scatter.py"

rule idf_dimension_reduction:
    input:
        join(output_dir, "data/{source}/embeddings/{idf_type}-{processing}.feather"),
    output:
        join(output_dir, "data/{source}/projections/{projection}/{target}/{idf_type}-{processing}.feather"),
    script:
        "scripts/reduce_dimension.py"

rule bert_dimension_reduction:
    input:
        join(output_dir, "data/{source}/embeddings/bert-{agg_func}.feather")
    output:
        join(output_dir, "data/{source}/projections/{projection}/{target}/bert-{agg_func}.feather"),
    script:
        "scripts/reduce_dimension.py"

rule compute_idf_article_clusters:
    input:
        join(output_dir, "data/{source}/embeddings/{idf_type}-{processing}.feather"),
    output:
        join(output_dir, "data/{source}/clusters/articles/{idf_type}-{processing}-clusters.feather"),
    script: "scripts/cluster_articles.py"

rule compute_idf_embedding_column_clusters:
    input:
        join(output_dir, "data/{source}/embeddings/{idf_type}-{processing}.feather"),
    output:
        join(output_dir, "data/{source}/clusters/embedding_columns/{idf_type}-{processing}-clusters.feather"),
    script: "scripts/cluster_embedding_columns.py"

rule compute_bert_embedding_article_clusters:
    input:
        join(output_dir, "data/{source}/embeddings/bert-{agg_func}.feather")
    output:
        join(output_dir, "data/{source}/clusters/articles/bert-{agg_func}-clusters.feather"),
    script: "scripts/cluster_articles.py"

rule compute_bert_embedding_column_clusters:
    input:
        join(output_dir, "data/{source}/embeddings/bert-{agg_func}.feather")
    output:
        join(output_dir, "data/{source}/clusters/embedding_columns/bert-{agg_func}-clusters.feather"),
    script: "scripts/cluster_embedding_columns.py"

rule compute_tfidf_matrix:
    input:
       join(output_dir, "data/{source}/corpus/articles-{processing}.csv") 
    output:
        join(output_dir, "data/{source}/embeddings/tfidf-{processing}.feather"),
        join(output_dir, "data/{source}/embeddings/tfidf-{processing}-sparse-mat.npz"),
        join(output_dir, "data/{source}/embeddings/tfidf-{processing}-stats.feather"),
    script: "scripts/compute_tfidf_matrix.py"

rule combine_arxiv_lemmatized_articles:
    input:
        expand(join(input_dir, "arxiv/lemmatized/{arxiv_num}.feather"), arxiv_num=arxiv_num)
    output:
        join(output_dir, "data/arxiv/corpus/articles-lemmatized.csv")
    script:
        "scripts/combine_articles.py"

# combine articles and sub-sample, if enabled
rule combine_arxiv_articles:
    input:
        expand(join(input_dir, "arxiv/baseline/{arxiv_num}.feather"), arxiv_num=arxiv_num)
    output:
        join(output_dir, "data/arxiv/corpus/articles-baseline.csv")
    script:
        "scripts/combine_articles.py"

rule combine_pubmed_lemmatized_articles:
    input:
        expand(join(input_dir, "pubmed/lemmatized/{pubmed_num}.feather"), pubmed_num=pubmed_all)
    output:
        join(output_dir, "data/pubmed/corpus/articles-lemmatized.csv")
    script:
        "scripts/combine_articles.py"

# compute modified RIDF matrices
rule compute_mridf_mat:
    input:
       join(output_dir, "data/{source}/corpus/articles-{processing}.csv"),
       join(output_dir, "data/{source}/word-stats/{processing}.feather")
    output:
       join(output_dir, "data/{source}/embeddings/mridf-{processing}.feather")
    script:
        "scripts/compute_mridf_matrix.py"

rule compute_word_stats:
    input:
        join(output_dir, "data/{source}/word-counts/{processing}.feather"),
    output: 
        join(output_dir, "data/{source}/word-stats/{processing}.feather"),
    script:
        "scripts/compute_word_stats.py"

rule combine_baseline_arxiv_word_counts:
    input:
        expand(join(output_dir, "data/arxiv/word-counts/batches/baseline/{arxiv_num}.feather"), arxiv_num=arxiv_num)
    output:
        join(output_dir, "data/arxiv/word-counts/baseline.feather")
    script:
        "scripts/combine_word_counts.py"

rule combine_lemmatized_arxiv_word_counts:
    input:
        expand(join(output_dir, "data/arxiv/word-counts/batches/lemmatized/{arxiv_num}.feather"), arxiv_num=arxiv_num)
    output:
        join(output_dir, "data/arxiv/word-counts/lemmatized.feather")
    script:
        "scripts/combine_word_counts.py"

rule compute_baseline_arxiv_word_counts:
    input:
        join(input_dir, "arxiv/baseline/{arxiv_num}.feather")
    output:
        join(output_dir, "data/arxiv/word-counts/batches/baseline/{arxiv_num}.feather")
    script:
        "scripts/create_word_count_matrix.py"

rule compute_lemmatized_arxiv_word_counts:
    input:
        join(input_dir, "arxiv/lemmatized/{arxiv_num}.feather")
    output:
        join(output_dir, "data/arxiv/word-counts/batches/lemmatized/{arxiv_num}.feather")
    script:
        "scripts/create_word_count_matrix.py"

rule combine_baseline_pubmed_word_counts:
    input:
        expand(join(output_dir, "data/pubmed/word-counts/batches/baseline/{pubmed_num}.feather"), pubmed_num=pubmed_all)
    output:
        join(output_dir, "data/pubmed/word-counts/baseline.feather")
    script:
        "scripts/combine_word_counts.py"

rule combine_lemmatized_pubmed_word_counts:
    input:
        expand(join(output_dir, "data/pubmed/word-counts/batches/lemmatized/{pubmed_num}.feather"), pubmed_num=pubmed_all)
    output:
        join(output_dir, "data/pubmed/word-counts/lemmatized.feather")
    script:
        "scripts/combine_word_counts.py"

rule compute_baseline_pubmed_word_counts:
    input:
        join(input_dir, "pubmed/baseline/{pubmed_num}.feather")
    output:
        join(output_dir, "data/pubmed/word-counts/batches/baseline/{pubmed_num}.feather")
    script:
        "scripts/create_word_count_matrix.py"

rule compute_lemmatized_pubmed_word_counts:
    input:
        join(input_dir, "pubmed/lemmatized/{pubmed_num}.feather")
    output:
        join(output_dir, "data/pubmed/word-counts/batches/lemmatized/{pubmed_num}.feather")
    script:
        "scripts/create_word_count_matrix.py"

rule combine_arxiv_bert_embeddings:
    input:
        expand(join(output_dir, "data/arxiv/bert/{{agg_func}}/{arxiv_num}.feather"), arxiv_num=arxiv_num),
    output:
        join(output_dir, "data/arxiv/embeddings/bert-{agg_func}.feather")
    script:
        "scripts/combine_embeddings.py"

rule combine_pubmed_bert_embeddings:
    input:
        expand(join(output_dir, "data/pubmed/embeddings/bert/{{agg_func}}/{pubmed_num}.feather"), pubmed_num=pubmed_all),
    output:
        join(output_dir, "data/pubmed/embeddings/bert-{agg_func}.feather")
    script:
        "scripts/combine_embeddings.py"

rule create_arxiv_scibert_embeddings:
    input:
        join(input_dir, "arxiv/baseline/{arxiv_num}.feather"),
        join(model_dir, "scibert_scivocab_uncased/pytorch_model.bin")
    output:
        join(output_dir, "data/arxiv/bert/mean/{arxiv_num}.feather"),
        join(output_dir, "data/arxiv/bert/median/{arxiv_num}.feather"),
        join(output_dir, "data/arxiv/bert/max/{arxiv_num}.feather")
    script:
        "scripts/create_bert_embeddings.py"

rule create_pubmed_biobert_embeddings:
    input:
        join(input_dir, "pubmed/baseline/{pubmed_num}.feather"),
        join(model_dir, "biobert-v1.1/pytorch_model.bin")
    output:
        join(output_dir, "data/pubmed/embeddings/bert/mean/{pubmed_num}.feather"),
        join(output_dir, "data/pubmed/embeddings/bert/median/{pubmed_num}.feather"),
        join(output_dir, "data/pubmed/embeddings/bert/max/{pubmed_num}.feather")
    script:
        "scripts/create_bert_embeddings.py"

rule combine_pubmed_articles:
    input:
        expand(join(input_dir, "pubmed/baseline/{pubmed_num}.feather"), pubmed_num=pubmed_all)
    output:
        join(output_dir, "data/pubmed/corpus/articles-baseline.csv")
    script:
        "scripts/combine_articles.py"

rule create_lemmatized_arxiv_corpus:
    input:
        join(input_dir, "arxiv/baseline/{arxiv_num}.feather")
    output:
        join(input_dir, "arxiv/lemmatized/{arxiv_num}.feather")
    script:
        "scripts/lemmatize_text.py"

rule create_lemmatized_pubmed_corpus:
    input:
        join(input_dir, "pubmed/baseline/{pubmed_num}.feather")
    output:
        join(input_dir, "pubmed/lemmatized/{pubmed_num}.feather")
    script:
        "scripts/lemmatize_text.py"

rule parse_pubmed_xml:
    input: 
        join(input_dir, "pubmed/raw/pubmed22n{pubmed_num}.xml.gz")
    output:
        join(input_dir, "pubmed/baseline/{pubmed_num}.feather")
    script:
        "scripts/parse_pubmed_xml.py"

rule parse_arxiv_json:
    input:
        join(input_dir, "arxiv/raw/arxiv-metadata-oai-snapshot.json")
    output:
        join(input_dir, "arxiv/baseline/{arxiv_num}.feather")
    script:
        "scripts/parse_arxiv_json.py"

rule download_scibert:
    output:
        join(model_dir, "scibert_scivocab_uncased/pytorch_model.bin")
    shell:
        """
        cd `dirname {output}`
        cd ..
        git lfs install
        git clone https://huggingface.co/allenai/scibert_scivocab_uncased 
        """

rule download_biobert:
    output:
        join(model_dir, "biobert-v1.1/pytorch_model.bin")
    shell:
        """
        cd `dirname {output}`
        cd ..
        git lfs install
        git clone https://huggingface.co/dmis-lab/biobert-v1.1
        """

rule combine_pubmed_citations:
    input:
        citations=expand(join(input_dir, "pubmed/citations/{pubmed_num}.feather"), pubmed_num=pubmed_all),
        stats=expand(join(input_dir, "pubmed/citations/{pubmed_num}_stats.feather"), pubmed_num=pubmed_all)
    output:
        join(output_dir, "data/pubmed/citations/citations.feather"),
        join(output_dir, "data/pubmed/citations/citations-stats.feather"),
    script:
        "scripts/combine_pubmed_citations.py"

rule extract_pubmed_citations:
    input:
        join(input_dir, "pubmed/raw/pubmed22n{pubmed_num}.xml.gz")
    output:
        join(input_dir, "pubmed/citations/{pubmed_num}.feather"),
        join(input_dir, "pubmed/citations/{pubmed_num}_stats.feather")
    script:
        "scripts/extract_pubmed_citations.py"

rule download_pubmed_data:
    output:
        join(input_dir, "pubmed/raw/pubmed22n{pubmed_num}.xml.gz")
    shell:
        """
        cd `dirname {output}`

        if [ "{wildcards.pubmed_num}" -gt "{config[pubmed][annual_end]}" ]; then
            echo "daily!";
            ftpdir="updatefiles"
        else
            echo "annual!"
            ftpdir="baseline"
        fi

        echo "https://ftp.ncbi.nlm.nih.gov/pubmed/${{ftpdir}}/pubmed22n{wildcards.pubmed_num}.xml.gz"
        wget "https://ftp.ncbi.nlm.nih.gov/pubmed/${{ftpdir}}/pubmed22n{wildcards.pubmed_num}.xml.gz"
        """

rule download_arxiv_data: 
    output:
        join(input_dir, "arxiv/raw/arxiv-metadata-oai-snapshot.json")
    shell:
        """
        cd `dirname {output}`
        kaggle datasets download -d "Cornell-University/arxiv"
        unzip arxiv.zip
        rm arxiv.zip
        """
