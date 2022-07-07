"""
lit-embed: Scientific literature embedding pipeline
"""
import os
import pandas as pd
from scipy.stats import pearsonr

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
projection_types = ['tsne', 'umap']
targets = ['articles', 'embedding_columns']

common_embeddings = ['tfidf-baseline', 'tfidf-lemmatized', 'mridf-baseline', 'mridf-lemmatized']
pubmed_embeddings = common_embeddings + ['biobert-mean', 'biobert-median']
arxiv_embeddings = common_embeddings + ['scibert-mean', 'scibert-median']

#
# DROP BERT AGG FUNCS? (Simplify..)
#
#

rule all:
    input:
        expand(os.path.join(output_dir, "fig/{source}/{projection}/{target}/tfidf-{processing}-scatterplot.png"), source=data_sources, processing=processing_versions, target=targets, projection=projection_types),
        expand(os.path.join(output_dir, "fig/pubmed/{projection}/{target}/biobert-{agg_func}-scatterplot.png"), projection=projection_types, target=targets, agg_func=agg_funcs),
        expand(os.path.join(output_dir, "fig/arxiv/{projection}/{target}/scibert-{agg_func}-scatterplot.png"), projection=projection_types, target=targets, agg_func=agg_funcs),
        expand(os.path.join(output_dir, "data/{source}/embeddings/cluster-mean-embeddings/tfidf-{processing}-mean-embedding.feather"), source=data_sources, processing=processing_versions),
        expand(os.path.join(output_dir, "data/pubmed/comparison/biobert-{agg_func}-tfidf-{processing}.feather"), agg_func=agg_funcs, processing=processing_versions),
        expand(os.path.join(output_dir, "data/pubmed/word-stats/{processing}.feather"), processing=processing_versions),
        expand(os.path.join(output_dir, "data/pubmed/embeddings/mridf-{processing}.feather"), processing=processing_versions),
        expand(os.path.join(output_dir, "data/pubmed/clusters/articles/associations/{embedding}-cluster-mridf-word-associations.feather"), embedding=pubmed_embeddings),
        expand(os.path.join(output_dir, "data/arxiv/word-stats/{processing}.feather"), processing=processing_versions),
        expand(os.path.join(output_dir, "data/arxiv/embeddings/mridf-{processing}.feather"), processing=processing_versions),
        expand(os.path.join(output_dir, "data/arxiv/clusters/articles/associations/{embedding}-cluster-mridf-word-associations.feather"), embedding=arxiv_embeddings),
        #expand(os.path.join(output_dir, "data/arxiv/clusters/articles/associations/{embedding}-cluster-associations.feather"), embedding=arxiv_embeddings)
        #expand(os.path.join(output_dir, "data/pubmed/clusters/articles/associations/biobert-{agg_func}-cluster-mridf-word-associations.feather"),

rule datashader:
    input:
        expand(os.path.join(output_dir, "fig/{source}/umap/articles/tfidf-{processing}-datashader.png"), source=data_sources, processing=processing_versions),
        expand(os.path.join(output_dir, "fig/pubmed/umap/articles/biobert-{agg_func}-datashader.png"), agg_func=agg_funcs),
        expand(os.path.join(output_dir, "fig/arxiv/umap/articles/scibert-{agg_func}-datashader.png"), agg_func=agg_funcs)

rule compute_pubmed_embedding_cluster_word_associations:
    input:
       os.path.join(output_dir, "data/pubmed/embeddings/mridf-lemmatized.feather"),
       os.path.join(output_dir, "data/pubmed/clusters/articles/{embedding}-clusters.feather"),
    output:
       os.path.join(output_dir, "data/pubmed/clusters/articles/associations/{embedding}-cluster-mridf-word-associations.feather")
    script: "scripts/compute_cluster_embedding_associations.py"

rule compute_arxiv_embedding_cluster_word_associations:
    input:
       os.path.join(output_dir, "data/arxiv/embeddings/mridf-lemmatized.feather"),
       os.path.join(output_dir, "data/arxiv/clusters/articles/{embedding}-clusters.feather"),
    output:
       os.path.join(output_dir, "data/arxiv/clusters/articles/associations/{embedding}-cluster-mridf-word-associations.feather")
    script: "scripts/compute_cluster_embedding_associations.py"

rule compute_tfidf_cluster_average_embeddings:
    input:
        os.path.join(output_dir, "data/{source}/embeddings/tfidf-{processing}.feather"),
        os.path.join(output_dir, "data/{source}/clusters/articles/tfidf-{processing}-clusters.feather"),
    output:
        os.path.join(output_dir, "data/{source}/embeddings/cluster-mean-embeddings/tfidf-{processing}-mean-embedding.feather"),
    script:
        "scripts/compute_cluster_average_embeddings.py"

rule compute_embedding_correlation:
    input:
        os.path.join(output_dir, "data/pubmed/biobert-{agg_func}.feather"),
        os.path.join(output_dir, "data/pubmed/embeddings/tfidf-{processing}.feather")
    output:
        os.path.join(output_dir, "data/pubmed/comparison/biobert-{agg_func}-tfidf-{processing}.feather")
    run:
        df1 = pd.read_feather(input[0]).set_index('article_id')
        df2 = pd.read_feather(input[1]).set_index('article_id')

        shared_ids = list(set(df1.index).intersection(df2.index))

        df1 = df1.loc[shared_ids]
        df2 = df2.loc[shared_ids]
 
        # corrwith doesn't scale
        # df2.columns = df1.columns
        # cor_mat = df1.corrwith(df2, axis=1)

        # compare article correlations across embeddings
        cors = []

        for i in range(df1.shape[0]):
            cors.append(pearsonr(df1.iloc[i, :], df2.iloc[i, :])[0])

        res = pd.DataFrame({"article_id": df1.index, "pearson_cor": cors})

        res.to_feather(output[0])

rule plot_tfidf_datashader:
    input:
        os.path.join(output_dir, "data/{source}/projections/umap/articles/tfidf-{processing}.feather"),
        os.path.join(output_dir, "data/{source}/clusters/articles/tfidf-{processing}-clusters.feather"),
    output:
        os.path.join(output_dir, "fig/{source}/umap/articles/tfidf-{processing}-datashader.png"),
    params:
        name="TF-IDF"
    conda:
        "envs/datashader.yml"
    script:
        "scripts/plot_datashader.py"

rule plot_scibert_datashader:
    input:
        os.path.join(output_dir, "data/arxiv/projections/umap/articles/scibert-{agg_func}.feather"),
        os.path.join(output_dir, "data/arxiv/clusters/articles/scibert-{agg_func}-clusters.feather"),
    output:
        os.path.join(output_dir, "fig/arxiv/umap/articles/scibert-{agg_func}-datashader.png"),
    params:
        name="SciBERT"
    conda:
        "envs/datashader.yml"
    script:
        "scripts/plot_datashader.py"

rule plot_biobert_datashader:
    input:
        os.path.join(output_dir, "data/pubmed/projections/umap/articles/biobert-{agg_func}.feather"),
        os.path.join(output_dir, "data/pubmed/clusters/articles/biobert-{agg_func}-clusters.feather"),
    output:
        os.path.join(output_dir, "fig/pubmed/umap/articles/biobert-{agg_func}-datashader.png"),
    params:
        name="BioBERT"
    conda:
        "envs/datashader.yml"
    script:
        "scripts/plot_datashader.py"

rule plot_tfidf_scatterplot:
    input:
        os.path.join(output_dir, "data/{source}/projections/{projection}/{target}/tfidf-{processing}.feather"),
        os.path.join(output_dir, "data/{source}/clusters/{target}/tfidf-{processing}-clusters.feather"),
    output:
        os.path.join(output_dir, "fig/{source}/{projection}/{target}/tfidf-{processing}-scatterplot.png"),
    params:
        name="TF-IDF"
    script:
        "scripts/plot_scatter.py"

rule plot_biobert_scatterplot:
    input:
        os.path.join(output_dir, "data/pubmed/projections/{projection}/{target}/biobert-{agg_func}.feather"),
        os.path.join(output_dir, "data/pubmed/clusters/{target}/biobert-{agg_func}-clusters.feather"),
    output:
        os.path.join(output_dir, "fig/pubmed/{projection}/{target}/biobert-{agg_func}-scatterplot.png"),
    params:
        name="BioBERT"
    script:
        "scripts/plot_scatter.py"

rule plot_scibert_scatterplot:
    input:
        os.path.join(output_dir, "data/arxiv/projections/{projection}/{target}/scibert-{agg_func}.feather"),
        os.path.join(output_dir, "data/arxiv/clusters/{target}/scibert-{agg_func}-clusters.feather"),
    output:
        os.path.join(output_dir, "fig/arxiv/{projection}/{target}/scibert-{agg_func}-scatterplot.png"),
    params:
        name="SciBERT"
    script:
        "scripts/plot_scatter.py"

rule tfidf_dimension_reduction:
    input:
        os.path.join(output_dir, "data/{source}/tfidf-{processing}.feather"),
    output:
        os.path.join(output_dir, "data/{source}/projections/{projection}/{target}/tfidf-{processing}.feather"),
    script:
        "scripts/reduce_dimension.py"

rule biobert_dimension_reduction:
    input:
        os.path.join(output_dir, "data/pubmed/biobert-{agg_func}.feather")
    output:
        os.path.join(output_dir, "data/pubmed/projections/{projection}/{target}/biobert-{agg_func}.feather"),
    script:
        "scripts/reduce_dimension.py"

rule scibert_dimension_reduction:
    input:
        os.path.join(output_dir, "data/arxiv/scibert-{agg_func}.feather")
    output:
        os.path.join(output_dir, "data/arxiv/projections/{projection}/{target}/scibert-{agg_func}.feather"),
    script:
        "scripts/reduce_dimension.py"

rule compute_tfidf_article_clusters:
    input:
        os.path.join(output_dir, "data/{source}/embeddings/tfidf-{processing}.feather"),
    output:
        os.path.join(output_dir, "data/{source}/clusters/articles/tfidf-{processing}-clusters.feather"),
    script: "scripts/cluster_articles.py"

rule compute_tfidf_embedding_column_clusters:
    input:
        os.path.join(output_dir, "data/{source}/embeddings/tfidf-{processing}.feather"),
    output:
        os.path.join(output_dir, "data/{source}/clusters/embedding_columns/tfidf-{processing}-clusters.feather"),
    script: "scripts/cluster_embedding_columns.py"

rule compute_mridf_article_clusters:
    input:
        os.path.join(output_dir, "data/{source}/embeddings/mridf-{processing}.feather"),
    output:
        os.path.join(output_dir, "data/{source}/clusters/articles/mridf-{processing}-clusters.feather"),
    script: "scripts/cluster_articles.py"

rule compute_mridf_embedding_column_clusters:
    input:
        os.path.join(output_dir, "data/{source}/embeddings/mridf-{processing}.feather"),
    output:
        os.path.join(output_dir, "data/{source}/clusters/embedding_columns/mridf-{processing}-clusters.feather"),
    script: "scripts/cluster_embedding_columns.py"

rule compute_biobert_embedding_article_clusters:
    input:
        os.path.join(output_dir, "data/pubmed/biobert-{agg_func}.feather")
    output:
        os.path.join(output_dir, "data/pubmed/clusters/articles/biobert-{agg_func}-clusters.feather"),
    script: "scripts/cluster_articles.py"

rule compute_biobert_embedding_column_clusters:
    input:
        os.path.join(output_dir, "data/pubmed/biobert-{agg_func}.feather")
    output:
        os.path.join(output_dir, "data/pubmed/clusters/embedding_columns/biobert-{agg_func}-clusters.feather"),
    script: "scripts/cluster_embedding_columns.py"

rule compute_scibert_embedding_article_clusters:
    input:
        os.path.join(output_dir, "data/arxiv/scibert-{agg_func}.feather")
    output:
        os.path.join(output_dir, "data/arxiv/clusters/articles/scibert-{agg_func}-clusters.feather"),
    script: "scripts/cluster_articles.py"

rule compute_scibert_embedding_column_clusters:
    input:
        os.path.join(output_dir, "data/arxiv/scibert-{agg_func}.feather")
    output:
        os.path.join(output_dir, "data/arxiv/clusters/embedding_columns/scibert-{agg_func}-clusters.feather"),
    script: "scripts/cluster_embedding_columns.py"

rule compute_tfidf_matrix:
    input:
       os.path.join(output_dir, "data/{source}/corpus/articles-{processing}.csv") 
    output:
        os.path.join(output_dir, "data/{source}/embeddings/tfidf-{processing}.feather"),
        os.path.join(output_dir, "data/{source}/embeddings/tfidf-{processing}-sparse-mat.npz"),
        os.path.join(output_dir, "data/{source}/embeddings/tfidf-{processing}-stats.feather"),
    script: "scripts/compute_tfidf_matrix.py"

rule combine_arxiv_lemmatized_articles:
    input:
        expand(os.path.join(input_dir, "arxiv/lemmatized/{arxiv_num}.feather"), arxiv_num=arxiv_num)
    output:
        os.path.join(output_dir, "data/arxiv/corpus/articles-lemmatized.csv")
    script:
        "scripts/combine_articles.py"

# combine articles and sub-sample, if enabled
rule combine_arxiv_articles:
    input:
        expand(os.path.join(input_dir, "arxiv/baseline/{arxiv_num}.feather"), arxiv_num=arxiv_num)
    output:
        os.path.join(output_dir, "data/arxiv/corpus/articles-baseline.csv")
    script:
        "scripts/combine_articles.py"

rule combine_pubmed_lemmatized_articles:
    input:
        expand(os.path.join(input_dir, "pubmed/lemmatized/{pubmed_num}.feather"), pubmed_num=pubmed_all)
    output:
        os.path.join(output_dir, "data/pubmed/corpus/articles-lemmatized.csv")
    script:
        "scripts/combine_articles.py"

# compute modified RIDF matrices
rule compute_arxiv_ridf_mat:
    input:
       os.path.join(output_dir, "data/arxiv/corpus/articles-{processing}.csv"),
       os.path.join(output_dir, "data/arxiv/word-stats/{processing}.feather")
    output:
       os.path.join(output_dir, "data/arxiv/embeddings/mridf-{processing}.feather")
    script:
        "scripts/compute_mridf_matrix.py"

rule compute_pubmed_ridf_mat:
    input:
       os.path.join(output_dir, "data/pubmed/corpus/articles-{processing}.csv"),
       os.path.join(output_dir, "data/pubmed/word-stats/{processing}.feather")
    output:
       os.path.join(output_dir, "data/pubmed/embeddings/mridf-{processing}.feather")
    script:
        "scripts/compute_mridf_matrix.py"

# arxiv word stats
rule compute_arxiv_word_stats:
    input:
        os.path.join(output_dir, "data/arxiv/word-counts/{processing}.feather"),
    output: 
        os.path.join(output_dir, "data/arxiv/word-stats/{processing}.feather"),
    script:
        "scripts/compute_word_stats.py"

rule combine_baseline_arxiv_word_counts:
    input:
        expand(os.path.join(output_dir, "data/arxiv/word-counts/batches/baseline/{arxiv_num}.feather"), arxiv_num=arxiv_num)
    output:
        os.path.join(output_dir, "data/arxiv/word-counts/baseline.feather")
    script:
        "scripts/combine_word_counts.py"

rule combine_lemmatized_arxiv_word_counts:
    input:
        expand(os.path.join(output_dir, "data/arxiv/word-counts/batches/lemmatized/{arxiv_num}.feather"), arxiv_num=arxiv_num)
    output:
        os.path.join(output_dir, "data/arxiv/word-counts/lemmatized.feather")
    script:
        "scripts/combine_word_counts.py"

rule compute_baseline_arxiv_word_counts:
    input:
        os.path.join(input_dir, "arxiv/baseline/{arxiv_num}.feather")
    output:
        os.path.join(output_dir, "data/arxiv/word-counts/batches/baseline/{arxiv_num}.feather")
    script:
        "scripts/create_word_count_matrix.py"

rule compute_lemmatized_arxiv_word_counts:
    input:
        os.path.join(input_dir, "arxiv/lemmatized/{arxiv_num}.feather")
    output:
        os.path.join(output_dir, "data/arxiv/word-counts/batches/lemmatized/{arxiv_num}.feather")
    script:
        "scripts/create_word_count_matrix.py"

# pubmed word stats
rule compute_pubmed_word_stats:
    input:
        os.path.join(output_dir, "data/pubmed/word-counts/{processing}.feather"),
    output: 
        os.path.join(output_dir, "data/pubmed/word-stats/{processing}.feather"),
    script:
        "scripts/compute_word_stats.py"

rule combine_baseline_pubmed_word_counts:
    input:
        expand(os.path.join(output_dir, "data/pubmed/word-counts/batches/baseline/{pubmed_num}.feather"), pubmed_num=pubmed_all)
    output:
        os.path.join(output_dir, "data/pubmed/word-counts/baseline.feather")
    script:
        "scripts/combine_word_counts.py"

rule combine_lemmatized_pubmed_word_counts:
    input:
        expand(os.path.join(output_dir, "data/pubmed/word-counts/batches/lemmatized/{pubmed_num}.feather"), pubmed_num=pubmed_all)
    output:
        os.path.join(output_dir, "data/pubmed/word-counts/lemmatized.feather")
    script:
        "scripts/combine_word_counts.py"

rule compute_baseline_pubmed_word_counts:
    input:
        os.path.join(input_dir, "pubmed/baseline/{pubmed_num}.feather")
    output:
        os.path.join(output_dir, "data/pubmed/word-counts/batches/baseline/{pubmed_num}.feather")
    script:
        "scripts/create_word_count_matrix.py"

rule compute_lemmatized_pubmed_word_counts:
    input:
        os.path.join(input_dir, "pubmed/lemmatized/{pubmed_num}.feather")
    output:
        os.path.join(output_dir, "data/pubmed/word-counts/batches/lemmatized/{pubmed_num}.feather")
    script:
        "scripts/create_word_count_matrix.py"

rule combine_arxiv_scibert_embeddings:
    input:
        expand(os.path.join(output_dir,
            "data/arxiv/scibert/{{agg_func}}/{arxiv_num}.feather"), arxiv_num=arxiv_num),
    output:
        os.path.join(output_dir, "data/arxiv/embeddings/scibert-{agg_func}.feather")
    script:
        "scripts/combine_embeddings.py"

rule combine_pubmed_biobert_embeddings:
    input:
        expand(os.path.join(output_dir, "data/pubmed/embeddings/biobert/{{agg_func}}/{pubmed_num}.feather"), pubmed_num=pubmed_all),
    output:
        os.path.join(output_dir, "data/pubmed/embeddings/biobert-{agg_func}.feather")
    script:
        "scripts/combine_embeddings.py"

rule create_arxiv_scibert_embeddings:
    input:
        os.path.join(input_dir, "arxiv/baseline/{arxiv_num}.feather"),
        os.path.join(model_dir, "scibert_scivocab_uncased/pytorch_model.bin")
    output:
        os.path.join(output_dir, "data/arxiv/scibert/mean/{arxiv_num}.feather"),
        os.path.join(output_dir, "data/arxiv/scibert/median/{arxiv_num}.feather"),
    script:
        "scripts/create_bert_embeddings.py"

rule create_pubmed_biobert_embeddings:
    input:
        os.path.join(input_dir, "pubmed/baseline/{pubmed_num}.feather"),
        os.path.join(model_dir, "biobert-v1.1/pytorch_model.bin")
    output:
        os.path.join(output_dir, "data/pubmed/embeddings/biobert/mean/{pubmed_num}.feather"),
        os.path.join(output_dir, "data/pubmed/embeddings/biobert/median/{pubmed_num}.feather"),
    script:
        "scripts/create_bert_embeddings.py"

rule combine_pubmed_articles:
    input:
        expand(os.path.join(input_dir, "pubmed/baseline/{pubmed_num}.feather"), pubmed_num=pubmed_all)
    output:
        os.path.join(output_dir, "data/pubmed/corpus/articles-baseline.csv")
    script:
        "scripts/combine_articles.py"

rule create_lemmatized_arxiv_corpus:
    input:
        os.path.join(input_dir, "arxiv/baseline/{arxiv_num}.feather")
    output:
        os.path.join(input_dir, "arxiv/lemmatized/{arxiv_num}.feather")
    script:
        "scripts/lemmatize_text.py"

rule create_lemmatized_pubmed_corpus:
    input:
        os.path.join(input_dir, "pubmed/baseline/{pubmed_num}.feather")
    output:
        os.path.join(input_dir, "pubmed/lemmatized/{pubmed_num}.feather")
    script:
        "scripts/lemmatize_text.py"

rule parse_pubmed_xml:
    input: 
        os.path.join(input_dir, "pubmed/raw/pubmed22n{pubmed_num}.xml.gz")
    output:
        os.path.join(input_dir, "pubmed/baseline/{pubmed_num}.feather")
    script:
        "scripts/parse_pubmed_xml.py"

rule parse_arxiv_json:
    input:
        os.path.join(input_dir, "arxiv/raw/arxiv-metadata-oai-snapshot.json")
    output:
        os.path.join(input_dir, "arxiv/baseline/{arxiv_num}.feather")
    script:
        "scripts/parse_arxiv_json.py"

rule download_scibert:
    output:
        os.path.join(model_dir, "scibert_scivocab_uncased/pytorch_model.bin")
    shell:
        """
        cd `dirname {output}`
        cd ..
        git lfs install
        git clone https://huggingface.co/allenai/scibert_scivocab_uncased 
        """

rule download_biobert:
    output:
        os.path.join(model_dir, "biobert-v1.1/pytorch_model.bin")
    shell:
        """
        cd `dirname {output}`
        cd ..
        git lfs install
        git clone https://huggingface.co/dmis-lab/biobert-v1.1
        """

rule download_pubmed_data:
    output:
        os.path.join(input_dir, "pubmed/raw/pubmed22n{pubmed_num}.xml.gz")
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
        os.path.join(input_dir, "arxiv/raw/arxiv-metadata-oai-snapshot.json")
    shell:
        """
        cd `dirname {output}`
        kaggle datasets download -d "Cornell-University/arxiv"
        unzip arxiv.zip
        rm arxiv.zip
        """
