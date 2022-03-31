"""
Science literature embedding pipeline
"""
import os

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
targets = ['articles', 'topics']

rule all:
    input:
        expand(os.path.join(output_dir, "fig/{source}/{target}/{projection}/tfidf-{processing}-scatterplot.png"), source=data_sources, processing=processing_versions, target=targets, projection=projection_types),
        expand(os.path.join(output_dir, "fig/pubmed/{target}/{projection}/biobert-{agg_func}-scatterplot.png"), agg_func=agg_funcs, target=targets, projection=projection_types)

rule datashader:
    input:
        expand(os.path.join(output_dir, "fig/{source}/articles/umap/tfidf-{processing}-datashader.png"), source=data_sources, processing=processing_versions),
        expand(os.path.join(output_dir, "fig/pubmed/articles/umap/biobert-{agg_func}-datashader.png"), agg_func=agg_funcs)

rule plot_tfidf_datashader:
    input:
        os.path.join(output_dir, "data/{source}/articles/umap/tfidf-{processing}.feather"),
        os.path.join(output_dir, "data/{source}/articles/tfidf-{processing}-clusters.feather"),
    output:
        os.path.join(output_dir, "fig/{source}/articles/umap/tfidf-{processing}-datashader.png"),
    params:
        name="TF-IDF"
    conda:
        "envs/datashader.yml"
    script:
        "scripts/plot_datashader.py"

rule plot_biobert_datashader:
    input:
        os.path.join(output_dir, "data/pubmed/articles/umap/biobert-{agg_func}.feather"),
        os.path.join(output_dir, "data/pubmed/articles/biobert-{agg_func}-clusters.feather"),
    output:
        os.path.join(output_dir, "fig/pubmed/articles/umap/biobert-{agg_func}-datashader.png"),
    params:
        name="BioBERT"
    conda:
        "envs/datashader.yml"
    script:
        "scripts/plot_datashader.py"

rule plot_tfidf_scatterplot:
    input:
        os.path.join(output_dir, "data/{source}/{target}/{projection}/tfidf-{processing}.feather"),
        os.path.join(output_dir, "data/{source}/{target}/tfidf-{processing}-clusters.feather"),
    output:
        os.path.join(output_dir, "fig/{source}/{target}/{projection}/tfidf-{processing}-scatterplot.png"),
    params:
        name="TF-IDF"
    script:
        "scripts/plot_scatter.py"

rule plot_biobert_scatterplot:
    input:
        os.path.join(output_dir, "data/pubmed/{target}/{projection}/biobert-{agg_func}.feather"),
        os.path.join(output_dir, "data/pubmed/{target}/biobert-{agg_func}-clusters.feather"),
    output:
        os.path.join(output_dir, "fig/pubmed/{target}/{projection}/biobert-{agg_func}-scatterplot.png"),
    params:
        name="BioBERT"
    script:
        "scripts/plot_scatter.py"

rule tfidf_dimension_reduction:
    input:
        os.path.join(output_dir, "data/{source}/tfidf-{processing}.feather"),
    output:
        os.path.join(output_dir, "data/{source}/{target}/{projection}/tfidf-{processing}.feather"),
    script:
        "scripts/reduce_dimension.py"

rule biobert_dimension_reduction:
    input:
        os.path.join(output_dir, "data/pubmed/biobert-{agg_func}.feather")
    output:
        os.path.join(output_dir, "data/pubmed/{target}/{projection}/biobert-{agg_func}.feather"),
    script:
        "scripts/reduce_dimension.py"

rule compute_tfidf_clusters:
    input:
        os.path.join(output_dir, "data/{source}/tfidf-{processing}.feather"),
    output:
        os.path.join(output_dir, "data/{source}/{target}/tfidf-{processing}-clusters.feather"),
    script: "scripts/cluster_kmeans.py"


rule compute_biobert_embedding_clusters:
    input:
        os.path.join(output_dir, "data/pubmed/biobert-{agg_func}.feather")
    output:
        os.path.join(output_dir, "data/pubmed/{target}/biobert-{agg_func}-clusters.feather"),
    script: "scripts/cluster_kmeans.py"

rule compute_tfidf_matrix:
    input:
       os.path.join(output_dir, "data/{source}/articles-{processing}.csv") 
    output:
        os.path.join(output_dir, "data/{source}/tfidf-{processing}.feather"),
        os.path.join(output_dir, "data/{source}/tfidf-{processing}-sparse-mat.npz"),
        os.path.join(output_dir, "data/{source}/tfidf-{processing}-stats.feather"),
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
        expand(os.path.join(input_dir, "pubmed/lemmatized/{pubmed_num}.feather"), pubmed_num=pubmed_all)
    output:
        os.path.join(output_dir, "data/pubmed/articles-lemmatized.csv")
    script:
        "scripts/combine_articles.py"

# combine articles and sub-sample, if enabled
rule combine_pubmed_articles:
    input:
        expand(os.path.join(input_dir, "pubmed/orig/{pubmed_num}.feather"), pubmed_num=pubmed_all)
    output:
        os.path.join(output_dir, "data/pubmed/articles-baseline.csv")
    script:
        "scripts/combine_articles.py"

rule combine_embeddings:
    input:
        expand(os.path.join(output_dir, "data/pubmed/biobert/{{agg_func}}/{pubmed_num}.feather"), pubmed_num=pubmed_all),
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
