lit-embed
=========

Overview
--------

![BioBERT vs. TF-IDF UMAP (1m Pubmed articles)](extra/biobert_vs_tfidf_lemmatized_1m_articles.png)

`lit-embed` is a computational pipeline used to explore alternative approaches to
scientific literature embeddings, detection of maximally informative terms associated
with different areas of research, and to infer and visualize a concept-concept network
that refelects the topics of research contained in the different bodies of research.

Two main sources of open data are explored:

1. [arXiv](https://arxiv.org/)
2. [Pubmed](https://pubmed.ncbi.nlm.nih.gov/)

_Figure_: Comparison of article embeddings generated using BioBERT (left) and TF-IDF
(right) for 1 million Pubmed articles. Color corresponds to the cluster of the embedded
article.

Learning Objective
------------------

One of the main goals of the analysis is to infer a highly compressed set of terms
(either words or named entities) which, when quantified across articles, is sufficient
to preserve much of the content of the relationships between articles.

This can be useful, for example, for inferring sets of human-interpretable "keywords"
associated with articles, in an unbiased manner.

In order to infer such a set of terms, a _baseline_ or _ground truth_ for "article
similarity" be defined.

There are at least a couple possible ways this can be assessed:

1. ~Jaccard index or cosine similarity of article term (word/named entity) vectors using
   a much large set of terms.
2. Overlap, etc. of neighbors in a citation network.

The preferred approach to use is still tbd.

Both of the above approaches for assessing ground truth article similarity have
drawbacks:

The first approach is essentially using the same approach we are attempting to optimize,
but simply with less regularization.

The second approach relies on _human interpretation_ of the relationship between
different topics, and may miss meaningful connections between topics, simply due to the
lack of collaboration across different sub-fields.

Setup
-----

To begin, create a [conda](https://docs.conda.io/en/latest/) environment with the
necessary requirements, using:

```
create create -n lit-embed --file requirements.txt
```

Next, copy the example config file located in the `config/` dir to `config/config.yml`,
and modify as desired:

```
cp config/config.example.yml config/config.yml
```

The `out_dir` parameter indicates where the pipeline should store data and results.

Usage
-----

To run the main pipeline, activate the conda environment and call `snakemake`,
specifying the desired number of threads to use with the `-j` option, e.g.:

```
conda activate lit-embed
snakemake -j8
```

A separate entrypoint also exists to construct [datashader](https://datashader.org/)
visualizations using large subsets of the UMAP article projections.

Because datashader depends on earlier versions of Python, a separate conda environment
has been specified for it, which snakemake will automatically detect and use when the
`--use-conda` parameter is included:

```
snakemake -j1 --use-conda datashader
```

Due to scaling issues, the datashader rule is currently limited to UMAP projected data.

