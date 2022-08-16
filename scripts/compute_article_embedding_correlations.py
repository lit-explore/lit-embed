#!/bin/env python
"""
Computes article correlations _within_ each embedding matrix.
"""
import pandas as pd
import numpy as np

# load embedding matrix
embed_mat = pd.read_feather(snakemake.input[0]).set_index('article_id')

# get article test sets
cite_mat1 = pd.read_feather(snakemake.input[1]).set_index('ref_pmid')
cite_mat2 = pd.read_feather(snakemake.input[2]).set_index('ref_pmid')
cite_mat3 = pd.read_feather(snakemake.input[3]).set_index('ref_pmid')

def compute_embedding_cocite_cor(embed_mat:pd.DataFrame, cite_mat:pd.DataFrame) -> pd.DataFrame:
    """
    Generates an embedding correlation matrix for the same articles present in an inpute
    co-citation matrix, and then computes the correlation of the two matrices.
    """
    pmids = cite_mat.index.values
    submat = embed_mat.loc[pmids]

    # sanity check
    if not (pd.Series(submat.index.values) == pd.Series(cite_mat.index.values)).all():
        raise Exception("Article mismatch!")
    
    embed_cor_mat = submat.T.corr()

    # compute correlation for lower triangular matrices, excluding diagonals
    ind = np.triu_indices(embed_cor_mat.shape[0], k=1)

    embed_vec = embed_cor_mat.values[ind]
    cite_vec = cite_mat.values[ind]

    return pd.DataFrame([embed_vec, cite_vec]).T.corr().values[1, 0]

cor1 = compute_embedding_cocite_cor(embed_mat, cite_mat1)
cor2 = compute_embedding_cocite_cor(embed_mat, cite_mat2)
cor3 = compute_embedding_cocite_cor(embed_mat, cite_mat3)

res = pd.DataFrame({
    "test_set": [1, 2, 3],
    "cor": [cor1, cor2, cor3]
})

res.to_feather(snakemake.output[0])
