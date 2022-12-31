#!/bin/env python
"""
Computes article correlations _within_ each embedding matrix.
"""
import pandas as pd
import numpy as np
from scipy.sparse import load_npz

# assign snakemake to a known variable to prevent excessive mypy, etc. warnings
snek = snakemake

# load embedding matrix
mat = load_npz(snek.input[0])

# load embedding tokens
embedding_tokens = pd.read_feather(snek.input[1])

# load embedding article ids
with open(snek.input[2], "rt", encoding="utf-8") as fp:
    article_ids = fp.read().split('\n')

# convert sparse matrix to a dense one
embed_mat = pd.DataFrame(mat.todense(),
                         index=pd.Series(article_ids, name='article_id'),
                         columns=embedding_tokens.tfidf.values)

# get article test sets
cite_mat1 = pd.read_feather(snek.input[3]).set_index('ref_pmid')
cite_mat2 = pd.read_feather(snek.input[4]).set_index('ref_pmid')
cite_mat3 = pd.read_feather(snek.input[5]).set_index('ref_pmid')

def compute_embedding_cocite_cor(X:pd.DataFrame, cite_mat:pd.DataFrame) -> pd.DataFrame:
    """
    Generates an embedding correlation matrix for the same articles present in an inpute
    co-citation matrix, and then computes the correlation of the two matrices.
    """
    pmids = cite_mat.index.values
    submat = X.loc[pmids]

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

res.to_feather(snek.output[0])
