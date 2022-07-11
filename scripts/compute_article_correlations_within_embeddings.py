#!/bin/env python
"""
Computes article correlations _within_ each embedding matrix.
"""
import pandas as pd
import numpy as np
#from scipy.stats import pearsonr

embed_mat = pd.read_feather(snakemake.input[0]).set_index('article_id')

# corrwith doesn't scale
# df2.columns = df1.columns
# cor_mat = df1.corrwith(df2, axis=1)

# choose a random subsample of N articles
rng = np.random.default_rng(snakemake.config['random_seed'])
embed_mat = embed_mat.sample(random_state = rng)

cor_mat = embed_mat.T.corr()

cor_mat.to_feather(snakemake.output[0])
