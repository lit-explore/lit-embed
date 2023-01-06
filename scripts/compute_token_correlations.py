"""
Compute pairwise token correlations based on co-occurrence
"""
import pandas as pd

snek = snakemake

# load co-occurrence matrix
df = pd.read_feather(snek.input[0]).set_index('id')

# compute correlation matrix and store result
cor_mat = df.corr()

cor_mat.reset_index().to_feather(snek.output[0])
