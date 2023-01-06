"""
Extracts token pairs with correlation coefficients above a specified threshold
"""
import numpy as np
import pandas as pd

snek = snakemake

# load token correlation matrix
df = pd.read_feather(snek.input[0]).set_index("token")

# set diagonal to 0
df.values[tuple([np.arange(df.shape[0])] * 2)] = 0

# list to keep track of correlated token pairs
token_pairs:list[tuple[str,str,float]] = []

# get indices for upper triangular matrix
ind = np.triu_indices(df.shape[0], k=1)

# correlation cutoff
TOKEN_MIN_COR:float = snek.config["token_min_cor"]

# iterate over token pairs
for i in range(len(ind[0])):
    ind1 = ind[0][i]
    ind2 = ind[1][i]

    cor = df.iloc[ind1, ind2]

    if cor >= TOKEN_MIN_COR:
        t1 = df.index[ind1]
        t2 = df.index[ind2]

        token_pairs.append((t1, t2, cor,))

# create dataframe with matched token pairs and store result
res = pd.DataFrame.from_records(token_pairs, columns =['token1', 'token2', 'cor'])

res.to_feather(snek.output[0])
