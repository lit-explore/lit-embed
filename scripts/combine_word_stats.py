"""
Combines batches of word stats into a single dataframe
"""
import pandas as pd

combined = pd.read_feather(snakemake.input[0])

for infile in snakemake.input[1:]:
    dat = pd.read_feather(infile)
    combined = pd.concat([combined, dat])

combined.reset_index(drop=True).to_feather(snakemake.output[0])

