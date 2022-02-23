"""
Create a single file with all article embeddings
"""
import pandas as pd

combined = pd.read_feather(snakemake.input[0])

for i, infile in enumerate(snakemake.input[1:]):
    df = pd.read_feather(infile)
    combined = pd.concat([combined, df])

df.to_feather(snakemake.output[0])
