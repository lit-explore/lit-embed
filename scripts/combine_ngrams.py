"""
Combines batches of n-gram data
"""
import pandas as pd

snek = snakemake

res = pd.read_feather(snek.input[0]).set_index("ngram")

for infile in snek.input[1:]:
    df = pd.read_feather(infile).set_index("ngram")
    res = res + df

res.reset_index().to_feather(snek.output[0])
