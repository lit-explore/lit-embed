#!/bin/python
"""
Summarizes word usage across all articles
"""
import pandas as pd

df = pd.read_feather(snakemake.input[0]).set_index("id")

# sum token counts for title/abstract
df = df.groupby('token').sum()

# generate a single combined sum for each token
df['total_count'] = df['title_count'] + df['abstract_count']

df = df.sort_values('total_count', ascending=False)

df.reset_index().to_feather(snakemake.output[0])

