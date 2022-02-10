"""
summary of word counts across all articles
"""
import pandas as pd

dat = pd.read_feather(snakemake.input[0]).set_index('article_id')

total_counts = dat.sum().sort_values(ascending=False).reset_index()
total_counts.columns = ['word', 'n']

total_counts.to_feather(snakemake.output[0])
