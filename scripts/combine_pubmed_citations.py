"""
Combines batched citation dataframes
"""
import pandas as pd

# combine citation stats dataframes
citation_df = pd.read_feather(snakemake.input.citations[0])

for infile in snakemake.input.citations[1:]:
    df = pd.read_feather(infile)

    citation_df = pd.concat([citation_df, df])

# generate filtering statistics dataframe
stats_df = pd.read_feather(snakemake.input.stats[0])

for infile in snakemake.input.stats[1:]:
    df = pd.read_feather(infile)

    stats_df = pd.concat([stats_df, df])

breakpoint()

citation_df.reset_index(drop=True).to_feather(snakemake.output[0])
stats_df.reset_index(drop=True).to_feather(snakemake.output[1])
