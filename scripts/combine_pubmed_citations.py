"""
Combines batched citation dataframes
"""
import pandas as pd

# combine citation mapping dataframes
cite_df = pd.read_feather(snakemake.input.citations[0])

for infile in snakemake.input.citations[1:]:
    df = pd.read_feather(infile)

    cite_df = pd.concat([cite_df, df])

# combine citation stats dataframes
stats_df = pd.read_feather(snakemake.input.stats[0])

for infile in snakemake.input.stats[1:]:
    df = pd.read_feather(infile)

    stats_df = pd.concat([stats_df, df])

# add number times article cited to stats dataframe
try:
    ref_counts = cite_df.ref_pmid.value_counts()
    ref_counts = ref_counts.to_frame().reset_index().rename(columns={
        'index': 'pmid', 'ref_pmid': 'times_cited'
    })

    #ref_counts = ref_counts[ref_counts.index.isin(stats_df.pmid)]
    stats_df = stats_df.merge(ref_counts, on=['pmid'])
except:
    breakpoint()

cite_df.reset_index(drop=True).to_feather(snakemake.output[0])
stats_df.reset_index(drop=True).to_feather(snakemake.output[1])
