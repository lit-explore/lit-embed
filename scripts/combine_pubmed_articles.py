"""
combined separate pubmed article dataframes into a single dataframe
"""
import pandas as pd

combined = pd.read_feather(snakemake.input[0])

# if dev-mode is enabled, subsample articles
if snakemake.config['dev_mode']['enabled']:
    max_articles = snakemake.config['dev_mode']['num_articles']
else:
    max_articles = float('inf') 

for infile in snakemake.input[1:]:
    if combined.shape[0] >= max_articles:
        break

    df = pd.read_feather(infile)
    combined = pd.concat([combined, df])

if snakemake.config['dev_mode']['enabled']:
    combined = combined.sample(max_articles, random_state=snakemake.config['random_seed'])

combined.reset_index(drop=True).to_feather(snakemake.output[0])
