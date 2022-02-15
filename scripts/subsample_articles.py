"""
choose a random subset of articles for testing
"""
import pandas as pd

dat = pd.read_feather(snakemake.input[0])

num_articles = snakemake.config['dev_mode']['num_articles']
seed = snakemake.config['random_seed']

dat = dat.sample(num_articles, random_state=seed)

dat.reset_index(drop=True).to_feather(snakemake.output[0])
