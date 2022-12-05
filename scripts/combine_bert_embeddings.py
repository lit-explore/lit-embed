"""
Combine article BERT embeddings into a single file
"""
import pandas as pd

combined = pd.read_feather(snakemake.input[0])

for i, infile in enumerate(snakemake.input[1:]):
    df = pd.read_feather(infile)
    combined = pd.concat([combined, df])

# if duplicate article ids are present (e.g. due to multiple versions), keep the most
# recent record for each article
combined = combined[~combined.article_id.duplicated(keep='last')]

combined.reset_index(drop=True).to_feather(snakemake.output[0])
