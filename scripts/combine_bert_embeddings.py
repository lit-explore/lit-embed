"""
Combine article BERT embeddings into a single file
"""
import pandas as pd

snek = snakemake

combined = pd.read_parquet(snek.input[0])

for i, infile in enumerate(snek.input[1:]):
    df = pd.read_parquet(infile)
    combined = pd.concat([combined, df])

if snek.config["corpus"] == "pubmed":
    combined.article_id = combined.article_id.astype(int)   

# if duplicate article ids are present (e.g. due to multiple versions), keep the most
# recent record for each article
num_before = combined.shape[0]

combined = combined[~combined.article_id.duplicated(keep='last')]

num_after = combined.shape[0]

if num_before != num_after:
    print("Duplicate articles detected in BERT embeddings!")

combined.reset_index(drop=True).to_parquet(snek.output[0])
