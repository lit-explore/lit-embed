"""
Combines batches of word stats into a single dataframe
"""
import pandas as pd

dfs = []

for infile in snakemake.input:
    print(f"Loading {infile}...")
    dfs.append(pd.read_feather(infile))

# remove token which don't meet minimum batch filtering requirements
#token_counts = res.groupby('token').total_count.agg(sum)

#to_keep = token_counts[token_counts >= snakemake.config["filtering"]["batch_min_count"]].index
#res = res[res.token.isin(to_keep)]

print("Combining article stat dataframes..")

combined = pd.concat(dfs)

combined.reset_index(drop=True).to_feather(snakemake.output[0])

