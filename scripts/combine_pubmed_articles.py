"""
combined separate pubmed article dataframes into a single dataframe
"""
import random
import pandas as pd

article_batches = snakemake.input

# if dev-mode is enabled, subsample articles
if snakemake.config['dev_mode']['enabled']:
    max_articles = snakemake.config['dev_mode']['num_articles']

    # shuffle article batches to avoid only including older articles
    random.seed(snakemake.config['random_seed'])
    random.shuffle(article_batches)
else:
    max_articles = float('inf') 


combined = pd.read_feather(article_batches[0])

num_batches = len(article_batches)

# iterate over batches of pubmed articles
for i, infile in enumerate(article_batches[1:]):
    if i % 100 == 0:
        print(f"Processing pubmed article batch {i}/{num_batches}...")

    # if sub-sampling is enabled, stop once the desired article count has been reached
    if combined.shape[0] >= max_articles:
        break

    # append article batch to growing dataframe
    df = pd.read_feather(infile)
    combined = pd.concat([combined, df])

print("Finished combining pubmed dataframes; saving result..")

if snakemake.config['dev_mode']['enabled']:
    combined = combined.sample(max_articles, random_state=snakemake.config['random_seed'])

combined.reset_index(drop=True).to_feather(snakemake.output[0])
