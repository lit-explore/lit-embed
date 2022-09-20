"""
Create a single file containing all article texts;
simplifies downstream processing.

CSV is used in place of feather for this step due to memory issues with feather and
large text-based columns.
"""
import random
import pandas as pd

article_batches = snakemake.input

# if dev-mode is enabled, subsample articles
if snakemake.config['dev_mode']['enabled']:
    max_articles = snakemake.config['dev_mode']['num_articles']

    # shuffle article batches (pubmed batches are ordered by date)
    random.seed(snakemake.config['random_seed'])
    random.shuffle(article_batches)
else:
    max_articles = float('inf')

combined = pd.read_feather(article_batches[0])

num_batches = len(article_batches)

# iterate over batches of articles
for i, infile in enumerate(article_batches[1:]):
    if (i % 100 == 0) and i != 0:
        print(f"Processing article batch {i}/{num_batches}...")

    # if sub-sampling is enabled, stop once the desired article count has been reached
    if combined.shape[0] >= max_articles:
        break

    # append article batch to growing dataframe
    df = pd.read_feather(infile)
    combined = pd.concat([combined, df])

if snakemake.config['dev_mode']['enabled']:
    combined = combined.sample(max_articles, random_state=snakemake.config['random_seed'])

# if multiple versions of the same article are encountered, keep only the most recent
# record for each
num_dups = combined['id'].duplicated().sum()

if num_dups > 0:
    print(f"Dropping {num_dups} duplicated article entries")
    combined = combined[~combined['id'].duplicated(keep='last')]

# storing as plain csv for now; arrow/feather runs into memory issues with larger
# text columns
combined = combined.reset_index(drop=True)
combined.to_csv(snakemake.output[0], index=False)
