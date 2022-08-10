"""
Parses arxiv metadata JSON, extracts article ids, titles, and abstracts, and stores
batches of articles in separate chunked files.
"""
import os
import ujson
import pandas as pd
from datetime import datetime

with open(snakemake.input[0]) as fp:
    lines = fp.readlines()

# total number of articles
num_articles = len(lines)

# number of articles to include, per output chunk
articles_per_batch = int(num_articles / snakemake.config['arxiv']['num_chunks'])

# determine indices of articles to include in chunk
start_ind = (int(snakemake.wildcards['arxiv_num']) - 1) * articles_per_batch
end_ind = start_ind + articles_per_batch

# limit to desired range
lines = lines[start_ind:end_ind]

# load arxiv articles
ids = []
dois = []
titles = []
abstracts = []
dates = []

for line in lines:
    article = ujson.loads(line)
    title = article['title'].replace("\n", " ").strip()

    if snakemake.config['exclude_articles']['missing_title'] and title == "":
        continue

    if snakemake.config['exclude_articles']['missing_abstract'] and article['abstract'] == "":
        continue

    abstracts.append(article['abstract'])

    date_created = article['versions'][-1]['created'][5:]
    date_str = datetime.strptime(date_created, "%d %b %Y %H:%M:%S %Z").isoformat()

    ids.append(article['id'])
    dois.append(article['doi'].lower())
    titles.append(title)
    dates.append(date_str)

# save batch of articles to dataframe
dat = pd.DataFrame({
    "id": ids,
    "doi": dois,
    "title": titles,
    "abstract": abstracts,
    "date": dates
})

dat.to_feather(snakemake.output[0])

#  batch_counter = 1
#
#  start_ind = 0
#  end_ind = articles_per_batch
#
#  out_dir = os.path.dirname(snakemake.output[0])
#
#  while batch_counter <= snakemake.config['arxiv']['num_chunks']:
#      # save batch of articles to dataframe
#      dat = pd.DataFrame({
#          "id": ids[start_ind:end_ind],
#          "doi": dois[start_ind:end_ind],
#          "title": titles[start_ind:end_ind],
#          "abstract": abstracts[start_ind:end_ind]
#      })
#
#      dat.to_feather(os.path.join(out_dir, f"{batch_counter:04}.feather"))
#
#      # update counters/indices
#      start_ind = start_ind + articles_per_batch
#      end_ind = end_ind + articles_per_batch
#
#      end_ind = min(end_ind, num_articles + 1)
#
#      batch_counter = batch_counter + 1
#
