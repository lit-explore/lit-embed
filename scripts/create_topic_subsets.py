"""
Retrieves a random subset of articles relating to one or more specific sub-topics.
"""
import os
import ujson
from typing import List

article_subset: List[str] = []

with open(snakemake.input[0]) as fp:
    lines = fp.readlines()

for line in lines:
    article = ujson.loads(line)

    if len(article_subset) == (snakemake.config['topic_subsets']['max_articles_per_topic'] + 1):
        break
                
    article_text = (article['title'] + " " + article['abstract']).lower()

    if snakemake.wildcards['topic'] in article_text:
        article_subset.append(ujson.dumps(article))

# store result
with open(snakemake.output[0], 'w') as fp:
    fp.write("\n".join(article_subset))
