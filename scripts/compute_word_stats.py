"""
summary of word counts, etc. across all articles

todo:

- [ ] perform for both normal/lemmatized docs

later (?)

steps:

- split title/abstract
- remove all single character tokens
- add entries to table containing:
    <article_id, title_count, abstract_count>

downstream:

1. [ ] summarize findings
    - distribution of token counts?
    - tokens with highest occurrence?
2. [ ]  create a _filtered_ version of word/token freqs:
    - preserve initial "full" dataset, but create a separate rule to generate a filtered
      set of terms that appear frequently enough s.t. they might actually be useful
        - filter out token that appear < n times (e.g. "2" to begin with)
    - either here/in a subsequent step, may also want to filter out words at the other
      end (common/stop words)
    - generate new stats for filtered data?
3. [ ] compute IDF, etc. for filtered "candidate" words?
"""
import pandas as pd
from gensim.utils import tokenize

dat = pd.read_feather(snakemake.input[0]).set_index('id')

rows = []

i = 1

for article_id, article in dat.iterrows():
    #  print(f"Processing article {i}/{dat.shape[0]}...")

    # generate tokens for title & abstract;
    # texts are first converted to lowercase (already the case for lemmatized input)
    title_parts = list(tokenize(article.title.lower()))
    abstract_parts = list(tokenize(article.abstract.lower()))

    # compute counts of each word, for each section
    title_counts = pd.Series(title_parts).value_counts()
    abstract_counts = pd.Series(abstract_parts).value_counts()

    # get a list of all tokens which appear in either title/abstract
    all_tokens = sorted(list(set(title_parts + abstract_parts)))

    for token in all_tokens:
        rows.append({
            'id': article_id,
            'token': token,
            'title_count': title_counts[token] if token in title_counts else 0,
            'abstract_count': abstract_counts[token] if token in abstract_counts else 0
        })

    i += 1

# generate a dataframe and store results
res = pd.DataFrame(rows)
res.to_feather(snakemake.output[0])
