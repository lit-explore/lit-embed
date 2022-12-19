"""
Computes title, abstract, and article-level word counts + term frequencies
"""
import pandas as pd
import re
from util.nlp import get_stop_words

# match all alphanumeric tokens;
regex = re.compile(r"[\w\d]+", re.UNICODE)

dat = pd.read_feather(snakemake.input[0]).set_index('id')

rows = []

i = 1

for article_id, article in dat.iterrows():
    # generate tokens for title & abstract;
    # texts are first converted to lowercase (already the case for lemmatized input)
    title_parts = [match.group() for match in regex.finditer(article.title.lower())]
    abstract_parts = [match.group() for match in regex.finditer(article.abstract.lower())]

    # compute counts of each word, for each section
    title_counts = pd.Series(title_parts, dtype=str).value_counts()
    abstract_counts = pd.Series(abstract_parts, dtype=str).value_counts()

    # get a list of all tokens which appear in either title/abstract
    all_tokens = sorted(list(set(title_parts + abstract_parts)))

    # number of tokens in title, abstract, or both
    title_len = int(title_counts.sum())
    abstract_len = int(abstract_counts.sum())
    total_len = title_len + abstract_len

    # iterate over token and compute counts, etc. for each
    for token in all_tokens:
        # word counts
        title_count = int(title_counts[token]) if token in title_counts else 0
        abstract_count = int(abstract_counts[token]) if token in abstract_counts else 0
        total_count = title_count + abstract_count

        # term frequency
        title_tf = (title_count / title_len) if title_len > 0 else 0
        abstract_tf = (abstract_count / abstract_len) if abstract_len > 0 else 0
        total_tf = (total_count / total_len) if total_len > 0 else 0

        rows.append({
            'id': int(article_id),
            'token': token,
            'title_count': title_count,
            'abstract_count': abstract_count,
            'total_count': total_count,
            'title_tf': title_tf,
            'abstract_tf': abstract_tf,
            'total_tf': total_tf
        })

    i += 1

# generate a dataframe and store results
res = pd.DataFrame(rows)

# exclude stop words
stop_words = get_stop_words(snakemake.config['processing'] == 'lemmatized')
res = res[~res.token.isin(stop_words)]

# set token type to "category" to reduce size
res.token = res.token.astype('category')

res.reset_index(drop=True).to_feather(snakemake.output[0])
