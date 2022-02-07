"""
Generate a global count of all words appearing in article titles/abstracts.

Eventually, it would also be useful to generate a sparse matrix representation of a
complete article x [word|term] matrix.
"""
import ujson
import pandas as pd
from util.stopwords import STOP_WORDS

word_counts = {}

with open(snakemake.input[0]) as fp:
    lines = fp.readlines()

for line in lines:
    article = ujson.loads(line)

    text = article["title"].lower() + article["abstract"].lower()

    tokens = [x for x in text.split() if x not in STOP_WORDS]

    for token in tokens:
        if token in word_counts:
            word_counts[token] += 1
        else:
            word_counts[token] = 1

df = pd.DataFrame.from_dict(word_counts, orient="index").reset_index()
df.columns = ["word", "n"]

df = df.sort_values("n", ascending=False)

# exclude any words which only appear once
df = df[df.n >= snakemake.config["min_word_freq"]]

df.reset_index(drop=True).to_csv(snakemake.output[0])
