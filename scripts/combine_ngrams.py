"""
Combines batches of n-gram data
"""
import pandas as pd

snek = snakemake

res = pd.read_feather(snek.input[0]).set_index("ngram")

for infile in snek.input[1:]:
    df = pd.read_feather(infile).set_index("ngram")
    res = res + df

# store combined token pair counts
res.reset_index().to_feather(snek.output[0])

# extend input n-grams with those detected which meet the minimum specified frequency requirement
with open(snek.config["ngram_input"], "rt", encoding="utf-8") as fp:
    input_ngrams = fp.read().split("\n")

res = res[res.freq >= snek.config["ngram_min_freq"]]

ngram_list = list(res.index.values) + input_ngrams
ngram_list = list(set(ngram_list))

# sort by token part length
token_counts = pd.DataFrame({
    "token": ngram_list,
    "size": pd.Series(ngram_list).str.count(" ")
})

token_counts = token_counts.sort_values("size", ascending=False)

with open(snek.output[1], "wt", encoding="utf-8") as fp:
    fp.write("\n".join(token_counts.token.values))
