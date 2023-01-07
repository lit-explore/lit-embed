"""
Modifies input corpus to add underscores between parts of detected n-grams.
"""
import re
import pandas as pd

snek = snakemake

df = pd.read_feather(snek.input[0])

ngrams = pd.read_feather(snek.input[1]).sort_values("freq", ascending=False)

# filter out low frequency token pairs
ngrams = ngrams[ngrams.freq >= snek.config["ngram_min_freq"]]

def rep_func(match:re.Match):
    """function to handle replacements, preserving original case"""
    return match.group(1) + "_" + match.group(2)

# iterate over n-grams and replace instances in input texts
for ngram in ngrams.ngram:
    # create regex to detect whitespace separated n-gram parts
    lhs, rhs = ngram.split()
    regex = re.compile(f"\\b({lhs})\\s+({rhs})", flags=re.I)

    # replace regex instances in title/abstract
    df.title = df.title.str.replace(regex, rep_func)
    df.abstract = df.abstract.str.replace(regex, rep_func)

# save result
df.to_feather(snek.output[0])

