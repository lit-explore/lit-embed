"""
Computes TF-IDF version of word frequencies
"""
import ujson
import scipy.sparse as sparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# load arxiv articles
corpus = []

with open(snakemake.input[0]) as fp:
    lines = fp.readlines()

for line in lines:
    article = ujson.loads(line)

    corpus.append(article['title'].lower() + " " + article['abstract'].lower())

# get tf-idf sparse matrix
vectorizer = TfidfVectorizer()
mat = vectorizer.fit_transform(corpus)

# get feature names list
feat_names = vectorizer.get_feature_names_out()

# convert to column-oriented (csc) sparse matrix
mat = mat.tocsc()

# drop all columns (tokens) with fewer than N matches in the corpus
col_sums = pd.Series(mat.nonzero()[1]).value_counts()
cols_to_keep = col_sums[col_sums >= snakemake.config['min_word_freq']].index

mat = mat[:, cols_to_keep]
feat_names = [feat_names[i] for i in cols_to_keep]

sparse.save_npz(snakemake.output[0], mat)

# store feature names in a separate .txt file
with open(snakemake.output[1], 'w') as fp:
    fp.write("\n".join(feat_names))
