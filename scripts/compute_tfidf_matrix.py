"""
Computes TF-IDF version of word frequencies
"""
import ujson
import numpy as np
import pandas as pd
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from util.nlp import get_stop_words

# get stop words list
stop_words = get_stop_words(snakemake.config['processing'] == 'lemmatized')

# load articles
dtypes = {'id': 'str', 'doi': 'str', 'title': 'str', 'abstract': 'str', 'date': 'str'}
dat = pd.read_csv(snakemake.input[0], dtype=dtypes)

# exclude articles with missing abstracts or titles
if snakemake.config['exclude_articles']['missing_abstract']:
    dat = dat[~dat.abstract.isna()]
if snakemake.config['exclude_articles']['missing_title']:
    dat = dat[~dat.title.isna()]

# fill missing title/abstract fields for any remaining articles with missing components
dat.title.fillna("", inplace=True)
dat.abstract.fillna("", inplace=True)

ids = dat.id.values

# combine lowercase title + abstract to form corpus
corpus = []

for index, row in dat.iterrows():
    text = (row.title + " " + row.abstract).lower()    
    corpus.append(text)

del dat

# default token pattern, modifed to account for minimum token lengths
min_length = snakemake.config['tokenization']['min_length']

token_pattern = r"(?u)\b\w{" + str(min_length) + r",}\b"

# get tf-idf sparse matrix
#  max_features=snakemake.config['word_freq']['max_features'],
vectorizer = TfidfVectorizer(max_df=snakemake.config['word_freq']['max_df'],
                             min_df=snakemake.config['word_freq']['min_df'],
                             stop_words=stop_words,
                             dtype=np.dtype(snakemake.config['word_freq']['dtype']),
                             token_pattern=token_pattern)

mat = vectorizer.fit_transform(corpus)

# save full sparse matrix prior to filtering
scipy.sparse.save_npz(snakemake.output[1], mat)

# get feature names list
feat_names = vectorizer.get_feature_names_out()

# determine mean tf-idf scores & number of articles each term found for
article_counts = []
mean_tfidfs = []
max_tfidfs = []

for i in range(mat.shape[1]):
    feat_vec = mat[:, i]
    article_counts.append(feat_vec.nnz)
    mean_tfidfs.append(feat_vec.mean())
    max_tfidfs.append(feat_vec.max())

mean_tfidfs = pd.Series(mean_tfidfs)

# normalize contribution of idf/mean tfidf scores; arbitrary 1e9 multiplier used for
# convenience / human readability
scaled_idf = (vectorizer.idf_/vectorizer.idf_.sum()) * 1e9
scaled_mean_tfidf = (mean_tfidfs / mean_tfidfs.sum()) * 1e9

# weights for combining IDF & mean TF-IDF contributions
tfidf_coef = snakemake.config['word_freq']['alpha']
idf_coef = 1 - tfidf_coef

# assign a score to each term based on a weighted combination IDF & mean TF-IDF

# store idf scores, for each term
stats = pd.DataFrame({
    "feature": feat_names,
    "idf": vectorizer.idf_,
    "idf_scaled": scaled_idf,
    "num_articles": article_counts,
    "mean_tfidf": mean_tfidfs,
    "mean_tfidf_scaled": scaled_mean_tfidf,
    "max_tfidf": max_tfidfs,
    "score": (tfidf_coef * scaled_mean_tfidf) + (idf_coef * scaled_idf)
})

stats.to_feather(snakemake.output[2])

# get indices of top N features
num_feats = snakemake.config['word_freq']['num_features']
top_features = stats.sort_values("score", ascending=False).head(num_feats).feature

mask = pd.Series(feat_names).isin(top_features)

# for now, convert to dense matrix for convenience; the "word_freq.max_features" config 
# param can be used to limit the size of the matrix
tfidf_mat = pd.DataFrame(mat[:, mask].todense(), 
                         index=pd.Series(ids, name='article_id'), 
                         columns=feat_names[mask])

tfidf_mat = tfidf_mat.reset_index()

# explicitly convert "article_id" column to a string
# work-around for pandas/arrow serialization issue
tfidf_mat['article_id'] = tfidf_mat['article_id'].apply(str)

# sanity check: make sure no duplicated ids are found
if tfidf_mat.article_id.duplicated().sum() > 0:
    raise Exception("Encountered duplicate article IDs!")

tfidf_mat.to_feather(snakemake.output[0])
