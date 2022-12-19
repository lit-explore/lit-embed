"""
Generates an article x term count matrix based on words with high modified residual IDF
scores
"""
import re
import ujson
import numpy as np
import pandas as pd
import scipy
from sklearn.feature_extraction.text import CountVectorizer
from util.nlp import get_stop_words

# get stop words list
stop_words = get_stop_words(snakemake.config['processing'] == 'lemmatized')

# number of words to include
num_feats = snakemake.config['word_freq']['num_features']

# minimum character length for a token to be considered
min_length = snakemake.config['tokenization']['min_length']

# default token pattern, modifed to account for minimum token length
token_pattern = r"(?u)\b\w{" + str(min_length) + r",}\b"

# load word stats and remove stop words
word_stats = pd.read_feather(snakemake.input[1])
word_stats = word_stats[~word_stats.token.isin(stop_words)]

# get top N words, ranked by mRIDF
word_stats = word_stats.sort_values("mridf", ascending=False)
word_stats = word_stats[word_stats.token.str.len() >= min_length]

# exclude words that don't match the token pattern to prevent them from being included
# in the vocabulary
mask = [bool(re.match(token_pattern, x)) for x in word_stats.token]
word_stats = word_stats.loc[mask, :]

vocab = dict(zip(word_stats.token.values[:num_feats], range(num_feats)))

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

# count vocab words
vectorizer = CountVectorizer(max_df=snakemake.config['word_freq']['max_df'],
                             min_df=snakemake.config['word_freq']['min_df'],
                             stop_words=stop_words,
                             vocabulary=vocab,
                             dtype=np.dtype(snakemake.config['word_freq']['dtype']),
                             token_pattern=token_pattern)

mat = vectorizer.fit_transform(corpus)

# get feature names list
feat_names = vectorizer.get_feature_names_out()

# for now, convert to dense matrix for convenience; the "word_freq.max_features" config 
# param can be used to limit the size of the matrix
mridf_mat = pd.DataFrame(mat.todense(), 
                         index=pd.Series(ids, name='article_id'), 
                         columns=feat_names)

mridf_mat = mridf_mat.reset_index()

# explicitly convert "article_id" column to a string
# work-around for pandas/arrow serialization issue
mridf_mat['article_id'] = mridf_mat['article_id'].apply(str)

# sanity check: make sure no duplicated ids are found
if mridf_mat.article_id.duplicated().sum() > 0:
    raise Exception("Encountered duplicate article IDs!")

mridf_mat.to_feather(snakemake.output[0])
