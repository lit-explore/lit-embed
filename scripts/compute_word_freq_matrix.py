"""
Computes Word Frequency Matrix 
"""
import ujson
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from util.nlp import get_stop_words

# get stop words lsit
stop_words = get_stop_words(snakemake.wildcards['processing'] == 'lemmatized')

# load articles
dat = pd.read_csv(snakemake.input[0])

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

# default token pattern, modifed to account for minimum token lengths
min_length = snakemake.config['tokenization']['min_length']
token_pattern = r"(?u)\b\w{" + str(min_length) + r",}\b"

# get word frequency matrix
vectorizer = CountVectorizer(max_df=snakemake.config['word_freq']['max_df'],
                             min_df=snakemake.config['word_freq']['min_df'],
                             max_features=snakemake.config['word_freq']['num_features'],
                             dtype=np.dtype(snakemake.config['word_freq']['dtype']),
                             stop_words=stop_words,
                             token_pattern=token_pattern)

mat = vectorizer.fit_transform(corpus)

# get feature names list
feat_names = vectorizer.get_feature_names_out()

# for now, convert to dense matrix for convenience; the "word_freq.max_features" config 
# param can be used to limit the size of the matrix
dat = pd.DataFrame(mat.todense(), 
                   index=pd.Series(ids, name='article_id'), 
                   columns=feat_names)

dat.reset_index().to_feather(snakemake.output[0])
