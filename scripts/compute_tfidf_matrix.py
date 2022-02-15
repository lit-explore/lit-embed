"""
Computes TF-IDF version of word frequencies
"""
import ujson
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from util.nlp import STOP_WORDS_LEMMA

# load articles
dat = pd.read_feather(snakemake.input[0])

ids = dat.id.values
corpus = dat.text.values

# default token pattern, modifed to account for minimum token lengths
min_length = snakemake.config['tokenization']['min_length']
token_pattern = r"(?u)\b\w{" + str(min_length) + r",}\b"

# get tf-idf sparse matrix
vectorizer = TfidfVectorizer(max_df=snakemake.config['word_freq']['max_df'],
                             min_df=snakemake.config['word_freq']['min_df'],
                             max_features=snakemake.config['word_freq']['max_features'],
                             stop_words=STOP_WORDS_LEMMA,
                             token_pattern=token_pattern)

mat = vectorizer.fit_transform(corpus)

# get feature names list
feat_names = vectorizer.get_feature_names_out()

# for now, convert to dense matrix for convenience; the "word_freq.max_features" config 
# param can be used to limit the size of the matrix
tfidf_mat = pd.DataFrame(mat.todense(), 
                   index=pd.Series(ids, name='article_id'), 
                   columns=feat_names)

tfidf_mat.reset_index().to_feather(snakemake.output[0])
