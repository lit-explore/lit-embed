"""
Computes TF-IDF version of word frequencies
"""
import ujson
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from util.nlp import STOP_WORDS, STOP_WORDS_LEMMA

# determine version of stop words to use
stop_words = STOP_WORDS if snakemake.wildcards['processing'] == 'baseline' else STOP_WORDS_LEMMA

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
vectorizer = TfidfVectorizer(max_df=snakemake.config['word_freq']['max_df'],
                             min_df=snakemake.config['word_freq']['min_df'],
                             max_features=snakemake.config['word_freq']['max_features'],
                             stop_words=stop_words,
                             token_pattern=token_pattern)

mat = vectorizer.fit_transform(corpus)

# get feature names list
feat_names = vectorizer.get_feature_names_out()

# for now, convert to dense matrix for convenience; the "word_freq.max_features" config 
# param can be used to limit the size of the matrix
tfidf_mat = pd.DataFrame(mat.todense(), 
                   index=pd.Series(ids, name='article_id'), 
                   columns=feat_names)

tfidf_mat = tfidf_mat.reset_index()

# explicitly convert "article_id" column to a string
# work-around for pandas/arrow serialization issue
tfidf_mat['article_id'] = tfidf_mat['article_id'].apply(str)

tfidf_mat.to_feather(snakemake.output[0])
