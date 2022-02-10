"""
Computes Word Frequency Matrix 
"""
import ujson
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from util.stopwords import STOP_WORDS

# load arxiv articles
ids = []
corpus = []

with open(snakemake.input[0]) as fp:
    lines = fp.readlines()

for line in lines:
    article = ujson.loads(line)

    ids.append(article['id'])
    corpus.append(article['title'].lower() + " " + article['abstract'].lower())

# default token pattern, modifed to account for minimum token lengths
min_length = snakemake.config['tokenization']['min_length']
token_pattern = r"(?u)\b\w{" + str(min_length) + r",}\b"

# get word frequency matrix
vectorizer = CountVectorizer(max_df=snakemake.config['word_freq']['max_df'],
                             min_df=snakemake.config['word_freq']['min_df'],
                             max_features=snakemake.config['word_freq']['max_features'],
                             stop_words=STOP_WORDS,
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

# convert to column-oriented (csc) sparse matrix
#mat = mat.tocsc()
#sparse.save_npz(snakemake.output[0], mat)

# store feature names in a separate .txt file
#  with open(snakemake.output[1], 'w') as fp:
#      fp.write("\n".join(feat_names))
