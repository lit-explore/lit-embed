"""
Generates lemmatized version of a corpus
"""
import stanza
import pandas as pd

# initialize stanza lemmatizer
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma')

# load corpus
corpus = pd.read_feather(snakemake.input[0])

corpus.abstract.fillna("", inplace=True)
corpus.title.fillna("", inplace=True)

# lists to store output row parts
ids = []
dois = []
lemma_titles = []
lemma_abstracts = []

# iterate over article texts, and apply lemmatizer
for ind, article in corpus.iterrows():
    ids.append(article.id)
    dois.append(article.doi)

    # lemmatize title
    text = article.title.lower()
    doc = nlp(text)

    lemma_words = []

    # [ ] add "verbose" option to config..
    if ind % 100 == 0:
        print(f"Processing article {ind + 1}...")

    for sentence in doc.sentences:
        for word in sentence.words:
            lemma_words.append(word.lemma)

    lemma_titles.append(" ".join(lemma_words).replace(" .", "."))

    # lemmatize abstract
    text = article.abstract.lower()

    # stanza work-around; fails if text is *exactly* the length of the batch size (3000)
    if len(text) == 3000:
        text = text + "."

    doc = nlp(text)

    lemma_words = []

    for sentence in doc.sentences:
        for word in sentence.words:
            lemma_words.append(word.lemma)

    # remove extra period added for work-around, if present
    if len(text) == 3001:
        lemma_words = lemma_words[:-1]

    lemma_abstracts.append(" ".join(lemma_words).replace(" .", "."))

df = pd.DataFrame({
    "id": ids, 
    "doi": dois, 
    "title": lemma_titles, 
    "abstract": lemma_abstracts
})

#  df.to_feather(snakemake.output[0])
df.to_feather(snakemake.output[0])
