"""
Generates lemmatized version of a corpus
"""
import stanza
import pandas as pd

# initialize stanza lemmatizer
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma')

# load corpus
corpus = pd.read_feather(snakemake.input[0])

# lists to store output row parts
ids = []
lemma_texts = []

# iterate over article texts, and apply lemmatizer
for ind, article in corpus.iterrows():
    ids.append(article.id)
    doc = nlp(article.text)

    lemma_words = []

    # [ ] add "verbose" option to config..
    if ind % 100 == 0:
        print(f"Processing article {ind + 1}...")

    for sentence in doc.sentences:
        for word in sentence.words:
            lemma_words.append(word.lemma)

    lemma_texts.append(" ".join(lemma_words).replace(" .", "."))

df = pd.DataFrame({"id": ids, "text": lemma_texts})

df.to_feather(snakemake.output[0])
