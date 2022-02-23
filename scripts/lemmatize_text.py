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
    if ind % 100 == 0:
        print(f"Processing article {ind + 1}...")

    # moved upstream..
    #  if snakemake.config['exclude_articles']['missing_title'] and article.title == "":
    #      continue
    #
    #  if snakemake.config['exclude_articles']['missing_abstract'] and article.abstract == "":
    #      continue

    ids.append(article.id)
    dois.append(article.doi)

    # process title and abstract together to reduce overhead;
    # newlines have been previously removed from titles to ensure that each
    # title contains only a single sentence
    text = article.title.lower() + "\n\n" + article.abstract.lower()

    # stanza work-around; fails if text is *exactly* the length of the batch size (3000)
    if len(text) == 3000:
        text = text + "."

    # if both title & abstract are empty, skip lemmatization
    if text == "\n\n":
        lemma_titles.append("")
        lemma_abstracts.append("")

        continue

    # lemmatize title
    doc = nlp(text)

    # extract title (first sentence in output)
    title_words = [word.lemma for word in doc.sentences[0].words]
    lemma_titles.append(" ".join(title_words))

    # extract abstract
    abstract_words = []

    for sentence in doc.sentences[1:]:
        for word in sentence.words:
            abstract_words.append(word.lemma)

    # remove extra period added for work-around, if present
    if len(text) == 3001:
        abstract_words = abstract_words[:-1]

    abstract = " ".join(abstract_words).replace(" .", ".")

    lemma_abstracts.append(abstract)

df = pd.DataFrame({
    "id": ids, 
    "doi": dois, 
    "title": lemma_titles, 
    "abstract": lemma_abstracts
})

#  df.to_feather(snakemake.output[0])
df.to_feather(snakemake.output[0])
