"""
Create BioBERT, etc. article embeddings
"""
import os
import torch
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

snek = snakemake

# create BERT tokenizer
# https://huggingface.co/docs/transformers/v4.20.1/en/model_doc/bert#transformers.BertTokenizer
tokenizer = AutoTokenizer.from_pretrained(snek.config["bert"]["model"])
model = AutoModelForMaskedLM.from_pretrained(snek.config["bert"]["model"], 
                                             output_hidden_states=True)

max_len:int = snek.config["bert"]["max_length"]

# load articles dataframe
dat = pd.read_feather(snek.input[0])

# sanity check
if dat.shape[0] == 0:
    raise Exception("Empty data frame encountered!")

# exclude articles with missing abstracts or titles
if snek.config['exclude_articles']['missing_abstract']:
    dat = dat[~dat.abstract.isna()]
if snek.config['exclude_articles']['missing_title']:
    dat = dat[~dat.title.isna()]

# fill missing title/abstract fields for any remaining articles with missing components
dat.title.fillna("", inplace=True)
dat.abstract.fillna("", inplace=True)

# combine title & abstract for each article to construct corpus
corpus = []

for ind, article in dat.iterrows():
    doc = article.title + " " + article.abstract
    doc = doc.replace('\n', ' ')
    corpus.append(doc)

# iterate over articles, and create embeddings
mean_embeddings = []

# use gpu, if available
# https://github.com/huggingface/transformers/issues/2704
device = "cuda:0" if torch.cuda.is_available() else "cpu"

with torch.no_grad():

    model = model.to(device)

    for doc in tqdm(corpus):
        sentences = nltk.sent_tokenize(doc)

        inputs = tokenizer(sentences, padding=True, max_length=max_len, truncation=True,
                           return_tensors='pt').to(device)

        try:
            outputs = model(**inputs)
        except torch.cuda.OutOfMemoryError:
            # fallback to cpu
            print("========================================================================")
            print("GPU OOM! Falling back on CPU..")
            print("Text:")
            print(doc)
            print("========================================================================")
            model = model.to("cpu")
            inputs = inputs.to("cpu")

            outputs = model(**inputs)

            model = model.to(device)

        # get second-from-last hidden state vector
        # https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#35-pooling-strategy--layer-choice
        token_embeddings = outputs.hidden_states[-2]

        # compute sentence-level mean embedding vectors
        # https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Pooling.py
        input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_embeddings.append(sentence_embeddings.mean(0).tolist())
        
# store embeddings
mean_df = pd.DataFrame(np.vstack(mean_embeddings), 
                       index=pd.Series(dat.id.values, name='article_id'))
mean_df.columns = [f"dim_{i}" for i in mean_df.columns]

mean_df.reset_index().to_parquet(snek.output[0])
