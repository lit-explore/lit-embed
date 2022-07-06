"""
Create BioBERT, etc. article embeddings
"""
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

model_dir = os.path.dirname(snakemake.input[1])
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# instantiate language model
model = AutoModel.from_pretrained(model_dir).eval().cuda()

# load articles dataframe
dat = pd.read_feather(snakemake.input[0])

# sanity check
if dat.shape[0] == 0:
    raise Exception("Empty data frame encountered!")

# exclude articles with missing abstracts or titles
if snakemake.config['exclude_articles']['missing_abstract']:
    dat = dat[~dat.abstract.isna()]
if snakemake.config['exclude_articles']['missing_title']:
    dat = dat[~dat.title.isna()]

# fill missing title/abstract fields for any remaining articles with missing components
dat.title.fillna("", inplace=True)
dat.abstract.fillna("", inplace=True)

ids = dat.id.values

corpus = []

for ind, article in dat.iterrows():
    doc = article.title + " " + article.abstract
    doc = doc.replace('\n', ' ')
    corpus.append(doc)

# iterate over articles, and create embeddings
mean_embeddings = []
median_embeddings = []

with torch.no_grad():
    for doc in tqdm(corpus):
        # tokenize article title + abstract
        tokens = tokenizer.tokenize(doc)
        tokens = tokenizer.encode(tokens, is_split_into_words=True)[:512]
        
        # create embedding and store
        token_tensor = torch.tensor(tokens, dtype=torch.long)
        token_embeddings, abs_embedding = model(token_tensor[None,:].cuda()).to_tuple()
        
        # compute article embedding as the mean of its word embeddings
        mean_embeddings.append(token_embeddings.mean(1)[0].detach().cpu().numpy())
        median_embeddings.append(token_embeddings.median(1)[0].detach().cpu().numpy())

# store embeddings
mean_df = pd.DataFrame(np.vstack(mean_embeddings), index=pd.Series(ids, name='article_id'))
mean_df.columns = [f"dim_{i}" for i in mean_df.columns]

mean_df.reset_index().to_feather(snakemake.output[0])

median_df = pd.DataFrame(np.vstack(median_embeddings), index=pd.Series(ids, name='article_id'))
median_df.columns = [f"dim_{i}" for i in median_df.columns]

median_df.reset_index().to_feather(snakemake.output[1])
