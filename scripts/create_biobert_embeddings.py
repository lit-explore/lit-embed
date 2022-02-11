"""
Create BioBERT-based article embeddings
"""
import os
import torch
import ujson
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

model_dir = os.path.dirname(snakemake.input[1])
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# instantiate BioBERT model
model = AutoModel.from_pretrained(model_dir).eval().cuda()

# load arxiv articles
ids = []
corpus = []

with open(snakemake.input[0]) as fp:
    lines = fp.readlines()

for line in lines:
    article = ujson.loads(line)

    ids.append(article['id'])

    doc = article['title'] + " " + article['abstract']
    doc = doc.replace('\n', ' ')

    corpus.append(doc)

# iterate over articles, and create embeddings
embeddings = []

with torch.no_grad():
    for doc in tqdm(corpus):
        # tokenize article title + abstract
        tokens = tokenizer.tokenize(doc)
        tokens = tokenizer.encode(tokens, is_split_into_words=True)[:512]
        
        # create embedding and store
        token_tensor = torch.tensor(tokens, dtype=torch.long)
        token_embeddings, abs_embedding = model(token_tensor[None,:].cuda()).to_tuple()
        
        # compute article embedding as the mean of its word embeddings
        embedding = token_embeddings.mean(1)[0].detach().cpu().numpy()
        embeddings.append(embedding)

dat = pd.DataFrame(np.vstack(embeddings), 
                   index=pd.Series(ids, name='article_id'))
dat.columns = [f"dim_{i}" for i in dat.columns]

dat.reset_index().to_feather(snakemake.output[0])
