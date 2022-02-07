"""
Create BioBERT-based article embeddings
"""
import os
import torch
import ujson
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

model_dir = os.path.dirname(snakemake.input[1])
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# instantiate BioBERT model
model = AutoModel.from_pretrained(model_dir).eval().cuda()

# load article texts
with open(snakemake.input[0]) as fp:
    lines = fp.readlines()

articles = [ujson.loads(x) for x in lines]

# iterate over articles, and create embeddings
embeddings = []

with torch.no_grad():
    for item in tqdm(articles):
        # tokenize article title + abstract
        text = (item['title'] + item['abstract']).replace('\n', ' ')
        tokens = tokenizer.tokenize(text)
        tokens = tokenizer.encode(tokens, is_split_into_words=True)[:512]
        
        # create embedding and store
        token_tensor = torch.tensor(tokens, dtype=torch.long)
        token_embeddings, abs_embedding = model(token_tensor[None,:].cuda()).to_tuple()
        
        # compute article embedding as the mean of its word embeddings
        embedding = token_embeddings.mean(1)[0].detach().cpu().numpy()
        embeddings.append(embedding)
        
# save embedding matrix
np.savez(snakemake.output[0], np.array(embeddings))
