"""
Dimension reduction
"""
import random
import scipy
import umap
import numpy as np
import pandas as pd

snek = snakemake

random.seed(snek.config["random_seed"])

# load data
#dat = pd.read_feather(snek.input[0]).set_index("article_id")

if snek.input[0].endswith(".npz"):
    dat = scipy.sparse.load_npz(snek.input[0])

    with open(snek.input[1], "rt", encoding="utf-8") as fp:
        article_ids = pd.Series(fp.read().split())
else:
    dat = pd.read_parquet(snek.input[0]).set_index("article_id")

# subsample articles?
max_articles = snek.config["umap"]["max_articles"]

#  if  max_articles < dat.shape[0]:
#      ind = random.sample(range(dat.shape[0]), max_articles)
#      dat = dat.iloc[ind, :]

if max_articles < dat.shape[0]:
    ind = random.sample(range(dat.shape[0]), max_articles)

    print(f"Sub-sampling {max_articles} articles...")

    if isinstance(dat, pd.DataFrame):
        # dataframe
        dat = dat.iloc[ind]
    elif isinstance(dat, scipy.sparse.csr_matrix):
        # sparse matrix
        dat = dat[ind, :]
        article_ids = article_ids[ind]

# convert sparse matrix to dense
if isinstance(dat, scipy.sparse.csr_matrix):
    print("Converting sparse matrix to dense matrix")
    dat = pd.DataFrame(dat.todense(), index=article_ids)

# remove any articles with zero variance
print("Checking for articles with zero variance..")

mask = dat.apply(np.var, axis=1) > 0

num_zero_var = dat.shape[0] - mask.sum()

if num_zero_var > 0:
    print(f"Removing {num_zero_var} articles with zero variance prior to projection.")
    dat = dat[mask]

# generate projection
reducer = umap.UMAP(
    n_neighbors=snek.config["umap"]["n_neighbors"],
    min_dist=snek.config["umap"]["min_dist"],
    metric=snek.config["umap"]["metric"],
    densmap=snek.config["umap"]["densmap"],
    random_state=snek.config["random_seed"]
)

embedding = reducer.fit_transform(dat)
embedding_index = dat.index
index_name = "article_id"

# store result
colnames = ["UMAP" + str(i) for i in range(1, 3)]

df = pd.DataFrame(embedding, columns=colnames, index=embedding_index)

df = df.reset_index()
df.rename(columns={df.columns[0]: index_name}).to_feather(snek.output[0])
