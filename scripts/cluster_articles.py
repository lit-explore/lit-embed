"""
Computes article clusters
"""
import math
import pandas as pd
import scipy
from sklearn.cluster import MiniBatchKMeans

snek = snakemake

# load embedding matrix
if snek.input[0].endswith(".npz"):
    mat = scipy.sparse.load_npz(snek.input[0])

    with open(snek.input[1], "rt", encoding="utf-8") as fp:
        article_ids = fp.read().split()
else:
    mat = pd.read_parquet(snek.input[0]).set_index("article_id")
    article_ids = mat.index

# clustering settings
num_clusters = snek.config['clustering']['num_clusters']
batch_size = snek.config['clustering']['batch_size']

kmeans = MiniBatchKMeans(num_clusters, batch_size=batch_size)
clusters = kmeans.fit(mat).labels_

# assign generic cluster names for now..
precision = math.ceil(math.log(num_clusters + 1, 10))
cluster_labels = [f"cluster_{x:0{precision}}" for x in clusters]

res = pd.DataFrame({"article_id": article_ids, "cluster": cluster_labels})

# sanity check: make sure no duplicated article ids are found
if res.index.duplicated().sum() > 0:
    raise Exception("Encountered duplicate article IDs!")

res.to_feather(snek.output[0])
