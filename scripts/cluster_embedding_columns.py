"""
Computes embedding column clusters
"""
import math
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

# load tf-idf, etc. matrix
dat = pd.read_feather(snakemake.input[0]).set_index('article_id')

# clustering settings
cfg = snakemake.config['clustering']['embedding_columns']

num_clusters = cfg['num_clusters']

# subsample articles to speed things up and reduce memory requirements
if dat.shape[0] > cfg['max_articles']:
    print("Sub-sampling articles prior to embedding column clustering..")
    dat = dat.sample(cfg['max_articles'], random_state=snakemake.config['random_seed'])

clustering = AgglomerativeClustering(num_clusters, 
        affinity=cfg['affinity'],
        linkage=cfg['linkage'])

clusters = clustering.fit(dat.T).labels_

# note: modify if > 99 clusters are requested..
precision = math.ceil(math.log(num_clusters + 1, 10))

cluster_labels = [f"cluster_{x:0{precision}}" for x in clusters]

res = pd.DataFrame({"embedding_column": dat.columns, "cluster": cluster_labels})

res.to_feather(snakemake.output[0])
