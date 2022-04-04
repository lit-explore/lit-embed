"""
Computes article clusters
"""
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

# load tf-idf, etc. matrix
dat = pd.read_feather(snakemake.input[0]).set_index('article_id')

# clustering settings
cfg = snakemake.config['clustering']['articles']

num_clusters = cfg['num_clusters']
batch_size = cfg['batch_size']

kmeans = MiniBatchKMeans(num_clusters, batch_size=batch_size)
clusters = kmeans.fit(dat).labels_

# for word frequency / tfidf matrices where columns correspond to specific words,
# generate human-readable cluster labels corresponding to the top 3 tokens associated
# with the cluster
if "processing" in snakemake.wildcards:
    label_dict = []

    for i in range(num_clusters):
        mask = clusters == i

        dat_subset = dat[mask]
        dat_subset.sum().sort_values()

        ranked_terms = dat_subset.sum().sort_values(ascending=False)

        label_dict.append("_".join(ranked_terms.index[:3]))

    mapping = {i:x for i, x in enumerate(label_dict)}
    cluster_labels = [mapping[x] for x in clusters]

else:
    # otherwise, just assign generic cluster names for now
    cluster_labels = [f"cluster_{x}" for x in clusters]

res = pd.DataFrame({"article_id": dat.index, "cluster": cluster_labels})

# sanity check: make sure no duplicated article ids are found
if res.index.duplicated().sum() > 0:
    raise Exception("Encountered duplicate article IDs!")

res.to_feather(snakemake.output[0])
