"""
Computes TF-IDF article clusters
"""
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

# load tf-idf matrix
dat = pd.read_feather(snakemake.input[0]).set_index('article_id')

# cluster articles
num_clusters = snakemake.config['clustering']['num_clusters']

kmeans = MiniBatchKMeans(num_clusters,
                         batch_size=snakemake.config['clustering']['batch_size'])

clusters = kmeans.fit(dat).labels_

# for each cluster, generate a cluster label by combining the top 3 tokens associated
# with the cluster
cluster_labels = []

for i in range(num_clusters):
    mask = clusters == i

    dat_subset = dat[mask]

    dat_subset.sum().sort_values()

    ranked_terms = dat_subset.sum().sort_values(ascending=False)

    cluster_labels.append("_".join(ranked_terms.index[:3]))

mapping = {i:x for i, x in enumerate(cluster_labels)}

named_clusters = [mapping[x] for x in clusters]

res = pd.DataFrame({"article_id": dat.index, "cluster": named_clusters})

res.to_feather(snakemake.output[0])
