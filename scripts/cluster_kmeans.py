"""
Computes topic/article clusters
"""
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

# load tf-idf, etc. matrix
dat = pd.read_feather(snakemake.input[0]).set_index('article_id')

# determine number of clusters
target = snakemake.wildcards['target']

num_clusters = snakemake.config['clustering'][target]['num_clusters']
batch_size = snakemake.config['clustering'][target]['batch_size']

print("Clustering data..")

kmeans = MiniBatchKMeans(num_clusters, batch_size=batch_size)

if target == 'articles':
    clusters = kmeans.fit(dat).labels_
else:
    clusters = kmeans.fit(dat.T).labels_

# for word frequency / tfidf matrices where columns correspond to specific words,
# generate human-readable cluster labels corresponding to the top 3 tokens associated
# with the cluster
if target == "articles" and "processing" in snakemake.wildcards:
    label_dict = []

    print("Generating cluster labels..")

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

if target == 'articles':
    res = pd.DataFrame({"article_id": dat.index, "cluster": cluster_labels})
else:
    res = pd.DataFrame({"topic": dat.columns, "cluster": cluster_labels})

res.to_feather(snakemake.output[0])
