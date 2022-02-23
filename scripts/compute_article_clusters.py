"""
Computes article clusters
"""
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

# load tf-idf, etc. matrix
dat = pd.read_feather(snakemake.input[0]).set_index('article_id')

# cluster articles
num_clusters = snakemake.config['clustering']['num_clusters']

print("Clustering data..")

kmeans = MiniBatchKMeans(num_clusters,
                         batch_size=snakemake.config['clustering']['batch_size'])
clusters = kmeans.fit(dat).labels_

# for word frequency / tfidf matrices where columns correspond to specific words,
# generate human-readable cluster labels corresponding to the top 3 tokens associated
# with the cluster
if "processing" in snakemake.wildcards:
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
    # for biobert, etc. embeddings, just use generic "cluster_xx" labels for now
    cluster_labels = [f"cluster_{x}" for x in clusters]

print("Saving clustering results..")

res = pd.DataFrame({"article_id": dat.index, "cluster": cluster_labels})

res.to_feather(snakemake.output[0])
