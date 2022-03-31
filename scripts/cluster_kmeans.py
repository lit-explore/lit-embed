"""
Computes topic/article clusters
"""
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

# load tf-idf, etc. matrix
dat = pd.read_feather(snakemake.input[0]).set_index('article_id')

target = snakemake.wildcards['target']

# clustering settings
cfg = snakemake.config['clustering'][target]

num_clusters = cfg['num_clusters']
batch_size = cfg['batch_size']

# when clustering topics, limit number of articles used
if target == 'topics' and dat.shape[0] > cfg['max_articles']:
    print("Sub-sampling articles prior to topic clustering..")
    dat = dat.sample(cfg['max_articles'], random_state=snakemake.config['random_seed'])

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
    id_col = 'article_id'
    res = pd.DataFrame({"article_id": dat.index, "cluster": cluster_labels})

    # sanity check: make sure no duplicated article ids are found
    if res.index.duplicated().sum() > 0:
        raise Exception("Encountered duplicate article IDs!")

else:
    id_col = 'topic'
    res = pd.DataFrame({"topic": dat.columns, "cluster": cluster_labels})

res.to_feather(snakemake.output[0])
