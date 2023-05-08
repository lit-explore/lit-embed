"""
Generates a scatterplot using datashader
"""
import pandas as pd
import datashader as ds
import datashader.utils as utils
import datashader.transfer_functions as tf
import random

snek = snakemake

random.seed(snek.config['random_seed'])

# load UMAP-projected article embeddings
dat = pd.read_feather(snek.input[0]).set_index("article_id")

# load cluster assignments
clusters = pd.read_feather(snek.input[1]).set_index('article_id')

# subsample, if needed
max_articles = snek.config['datashader']['max_articles']

if max_articles < dat.shape[0]:
    ind = random.sample(range(dat.shape[0]), max_articles)

    dat = dat.iloc[ind]
    clusters = clusters.iloc[ind]

# add cluster assignments to projected data
dat["cluster"] = clusters.cluster.astype("category")

cluster_labels = set(clusters.cluster.values)

# rainbow (n=10)
if len(cluster_labels) <= 10:
    pal = ["#fc8e61", "#ffadad", "#ffd6a5", "#caffbf", "#84d879", "#40e6b9", "#9bf6ff", "#a0c4ff", "#bdb2ff", "#ffc6ff"]
elif len(cluster_labels) <= 20:
    #pal = sns.color_palette("turbo", len(cluster_labels)).as_hex()
    # temp work-around (avoid using seaborn; triggers conda/matplotlib.pyplot segfault)
    pal = ['#3d358b', '#4456c7', '#4776ee', '#4294ff', '#2fb2f4', '#1bd0d5', '#1ae4b6',
           '#35f394', '#61fc6c', '#8fff49', '#b4f836', '#d2e935', '#ebd339', '#faba39',
           '#fe9b2d', '#f9751d', '#ec530f', '#da3907', '#c12302', '#a11201']
    pal = pal[:len(cluster_labels)]
else:
    raise Exception("Datashader visualization is limited to 20 clusters at the moment")

color_key = dict(zip(cluster_labels, pal))

canvas = ds.Canvas(plot_width=snek.config['datashader']['width'], 
                   plot_height=snek.config['datashader']['height'])
agg = canvas.points(dat, dat.columns[0], dat.columns[1], ds.count_cat("cluster"))
img = tf.shade(agg, color_key=color_key, how="eq_hist")

utils.export_image(img, filename=snek.output[0].replace('.png', ''), background="black")
