"""
Generates a scatterplot using datashader
"""
import numpy as np
import pandas as pd
import datashader as ds
import datashader.utils as utils
import datashader.transfer_functions as tf
import random
#import seaborn as sns

random.seed(snakemake.config['random_seed'])

# load data
target = snakemake.wildcards['target']

if target == 'articles':
    index = 'article_id'
else:
    index = 'topic'

# add cluster assignments
dat = pd.read_feather(snakemake.input[0]).set_index(index)
clusters = pd.read_feather(snakemake.input[1]).set_index(index)

# work-around: remove duplicated entries from "clusters" dataframe;
# will be handled upstream in the future
clusters = clusters[~clusters.index.duplicated(keep='first')]

dat["cluster"] = clusters.cluster.astype("category")

# subsample data
max_points = snakemake.config['plots']['datashader']['max_points']

if max_points < dat.shape[0]:
    ind = random.sample(range(dat.shape[0]), max_points)
    dat = dat.iloc[ind]

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

canvas = ds.Canvas(plot_width=1000, plot_height=1000)
agg = canvas.points(dat, dat.columns[0], dat.columns[1], ds.count_cat("cluster"))
img = tf.shade(agg, color_key=color_key, how="eq_hist")

utils.export_image(img, filename=snakemake.output[0].replace('.png', ''), background="black")
