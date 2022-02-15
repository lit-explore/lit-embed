"""
choose a random subset of arxiv articles for testing
"""
import random

random.seed(snakemake.config['random_seed'])

# read all of the lines of the original json file in, and randomize the order, in order
# to store them in a random order
with open(snakemake.input[0], 'r') as fp:
    lines = fp.readlines()

random.shuffle(lines)

with open(snakemake.output[0], 'w') as fp:
    num_articles = snakemake.config['dev_mode']['num_articles']
    fp.write("".join(lines[:num_articles]))
