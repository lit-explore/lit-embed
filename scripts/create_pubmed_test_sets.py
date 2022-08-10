"""
Create PubMed co-citation test data

Generates co-citation matrices for three randomly-selected sets of PubMed articles.

Ideally, we would compare the article co-citation matrices and embedding similarity
matrices for _all_ articles, but, that's a whole lot of memory...

Instead, we were just compare a few sets of randomly selected edges from each.
"""
import numpy as np
import pandas as pd

# load data
cite_df = pd.read_feather(snakemake.input[0])
stats_df = pd.read_feather(snakemake.input[1])
article_df = pd.read_csv(snakemake.input[2], dtype={'id': str}) 

CFG = snakemake.config['validation']

CFG1 = CFG['test_sets']['set1']
CFG2 = CFG['test_sets']['set2']
CFG3 = CFG['test_sets']['set3']

MIN_CITATIONS = CFG['min_citations']

# exclude articles with too few citations
valid_pmids = stats_df.pmid[stats_df.times_cited >= MIN_CITATIONS]

# TODO: move check upstream?.. (should not be in stats df to begin with..)
#valid_pmids = [x for x in valid_pmids.values if x != '']

cite_df = cite_df[cite_df.ref_pmid.isin(valid_pmids)]
stats_df = stats_df[stats_df.pmid.isin(valid_pmids)]
article_df = article_df[article_df.id.isin(valid_pmids)]

def generate_test_set(topic, n, seed):
    """
    Function to generate a single test set

    topic: key phrase to restrict articles to
    n: sample size (number of articles)
    seed: random seed to use
    """
    # find all articles mentioning the topic key phrase in its title/abstract
    topic_pmids = article_df.id[article_df.title.str.lower().str.contains(topic) |
                                article_df.abstract.str.lower().str.contains(topic)]
    topic_stats = stats_df[stats_df.pmid.isin(topic_pmids)]

    # sample pubmed ids
    rng = np.random.default_rng(seed)
    pmids = rng.choice(topic_stats.pmid, min(topic_stats.shape[0], n))

    # get all articles citing at least one of the selected articles, limited to 
    # entries pertaining to one of the sampled pubmed ids
    citing_pmids = set(cite_df[cite_df.ref_pmid.isin(pmids)].source_pmid.values)

    df = cite_df[cite_df.source_pmid.isin(citing_pmids) & cite_df.ref_pmid.isin(pmids)]

    # convert to binary co-occurrence matrix
    df['presence'] = 1
    count_mat = df.pivot_table(index='source_pmid', columns='ref_pmid', values='presence', fill_value=0)
    comat = count_mat.T.dot(count_mat).reset_index()

    return comat

df1 = generate_test_set(CFG1['topic'], CFG1['n'], CFG1['seed'])
df2 = generate_test_set(CFG2['topic'], CFG2['n'], CFG2['seed'])
df3 = generate_test_set(CFG3['topic'], CFG3['n'], CFG3['seed'])

# save results
#  df1.reset_index(drop=True).to_feather(snakemake.output[0])
#  df2.reset_index(drop=True).to_feather(snakemake.output[1])
#  df3.reset_index(drop=True).to_feather(snakemake.output[2])
df1.to_feather(snakemake.output[0])
df2.to_feather(snakemake.output[1])
df3.to_feather(snakemake.output[2])
