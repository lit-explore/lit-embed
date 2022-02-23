"""
Parses Pubmed XML data and extract article id, title, and abstract information
"""
import gzip
import xml.etree.ElementTree as ET
import pandas as pd

with gzip.open(snakemake.input[0], "r") as fp:
    tree = ET.parse(fp)

root = tree.getroot()

ids = []
dois = []
titles = []
abstracts = []

# iterate over articles in xml file
for article in root.findall(".//PubmedArticle"):
    # extract title
    title_elem = article.find(".//ArticleTitle")

    if title_elem is None or title_elem.text is None:
        title = ""
    else:
        title = title_elem.text.replace('\n', ' ').strip()

    if snakemake.config['exclude_articles']['missing_title'] and title == "":
        continue

    # extract abstract
    abstract_elem = article.find(".//AbstractText")

    if abstract_elem is None or abstract_elem.text is None:
        abstract = ""
    else:
        abstract = abstract_elem.text.replace('\n', ' ').strip()

    if snakemake.config['exclude_articles']['missing_abstract'] and abstract == "":
        continue

    # extract id/doi
    id = article.find(".//ArticleId[@IdType='pubmed']").text

    doi_elem = article.find(".//ArticleId[@IdType='doi']")
    doi = "" if doi_elem is None else doi_elem.text;


    ids.append(id)
    dois.append(doi)
    titles.append(title)
    abstracts.append(abstract)

dat = pd.DataFrame({"id": ids, "doi": dois, "title": titles, "abstract": abstracts})

dat.reset_index(drop=True).to_feather(snakemake.output[0])
