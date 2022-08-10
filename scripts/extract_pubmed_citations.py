"""
Parses Pubmed XML data and constructs dataframe mapping from article ids to the ids of
articles cited.
"""
import gzip
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime
from typing import Any, List

with gzip.open(snakemake.input[0], "r") as fp:
    tree = ET.parse(fp)

root = tree.getroot()

# list to store main citation information
rows:List[dict[str,Any]] = []

# list to store citation stats
stat_rows:List[dict[str,Any]] = []

# iterate over articles in xml file
for article in root.findall(".//PubmedArticle"):
    # extract id/doi
    pmid_elem = article.find(".//ArticleId[@IdType='pubmed']")
    pmid = "" if pmid_elem is None else pmid_elem.text;

    doi_elem = article.find(".//ArticleId[@IdType='doi']")
    doi = "" if doi_elem is None else doi_elem.text

    # extract article date
    try:
        date_elem = article.find(".//PubDate")

        if date_elem is None:
            raise Exception("Unable to parse date")

        year = date_elem[0].text
        month = date_elem[1].text

        # if not day specified, default to the 1st
        if len(date_elem) == 3:
            day = date_elem[2].text
        else:
            day = "01"

        # check for numeric/string months
        date_format = "%Y %m %d" if month.isnumeric() else "%Y %b %d"

        date_str = datetime.strptime(f"{year} {month} {day}", date_format).isoformat()
    except:
        # if date parsing fails, just leave field blank
        date_str = ""

    # extract citations
    ref_elems = article.findall(".//ReferenceList/Reference")

    num_missing = 0

    for ref_elem in ref_elems:
        # note: if more than one pmid/doi is associated with a reference, only the first
        # one will be used
        ref_pmid_elem = ref_elem.find(".//ArticleId[@IdType='pubmed']")
        ref_pmid = "" if ref_pmid_elem is None else ref_pmid_elem.text;

        ref_doi_elem = ref_elem.find(".//ArticleId[@IdType='doi']")
        ref_doi = "" if ref_doi_elem is None else ref_doi_elem.text

        if ref_pmid == "" and ref_doi == "":
            num_missing += 1
            continue

        rows.append({
            "source_pmid": pmid,
            "source_doi": doi,
            "ref_pmid": ref_pmid,
            "ref_doi": ref_doi

        })

    # update stats
    stat_rows.append({
        "pmid": pmid,
        "doi": doi,
        "date": date_str,
        "num_citations": len(ref_elems),
        "num_missing_id": num_missing
    })

# store main dataframe
dat = pd.DataFrame(rows)

if dat.shape[0] == 0:
    raise Exception("No articles found with all required components!")

dat.reset_index(drop=True).to_feather(snakemake.output[0])

# store stats dataframe
stats_df = pd.DataFrame(stat_rows)
stats_df.to_feather(snakemake.output[1])
#  stats_df.reset_index(drop=True).to_feather(snakemake.output[1])
