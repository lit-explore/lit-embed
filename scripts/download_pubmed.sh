#!/bin/env bash
for i in $(seq -w 0001 1114); do
  echo $i
  wget "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed22n${i}.xml.gz"
  wget "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed22n${i}.xml.gz.md5"
done
