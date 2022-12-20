"""
Combines batches of word stats into a single dataframe
"""
import pandas as pd
import pyarrow as pa
from pyarrow.parquet import ParquetWriter

# pass 1: determine all possible category levels
tokens = set()

for infile in snakemake.input:
    df = pd.read_parquet(infile)
    tokens = tokens.union(set(df.token.values))

token_dtype = pd.CategoricalDtype(categories=sorted(list(tokens)))

# pass 2: read article batches in and append to output parquet file
df = pd.read_parquet(snakemake.input[0])
df.token = df.token.astype(token_dtype)
tbl = pa.Table.from_pandas(df)

pqwriter = ParquetWriter(snakemake.output[0], tbl.schema)
pqwriter.write_table(tbl)

for infile in snakemake.input[1:]:
    df = pd.read_parquet(infile)
    df.token = df.token.astype(token_dtype)
    pqwriter.write_table(pa.Table.from_pandas(df))

pqwriter.close()
