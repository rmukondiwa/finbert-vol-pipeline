import pandas as pd

df = pd.concat([
    pd.read_csv("out1.csv"),
    pd.read_csv("out2.csv")
])

df.to_parquet("reddit_full.parquet")