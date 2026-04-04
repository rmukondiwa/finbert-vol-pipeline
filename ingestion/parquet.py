import pandas as pd

df = pd.concat([
    pd.read_csv("data/reddit/out1.csv"),
    pd.read_csv("data/reddit/out2.csv")
])

df.to_parquet("data/reddit/reddit_full.parquet")