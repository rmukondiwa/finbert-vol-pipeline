import pandas as pd
import requests
import os
#DONT RUN
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

headers = {
    "User-Agent": "Mozilla/5.0"
}

res = requests.get(url, headers=headers)
html = res.text

df = pd.read_html(html)[0]

# keep only relevant columns
df = df[["Symbol", "Security", "GICS Sector"]]

# optional: rename for cleanliness
df.columns = ["ticker", "name", "sector"]

# clean ticker format (important)
df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)

# ensure data folder exists
os.makedirs("data", exist_ok=True)

# save
df.to_csv("data/reference/sp500.csv", index=False)

print("Saved cleaned sp500.csv")