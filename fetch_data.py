"""
Data acquisition script for the Brewery Type Classifier project.
Pulls data from Open Brewery DB (https://api.openbrewerydb.org/v1/breweries) - no API key required.
Run: python fetch_data.py
Produces: data/raw_data.csv
"""

import os
import time

import pandas as pd
import requests

BASE = "https://api.openbrewerydb.org/v1/breweries"
PER_PAGE = 200
PAGES = 3


def fetch_all_breweries(pages=PAGES, per_page=PER_PAGE):
    records = []
    for page in range(1, pages + 1):
        resp = requests.get(BASE, params={"page": page, "per_page": per_page})
        resp.raise_for_status()
        items = resp.json()
        if not items:
            break
        for b in items:
            name = b.get("name") or ""
            record = {
                "id": b.get("id"),
                "name": name,
                "brewery_type": b.get("brewery_type"),
                "city": b.get("city"),
                "state_province": b.get("state_province"),
                "country": b.get("country"),
                "latitude": b.get("latitude"),
                "longitude": b.get("longitude"),
                "has_website": 1 if b.get("website_url") else 0,
                "has_phone": 1 if b.get("phone") else 0,
                "name_length": len(name),
                "num_words_name": len(name.split()),
            }
            records.append(record)
        print(f"Fetched page {page} ({len(records)} total)")
        time.sleep(0.2)
    return records


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    records = fetch_all_breweries()
    df = pd.DataFrame(records)
    df.to_csv("data/raw_data.csv", index=False)
    print(f"Saved {len(df)} rows to data/raw_data.csv")
