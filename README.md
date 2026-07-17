# Brewery Type Classifier

Predicts a brewery's type (micro, brewpub, large, regional, ...) from its location
and name-derived features, using data pulled programmatically from
[Open Brewery DB](https://api.openbrewerydb.org/v1/breweries) - a free, key-free
public API.

## Project structure

```
project/
├── data/
│   ├── raw_data.csv
│   └── clean_data.csv
├── notebooks/
│   └── Untitled1.ipynb          (full Colab notebook: fetch -> EDA -> training)
├── fetch_data.py                 (standalone data-fetch script)
├── train_model.py                (standalone model-training script)
├── app.py                        (Streamlit dashboard)
├── model.pkl
├── scaler.pkl
├── label_encoder.pkl
├── model_metrics.json
├── requirements.txt
├── NOTES.md                      (cleaning decisions + write-up)
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
python fetch_data.py     # only needed to regenerate data/raw_data.csv
python train_model.py    # only needed to regenerate model.pkl and friends
streamlit run app.py
```

## Data source

Open Brewery DB (`https://api.openbrewerydb.org/v1/breweries`), no authentication
required. 600 breweries fetched across 3 paginated calls (page/per_page params).

## Models

Logistic Regression and Random Forest are trained and compared on macro-averaged
precision/recall/F1 (fairer than raw accuracy given class imbalance). See
`model_metrics.json` and `NOTES.md` for full results and the deployment choice.

## Deployment

See `NOTES.md` for the live dashboard URL once deployed.
