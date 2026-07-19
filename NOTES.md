# Write-up

## Data source and why I chose it

Open Brewery DB (`https://api.openbrewerydb.org/v1/breweries`) - free, no API key
or registration needed. 600 breweries were fetched across 3 paginated calls
(`page`/`per_page` params, 200 per page), a good example of handling classic
offset-style pagination.

## Key cleaning decisions and why

- **Missing target values**: none - every brewery record has a `brewery_type`.
- **Missing feature values**: `latitude`/`longitude` were missing for about 20% of
  rows (breweries without geocoding). Since this is under the 40-50% drop
  threshold and coordinates are a useful signal, missing values were imputed
  with the column median rather than dropping the column or the rows.
- **Duplicates**: checked with `df.duplicated().sum()`, none found.
- **Rare classes**: "bar" (1 example), "taproom" (3), and "proprietor" (4) each
  had too few examples for a stratified 80/20 split, so they were merged into
  an "other" bucket. "contract" (12) and "regional" (11) were kept separate
  since they clear the >=5 threshold.
- **Outliers**: not removed. Unusual coordinates or long/short brewery names
  are legitimate real businesses, not data errors.
- **Categorical encoding**: not needed beyond the binary `has_website`/`has_phone`
  flags, which were already engineered as 0/1 numeric features from the raw
  `website_url`/`phone` fields.

## Model comparison results

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| Logistic Regression | 0.525 | 0.135 | 0.151 | 0.138 |
| Random Forest | 0.567 | 0.172 | 0.187 | 0.175 |

(See `model_metrics.json` for the full confusion matrix across all 8 classes:
brewpub, closed, contract, large, micro, other, planning, regional.)

**Model deployed: Random Forest**, selected on macro-F1 since brewery types are
heavily imbalanced (micro and brewpub dominate the dataset). Random Forest wins
on every metric here, including raw accuracy, and its `feature_importances_`
show that `longitude` (0.32), `latitude` (0.27), and `name_length` (0.23) carry
most of the predictive signal - geography and how brewery names are written turn
out to matter more than whether a website or phone number is listed.

## Honest limitation / next step

Accuracy (~57%) looks better than the Pokemon project mainly because two classes
("micro" and "brewpub") make up the large majority of the data, so a model that
leans toward predicting those does reasonably well on raw accuracy while still
struggling on the macro-F1 for rarer classes like "contract" or "regional" (visible
in the confusion matrix, where most rare-class rows get misclassified as "micro").
A stronger next step would be to add real content features - for example scraping
brewery descriptions or using state/country as a categorical feature - rather than
relying only on coordinates and name length, which are weak proxies for what
actually distinguishes a "large" brewery from a "micro" one.

## Live deployed dashboard

URL: https://brewery-type-classifier.streamlit.app/
GitHub repo: https://github.com/AwatifAmd/brewery-type-classifier
