"""
Model training script for the Brewery Type Classifier project.
Reads data/clean_data.csv, trains and compares two classifiers, and saves the
artifacts the Streamlit dashboard needs.
Run: python train_model.py
Produces: model.pkl, scaler.pkl, label_encoder.pkl, model_metrics.json
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

FEATURES = ["latitude", "longitude", "has_website", "has_phone", "name_length", "num_words_name"]
TARGET = "brewery_type"


def main():
    df_clean = pd.read_csv("data/clean_data.csv")

    X = df_clean[FEATURES]
    y = df_clean[TARGET]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42),
    }

    results = {}
    trained = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="macro", zero_division=0
        )
        results[name] = {
            "accuracy": acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1,
        }
        trained[name] = model
        print(name, results[name])

    best_name = max(results, key=lambda n: results[n]["f1_macro"])
    best_model = trained[best_name]
    best_preds = best_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, best_preds, labels=range(len(le.classes_)))

    if hasattr(best_model, "feature_importances_"):
        importances = dict(zip(FEATURES, best_model.feature_importances_.tolist()))
    else:
        importances = dict(zip(FEATURES, np.abs(best_model.coef_).mean(axis=0).tolist()))

    print("Best model:", best_name)

    joblib.dump(best_model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(le, "label_encoder.pkl")

    with open("model_metrics.json", "w") as f:
        json.dump(
            {
                "all_model_results": results,
                "best_model": best_name,
                "confusion_matrix": cm.tolist(),
                "class_labels": le.classes_.tolist(),
                "feature_importance": importances,
                "features": FEATURES,
            },
            f,
            indent=2,
        )

    print("Saved model.pkl, scaler.pkl, label_encoder.pkl, model_metrics.json")


if __name__ == "__main__":
    main()
