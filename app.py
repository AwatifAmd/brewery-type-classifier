"""
Brewery Type Classifier - Streamlit Dashboard
Data source: Open Brewery DB (https://api.openbrewerydb.org/v1/breweries) - no API key required.
"""

import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Brewery Type Classifier", layout="wide")

DATA_SOURCE_URL = "https://api.openbrewerydb.org/v1/breweries"
LAST_UPDATED = "2026-07-17"


@st.cache_data
def load_raw_data():
    return pd.read_csv("raw_data.csv")


@st.cache_data
def load_clean_data():
    return pd.read_csv("clean_data.csv")


@st.cache_data
def load_metrics():
    with open("model_metrics.json") as f:
        return json.load(f)


@st.cache_resource
def load_model_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder


df_raw = load_raw_data()
df_clean = load_clean_data()
metrics = load_metrics()
model, scaler, label_encoder = load_model_artifacts()

FEATURES = metrics["features"]
CLASS_LABELS = metrics["class_labels"]
BEST_MODEL_NAME = metrics["best_model"]


st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    ["Intro", "Data overview", "EDA", "Model performance", "Live prediction"],
)

st.sidebar.markdown("---")
type_filter = st.sidebar.multiselect(
    "Filter by brewery type (EDA section)",
    options=sorted(df_clean["brewery_type"].unique()),
    default=sorted(df_clean["brewery_type"].unique()),
)


if section == "Intro":
    st.title("Brewery Type Classifier")
    st.markdown(
        """
        This dashboard predicts a brewery's **type** (micro, brewpub, large, regional, ...)
        from its location and name-derived features, using data pulled programmatically
        from [Open Brewery DB](%s) - a free, key-free public API.
        """
        % DATA_SOURCE_URL
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows (clean data)", len(df_clean))
    col2.metric("Number of classes", len(CLASS_LABELS))
    col3.metric("Best model", BEST_MODEL_NAME.replace("_", " ").title())
    col4.metric(
        "Best macro-F1",
        f"{metrics['all_model_results'][BEST_MODEL_NAME]['f1_macro']:.3f}",
    )

elif section == "Data overview":
    st.title("Data overview")

    tab1, tab2 = st.tabs(["Raw data", "Clean data"])
    with tab1:
        st.write(f"Shape: {df_raw.shape[0]} rows x {df_raw.shape[1]} columns")
        st.dataframe(df_raw.head(20))
    with tab2:
        st.write(f"Shape: {df_clean.shape[0]} rows x {df_clean.shape[1]} columns")
        st.dataframe(df_clean.head(20))

    st.subheader("Missing values: raw vs. clean")
    missing_df = pd.DataFrame(
        {"raw_missing": df_raw.isna().sum(), "clean_missing": df_clean.isna().sum()}
    ).fillna(0)
    st.dataframe(missing_df)

    with st.expander("Cleaning decisions"):
        st.markdown(
            """
            - No missing target values (brewery_type is always present).
            - latitude/longitude were missing for ~20% of rows -> imputed with the median,
              since dropping the columns would lose useful geographic signal.
            - No exact duplicate rows found.
            - Rare classes ("bar", "taproom", "proprietor" - each under 5 examples) were
              merged into an "other" bucket so a stratified split is possible.
            - No outlier removal: unusual coordinates or long names are real breweries.
            """
        )

elif section == "EDA":
    st.title("Exploratory Data Analysis")

    filtered = df_clean[df_clean["brewery_type"].isin(type_filter)]

    st.subheader("Class balance")
    counts = filtered["brewery_type"].value_counts().reset_index()
    counts.columns = ["brewery_type", "count"]
    fig_bar = px.bar(counts, x="brewery_type", y="count", title="Brewery count by type")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Correlation heatmap")
    corr = filtered[FEATURES].corr()
    fig_heat = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Geographic spread")
    fig_scatter = px.scatter(
        filtered, x="longitude", y="latitude", color="brewery_type",
        hover_data=["name", "city", "country"], title="Brewery locations by type"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

elif section == "Model performance":
    st.title("Model performance")

    st.subheader("Model comparison")
    results_df = pd.DataFrame(metrics["all_model_results"]).T
    results_df.index.name = "model"
    st.dataframe(results_df.style.highlight_max(axis=0, color="lightgreen"))

    st.subheader(f"Confusion matrix ({BEST_MODEL_NAME})")
    cm = np.array(metrics["confusion_matrix"])
    fig_cm = px.imshow(
        cm, x=CLASS_LABELS, y=CLASS_LABELS,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        text_auto=True, aspect="auto",
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("Feature importance")
    fi = pd.Series(metrics["feature_importance"]).sort_values(ascending=True)
    fig_fi = px.bar(fi, orientation="h", title=f"Feature importance ({BEST_MODEL_NAME})")
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown(
        f"""
        **Why `{BEST_MODEL_NAME}`?** Selected on macro-averaged F1 rather than raw
        accuracy, since brewery types are heavily imbalanced (micro/brewpub dominate,
        while regional/contract/other are rare). Macro-F1 rewards a model that does
        reasonably well across all classes, not just the majority ones.
        """
    )

elif section == "Live prediction":
    st.title("Live prediction")
    st.write("Enter a brewery's characteristics to predict its type.")

    col1, col2 = st.columns(2)
    inputs = {}
    defaults = df_clean[FEATURES].median()

    with col1:
        inputs["latitude"] = st.number_input("Latitude", value=float(defaults["latitude"]))
        inputs["longitude"] = st.number_input("Longitude", value=float(defaults["longitude"]))
        inputs["has_website"] = 1 if st.checkbox("Has a website", value=True) else 0

    with col2:
        inputs["has_phone"] = 1 if st.checkbox("Has a phone number", value=True) else 0
        inputs["name_length"] = st.slider("Name length (characters)", 2, 60, int(defaults["name_length"]))
        inputs["num_words_name"] = st.slider("Number of words in name", 1, 10, int(defaults["num_words_name"]))

    if st.button("Predict"):
        X_new = pd.DataFrame([inputs])[FEATURES]
        X_scaled = scaler.transform(X_new)
        pred_encoded = model.predict(X_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]

        st.success(f"Predicted brewery type: **{pred_label}**")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_scaled)[0]
            proba_df = pd.DataFrame({"type": CLASS_LABELS, "probability": proba}).sort_values(
                "probability", ascending=False
            )
            fig_proba = px.bar(proba_df, x="type", y="probability", title="Prediction probabilities")
            st.plotly_chart(fig_proba, use_container_width=True)


st.sidebar.markdown("---")
st.sidebar.caption(f"Data source: Open Brewery DB ({DATA_SOURCE_URL})")
st.sidebar.caption(f"Last updated: {LAST_UPDATED}")
