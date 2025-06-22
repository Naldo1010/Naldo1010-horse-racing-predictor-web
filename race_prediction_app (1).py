
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

# Simulated feature weights for composite score
FEATURE_WEIGHTS = {
    "Trainer Win %": 0.2,
    "Estimated Win %": 0.25,
    "Jockey Fee Tier": -0.1,  # lower tier is better
    "Horse Weight": 0.15,
    "Base Performance": 0.3
}

def scrape_race_card(race_card_name, race_number):
    # Simulated scraping logic (replace with real scraping logic)
    data = {
        "Horse Number": [1, 2, 3, 4, 5],
        "Horse Name": ["Thunder Bolt", "Lightning Flash", "Wind Runner", "Storm Chaser", "Rain Maker"],
        "Draw": [1, 2, 3, 4, 5],
        "Trainer Win %": [15, 20, 10, 5, 12],
        "Estimated Win %": [18, 25, 12, 6, 10],
        "Jockey Fee Tier": [1, 1, 2, 3, 2],
        "Horse Weight": [1150, 1170, 1130, 1120, 1145]
    }
    return pd.DataFrame(data)

def scrape_actual_results(race_card_name, race_number):
    # Simulated actual results
    results = {
        "Horse Name": ["Thunder Bolt", "Lightning Flash", "Wind Runner", "Storm Chaser", "Rain Maker"],
        "Actual Finish Position": [2, 1, 3, 5, 4]
    }
    return pd.DataFrame(results)

def calculate_composite_scores(df):
    df = df.copy()
    df["Base Performance"] = (df["Trainer Win %"] + df["Estimated Win %"]) / 2
    df["Composite Score"] = (
        df["Trainer Win %"] * FEATURE_WEIGHTS["Trainer Win %"] +
        df["Estimated Win %"] * FEATURE_WEIGHTS["Estimated Win %"] +
        df["Jockey Fee Tier"] * FEATURE_WEIGHTS["Jockey Fee Tier"] +
        df["Horse Weight"] * FEATURE_WEIGHTS["Horse Weight"] +
        df["Base Performance"] * FEATURE_WEIGHTS["Base Performance"]
    )
    return df

def predict_win_probabilities(df):
    df = df.copy()
    df["Win Probability"] = df["Composite Score"] / df["Composite Score"].sum()
    return df.sort_values(by="Win Probability", ascending=False)

def compare_predictions_with_actuals(pred_df, actual_df):
    merged = pd.merge(pred_df, actual_df, on="Horse Name", how="left")
    merged["Top 3 Predicted"] = merged["Win Probability"].rank(ascending=False) <= 3
    merged["Top 3 Actual"] = merged["Actual Finish Position"] <= 3
    hit_rate = (merged["Top 3 Predicted"] & merged["Top 3 Actual"]).sum() / 3
    return merged, hit_rate

# Streamlit UI
st.title("🏇 Race Prediction App")

race_card_name = st.text_input("Enter Race Card Name (e.g., Churchill Downs 21 June 2025)")
race_number = st.number_input("Enter Race Number", min_value=1, step=1)

if race_card_name and race_number:
    st.subheader("📥 Scraping Race Card...")
    race_df = scrape_race_card(race_card_name, race_number)
    st.dataframe(race_df)

    st.subheader("📊 Calculating Composite Scores...")
    scored_df = calculate_composite_scores(race_df)
    st.dataframe(scored_df[["Horse Name", "Composite Score"]])

    st.subheader("🔮 Predicting Win Probabilities...")
    predictions = predict_win_probabilities(scored_df)
    st.dataframe(predictions[["Horse Name", "Win Probability"]])

    st.subheader("📈 Scraping Actual Results...")
    actual_df = scrape_actual_results(race_card_name, race_number)
    st.dataframe(actual_df)

    st.subheader("📉 Comparing Predictions with Actual Results...")
    comparison_df, hit_rate = compare_predictions_with_actuals(predictions, actual_df)
    st.dataframe(comparison_df[["Horse Name", "Win Probability", "Actual Finish Position", "Top 3 Predicted", "Top 3 Actual"]])
    st.metric("Top-3 Hit Rate", f"{hit_rate:.2%}")
