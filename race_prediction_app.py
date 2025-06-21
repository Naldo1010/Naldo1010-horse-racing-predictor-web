
import streamlit as st
import pandas as pd
import numpy as np

# Simulated scraping functions
def scrape_race_card(race_card_name, race_number):
    # Simulated race card data
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

# Composite score calculation
def calculate_composite_score(df):
    df = df.copy()
    df["Jockey Fee Tier Inverted"] = df["Jockey Fee Tier"].max() - df["Jockey Fee Tier"]
    df["Composite Score"] = (
        0.2 * df["Trainer Win %"] +
        0.25 * df["Estimated Win %"] +
        0.1 * df["Jockey Fee Tier Inverted"] +
        0.15 * (df["Horse Weight"] - df["Horse Weight"].min()) / (df["Horse Weight"].max() - df["Horse Weight"].min()) * 100 +
        0.3 * np.random.uniform(50, 100, len(df))  # Simulated base performance
    )
    return df

# Prediction model
def predict_win_probabilities(df):
    df = df.copy()
    total_score = df["Composite Score"].sum()
    df["Predicted Win Probability"] = df["Composite Score"] / total_score * 100
    return df.sort_values(by="Predicted Win Probability", ascending=False)

# Model refinement
def refine_model(predictions, actuals):
    merged = predictions.merge(actuals, on="Horse Name")
    merged["Top 3 Predicted"] = merged["Predicted Win Probability"].rank(ascending=False) <= 3
    merged["Top 3 Actual"] = merged["Actual Finish Position"] <= 3
    hit_rate = (merged["Top 3 Predicted"] & merged["Top 3 Actual"]).sum() / 3 * 100
    return merged, hit_rate

# Streamlit UI
st.title("🏇 Race Prediction App")

race_card_name = st.text_input("Enter Race Card Name (e.g., Churchill Downs 21 June 2025)")
race_number = st.number_input("Enter Race Number", min_value=1, step=1)

if race_card_name and race_number:
    st.subheader("📋 Scraping Race Card...")
    race_df = scrape_race_card(race_card_name, race_number)
    st.dataframe(race_df)

    st.subheader("🧮 Calculating Composite Scores...")
    scored_df = calculate_composite_score(race_df)
    st.dataframe(scored_df[["Horse Name", "Composite Score"]])

    st.subheader("🔮 Predicting Win Probabilities...")
    predictions = predict_win_probabilities(scored_df)
    st.dataframe(predictions[["Horse Name", "Predicted Win Probability"]])

    st.subheader("📊 Comparing with Actual Results...")
    actuals = scrape_actual_results(race_card_name, race_number)
    comparison_df, hit_rate = refine_model(predictions, actuals)
    st.dataframe(comparison_df[["Horse Name", "Predicted Win Probability", "Actual Finish Position"]])
    st.markdown(f"**Top-3 Hit Rate:** {hit_rate:.2f}%")

    st.subheader("🛠️ Model Refinement Suggestions")
    st.markdown("- Consider adjusting weights for Trainer Win % or Estimated Win % if predictions consistently miss.")
    st.markdown("- Add more features like track condition, recent form, or jockey-trainer synergy.")
