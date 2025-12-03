# (Copy-paste the full Streamlit code from the last cell of Task-2.ipynb here)
# Streamlit App: ClimateScope Interactive Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

st.set_page_config(page_title="ClimateScope Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_weather_data.csv", parse_dates=["date", "last_updated"])

    # Ensure helper columns exist
    if "month" not in df.columns:
        df["month"] = df["date"].dt.month

    if "season" not in df.columns:
        def get_season(m):
            if m in [3, 4, 5]:
                return "Spring"
            elif m in [6, 7, 8]:
                return "Summer"
            elif m in [9, 10, 11]:
                return "Fall"
            else:
                return "Winter"
        df["season"] = df["month"].apply(get_season)

    if "latitude_zone" not in df.columns:
        def get_latitude_zone(lat):
            if lat >= 66.5:
                return "Arctic"
            elif lat >= 23.5:
                return "Northern Temperate"
            elif lat >= -23.5:
                return "Tropical"
            elif lat >= -66.5:
                return "Southern Temperate"
            else:
                return "Antarctic"
        df["latitude_zone"] = df["latitude"].apply(get_latitude_zone)

    return df


df_streamlit = load_data()

st.title("🌍 ClimateScope - Global Weather Dashboard")
st.markdown("Use the filters in the sidebar to explore the dataset interactively.")

# ------------------------- SIDEBAR FILTERS -------------------------
st.sidebar.header("Filters")

countries = sorted(df_streamlit["country"].dropna().unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Country", options=countries, default=countries
)

min_date = df_streamlit["date"].min().date()
max_date = df_streamlit["date"].max().date()
selected_date_range = st.sidebar.date_input(
    "Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)

seasons = sorted(df_streamlit["season"].dropna().unique().tolist())
selected_seasons = st.sidebar.multiselect(
    "Season", options=seasons, default=seasons
)

lat_zones = sorted(df_streamlit["latitude_zone"].dropna().unique().tolist())
selected_zones = st.sidebar.multiselect(
    "Latitude Zones", options=lat_zones, default=lat_zones
)

variables = [
    "temperature_celsius",
    "humidity",
    "pressure_mb",
    "wind_kph",
    "precip_mm",
    "cloud",
    "uv_index",
]
selected_variable = st.sidebar.selectbox("Weather Variable", options=variables, index=0)

extreme_only = st.sidebar.checkbox("Show only extreme events", value=False)

# ------------------------- APPLY FILTERS -------------------------
filtered = df_streamlit.copy()

if selected_countries:
    filtered = filtered[filtered["country"].isin(selected_countries)]

start_date, end_date = selected_date_range
filtered = filtered[(filtered["date"].dt.date >= start_date) & (filtered["date"].dt.date <= end_date)]

if selected_seasons:
    filtered = filtered[filtered["season"].isin(selected_seasons)]

if selected_zones:
    filtered = filtered[filtered["latitude_zone"].isin(selected_zones)]

if extreme_only and not filtered.empty:
    # Define extreme thresholds using percentiles (like notebook analysis)
    extreme_thresholds = {
        "temperature_celsius": {
            "hot": filtered["temperature_celsius"].quantile(0.95),
            "cold": filtered["temperature_celsius"].quantile(0.05),
        },
        "wind_kph": {
            "high": filtered["wind_kph"].quantile(0.95),
        },
        "precip_mm": {
            "heavy": filtered.loc[filtered["precip_mm"] > 0, "precip_mm"].quantile(0.95)
            if (filtered["precip_mm"] > 0).any() else 0,
        },
        "humidity": {
            "high": filtered["humidity"].quantile(0.95),
            "low": filtered["humidity"].quantile(0.05),
        },
    }

    is_extreme = (
        (filtered["temperature_celsius"] >= extreme_thresholds["temperature_celsius"]["hot"]) |
        (filtered["temperature_celsius"] <= extreme_thresholds["temperature_celsius"]["cold"]) |
        (filtered["wind_kph"] >= extreme_thresholds["wind_kph"]["high"]) |
        (filtered["precip_mm"] >= extreme_thresholds["precip_mm"]["heavy"]) |
        (filtered["humidity"] >= extreme_thresholds["humidity"]["high"]) |
        (filtered["humidity"] <= extreme_thresholds["humidity"]["low"])
    )
    filtered = filtered[is_extreme]

st.markdown(f"**Filtered records:** {len(filtered):,} (out of {len(df_streamlit):,})")

if filtered.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# ------------------------- KPI CARDS -------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Avg Temperature (°C)", f"{filtered['temperature_celsius'].mean():.2f}")
with col2:
    st.metric("Total Precipitation (mm)", f"{filtered['precip_mm'].sum():.2f}")
with col3:
    st.metric("Avg Wind (kph)", f"{filtered['wind_kph'].mean():.2f}")
with col4:
    st.metric("Locations", f"{filtered['location_name'].nunique():,}")

st.markdown("---")

# ------------------------- TIME SERIES CHARTS -------------------------

st.subheader("Time Series Trends")

# Aggregate daily
daily = (
    filtered.groupby("date").agg(
        temperature_celsius=("temperature_celsius", "mean"),
        precip_mm=("precip_mm", "sum"),
    ).reset_index()
)

col_ts1, col_ts2 = st.columns(2)

with col_ts1:
    st.markdown("**Temperature Over Time**")
    st.line_chart(daily.set_index("date")["temperature_celsius"])

with col_ts2:
    st.markdown("**Precipitation Over Time**")
    st.line_chart(daily.set_index("date")["precip_mm"])

# ------------------------- CORRELATION HEATMAP -------------------------

st.subheader("Correlation Matrix (Selected Variables)")

corr_vars = [
    "temperature_celsius",
    "humidity",
    "pressure_mb",
    "wind_kph",
    "precip_mm",
    "cloud",
    "uv_index",
]

corr = filtered[corr_vars].corr()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax)
plt.tight_layout()
st.pyplot(fig)

# ------------------------- REGIONAL BAR CHART -------------------------

st.subheader("Top Countries by Selected Variable")

country_metric = (
    filtered.groupby("country")[selected_variable]
    .mean()
    .sort_values(ascending=False)
    .head(15)
)

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.barh(country_metric.index, country_metric.values, color="tab:orange")
ax2.set_xlabel(selected_variable.replace("_", " ").title())
ax2.set_ylabel("Country")
ax2.invert_yaxis()
plt.tight_layout()
st.pyplot(fig2)

# ------------------------- EXTREME EVENTS TABLE -------------------------

st.subheader("Extreme Events (Sample)")

# Simple definition: top 200 rows with highest absolute z-score for temperature
if len(filtered) > 0:
    tmp = filtered.copy()
    tmp["temp_zscore"] = (tmp["temperature_celsius"] - tmp["temperature_celsius"].mean()) / tmp["temperature_celsius"].std()
    extreme_sample = (
        tmp.reindex(tmp["temp_zscore"].abs().sort_values(ascending=False).index)
        [["date", "country", "location_name", "temperature_celsius", "humidity", "wind_kph", "precip_mm"]]
        .head(200)
    )
    st.dataframe(extreme_sample)
else:
    st.info("No extreme events to display for the current filters.")

st.markdown("---")
st.caption("Run this app with: `streamlit run Task-2.ipynb` is not supported; instead, save the logic to a `.py` file if needed.")
