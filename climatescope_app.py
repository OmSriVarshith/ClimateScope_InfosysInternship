import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ------------------------- CONFIG & SETUP -------------------------
st.set_page_config(
    page_title="ClimateScope Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------- THEME & COLOR SCHEME -------------------------
# Consistent color palette for all visualizations
COLOR_SCHEME = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'accent': '#2ca02c',
    'warning': '#d62728',
    'info': '#9467bd',
    'success': '#8c564b',
    'season_winter': '#4A90E2',
    'season_spring': '#50C878',
    'season_summer': '#FF6B35',
    'season_fall': '#D2691E',
    'gradient': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
}

SEASON_COLORS = {
    'Winter': COLOR_SCHEME['season_winter'],
    'Spring': COLOR_SCHEME['season_spring'],
    'Summer': COLOR_SCHEME['season_summer'],
    'Fall': COLOR_SCHEME['season_fall']
}

# Plotly color scales for consistency
PLOTLY_COLORSCALE = "Viridis"
PLOTLY_CORR_COLORSCALE = "RdBu_r"

# Chart template
CHART_TEMPLATE = {
    'layout': go.Layout(
        font=dict(family="Arial, sans-serif", size=12, color="#333"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        hovermode='closest',
        legend=dict(bgcolor='rgba(255,255,255,0.8)', bordercolor='rgba(0,0,0,0.2)', borderwidth=1)
    )
}

# ------------------------- DATA LOADING & PROCESSING -------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("cleaned_weather_data.csv", parse_dates=["date", "last_updated"])
    except FileNotFoundError:
        st.error("Data file 'cleaned_weather_data.csv' not found. Please ensure the file is in the directory.")
        return pd.DataFrame()

    # --- Data Cleaning & Feature Engineering ---
    cols_to_numeric = ['temperature_celsius', 'humidity', 'precip_mm', 'wind_kph', 'pressure_mb', 'uv_index']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df["month_name"] = df["date"].dt.month_name()
    df["month_num"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    
    def get_season(m):
        if m in [12, 1, 2]: return "Winter"
        elif m in [3, 4, 5]: return "Spring"
        elif m in [6, 7, 8]: return "Summer"
        else: return "Fall"
    df["season"] = df["month_num"].apply(get_season)

    # Heat Index & Wind Chill (Approx)
    df["heat_index_c"] = df["temperature_celsius"] + (df["humidity"] / 100) * (df["temperature_celsius"] - 15)
    T = df["temperature_celsius"]
    V = df["wind_kph"]
    df["wind_chill_c"] = np.where(
        (T <= 10) & (V >= 4.8),
        13.12 + 0.6215*T - 11.37*(V**0.16) + 0.3965*T*(V**0.16),
        T
    )

    return df

df_raw = load_data()

if df_raw.empty:
    st.stop()

# ------------------------- SIDEBAR NAVIGATION & FILTERS -------------------------

st.sidebar.title("🌍 ClimateScope")

# Navigation
nav_selection = st.sidebar.radio(
    "Navigation", 
    ["Executive Dashboard", "Statistical Analysis", "Climate Trends", "Extreme Events", "Help & User Guide"]
)

st.sidebar.markdown("---")
st.sidebar.header("Global Filters")

# --- 1. COUNTRY FILTER (CLEAN UI) ---
all_countries = sorted(df_raw["country"].dropna().unique())

# "Select All" Toggle to avoid the messy tag cloud
select_all_countries = st.sidebar.checkbox("Select All Countries", value=True)

if select_all_countries:
    selected_countries = all_countries
else:
    # If unchecked, allow specific selection (starts empty for cleanliness)
    selected_countries = st.sidebar.multiselect(
        "Select Specific Countries", 
        options=all_countries, 
        default=[] 
    )
    if not selected_countries:
        st.sidebar.warning("⚠️ Please select at least one country.")

# --- 2. DATE RANGE FILTER ---
min_date = df_raw["date"].min().date()
max_date = df_raw["date"].max().date()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

# --- 3. LOCATION FILTER (NEW) ---
st.sidebar.markdown("### Location Filter")
all_locations = sorted(df_raw["location_name"].dropna().unique())
select_all_locations = st.sidebar.checkbox("Select All Locations", value=True, key="loc_all")

if select_all_locations:
    selected_locations = all_locations
else:
    selected_locations = st.sidebar.multiselect(
        "Select Specific Locations", 
        options=all_locations, 
        default=[],
        key="loc_select"
    )

# --- 4. METRIC SELECTOR ---
st.sidebar.markdown("### Metric Selection")
metric_options = {
    "Temperature (°C)": "temperature_celsius",
    "Precipitation (mm)": "precip_mm",
    "Wind Speed (kph)": "wind_kph",
    "Humidity (%)": "humidity",
    "Pressure (mb)": "pressure_mb",
    "UV Index": "uv_index",
    "Heat Index (°C)": "heat_index_c",
    "Wind Chill (°C)": "wind_chill_c"
}
selected_metric_label = st.sidebar.selectbox("Primary Metric", list(metric_options.keys()))
selected_metric = metric_options[selected_metric_label]

# --- 5. ADVANCED FILTERS (NEW) ---
st.sidebar.markdown("### Advanced Controls")
with st.sidebar.expander("📊 Visualization Settings", expanded=False):
    moving_avg_window = st.slider("Moving Average Window (days)", min_value=1, max_value=30, value=7, step=1)
    show_trend_line = st.checkbox("Show Trend Line", value=True)
    show_confidence_band = st.checkbox("Show Confidence Band", value=False)
    
with st.sidebar.expander("📈 Threshold Settings", expanded=False):
    percentile_threshold = st.slider("Percentile Threshold for Anomalies", min_value=90, max_value=99, value=95, step=1)
    enable_anomaly_detection = st.checkbox("Highlight Anomalies", value=False)

# ------------------------- CRITICAL: FILTERING LOGIC -------------------------
# This block was missing in the previous snippet, causing the NameError

# 1. Unpack the date range safely
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

# 2. Filter the DataFrame
df_filtered = df_raw[
    (df_raw["country"].isin(selected_countries)) & 
    (df_raw["location_name"].isin(selected_locations)) &
    (df_raw["date"].dt.date >= start_date) & 
    (df_raw["date"].dt.date <= end_date)
].copy()

# 3. Add Moving Averages (Dynamic) with configurable window
if not df_filtered.empty:
    df_filtered[f"{moving_avg_window}d_moving_avg"] = df_filtered.groupby("location_name")[selected_metric].transform(
        lambda x: x.rolling(moving_avg_window, min_periods=1).mean()
    )
    
    # Add percentile-based anomaly detection
    if enable_anomaly_detection:
        percentile_value = df_filtered[selected_metric].quantile(percentile_threshold / 100)
        df_filtered["is_anomaly_high"] = df_filtered[selected_metric] >= percentile_value
        df_filtered["is_anomaly_low"] = df_filtered[selected_metric] <= df_filtered[selected_metric].quantile((100 - percentile_threshold) / 100)

if df_filtered.empty:
    st.warning("No data matches your filters. Please adjust the sidebar.")
    st.stop()

# ------------------------- PAGE 1: EXECUTIVE DASHBOARD -------------------------
if nav_selection == "Executive Dashboard":
    st.title("📊 Executive Dashboard")
    st.markdown(f"**Snapshot:** {start_date} to {end_date} | **Focus:** {selected_metric_label} | **Locations:** {len(selected_locations)}")

    # --- KPI Cards with Delta Indicators ---
    avg_val = df_filtered[selected_metric].mean()
    total_precip = df_filtered["precip_mm"].sum()
    max_wind = df_filtered["wind_kph"].max()
    locations_count = df_filtered["location_name"].nunique()
    
    # Calculate deltas for comparison (first half vs second half of period)
    mid_date = start_date + (end_date - start_date) / 2
    first_half = df_filtered[df_filtered["date"].dt.date < mid_date][selected_metric].mean()
    second_half = df_filtered[df_filtered["date"].dt.date >= mid_date][selected_metric].mean()
    delta_val = second_half - first_half if not (pd.isna(first_half) or pd.isna(second_half)) else None

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"Avg {selected_metric_label}", f"{avg_val:.1f}", delta=f"{delta_val:.1f}" if delta_val else None)
    c2.metric("Total Precip (mm)", f"{total_precip:,.0f}")
    c3.metric("Max Wind (kph)", f"{max_wind:.1f}")
    c4.metric("Active Locations", locations_count)
    c5.metric("Data Points", f"{len(df_filtered):,}")

    st.markdown("---")

    # --- Automated Insights Section (NEW) ---
    st.subheader("🔍 Key Insights & Findings")
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        st.markdown("**📊 Statistical Highlights**")
        median_val = df_filtered[selected_metric].median()
        std_val = df_filtered[selected_metric].std()
        st.info(f"**Median:** {median_val:.1f} | **Std Dev:** {std_val:.1f}")
        
        # Variability insight
        cv = (std_val / avg_val * 100) if avg_val != 0 else 0
        if cv > 30:
            st.warning(f"⚠️ High variability detected (CV: {cv:.1f}%)")
        else:
            st.success(f"✓ Moderate variability (CV: {cv:.1f}%)")
    
    with insight_col2:
        st.markdown("**🌍 Geographic Insights**")
        top_country = df_filtered.groupby("country")[selected_metric].mean().idxmax()
        top_country_val = df_filtered.groupby("country")[selected_metric].mean().max()
        st.info(f"**Highest:** {top_country} ({top_country_val:.1f})")
        
        bottom_country = df_filtered.groupby("country")[selected_metric].mean().idxmin()
        bottom_country_val = df_filtered.groupby("country")[selected_metric].mean().min()
        st.info(f"**Lowest:** {bottom_country} ({bottom_country_val:.1f})")
    
    with insight_col3:
        st.markdown("**📅 Temporal Insights**")
        peak_date = df_filtered.loc[df_filtered[selected_metric].idxmax(), "date"]
        peak_val = df_filtered[selected_metric].max()
        st.info(f"**Peak:** {peak_date.strftime('%Y-%m-%d')} ({peak_val:.1f})")
        
        # Trend direction
        if delta_val and delta_val > 0:
            st.success(f"📈 Increasing trend detected")
        elif delta_val and delta_val < 0:
            st.warning(f"📉 Decreasing trend detected")
        else:
            st.info("➡️ Stable trend")

    st.markdown("---")

    # --- Global Map with Enhanced Styling ---
    st.subheader("🌐 Global Distribution")
    country_agg = df_filtered.groupby("country")[selected_metric].mean().reset_index()
    
    fig_map = px.choropleth(
        country_agg,
        locations="country",
        locationmode="country names",
        color=selected_metric,
        hover_name="country",
        hover_data={selected_metric: ':.2f'},
        color_continuous_scale=PLOTLY_COLORSCALE,
        title=f"Average {selected_metric_label} by Country",
        labels={selected_metric: selected_metric_label}
    )
    fig_map.update_layout(
        geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular'),
        font=dict(family="Arial, sans-serif", size=12),
        title_font_size=16,
        height=500
    )
    fig_map.update_traces(hovertemplate="<b>%{hovertext}</b><br>%{z:.2f}<extra></extra>")
    st.plotly_chart(fig_map, use_container_width=True)

    # --- Enhanced Key Insights with Moving Average ---
    st.subheader("📌 Detailed Analysis")
    col_ins1, col_ins2 = st.columns(2)
    
    with col_ins1:
        top_locs = df_filtered.groupby("location_name")[selected_metric].mean().nlargest(10).reset_index()
        fig_bar = px.bar(
            top_locs, 
            x=selected_metric, 
            y="location_name", 
            orientation='h', 
            title=f"Top 10 Locations by {selected_metric_label}",
            color=selected_metric,
            color_continuous_scale=PLOTLY_COLORSCALE,
            labels={selected_metric: selected_metric_label, "location_name": "Location"}
        )
        fig_bar.update_layout(
            yaxis={'categoryorder':'total ascending'},
            height=400,
            showlegend=False,
            font=dict(family="Arial, sans-serif", size=11)
        )
        fig_bar.update_traces(hovertemplate="<b>%{y}</b><br>%{x:.2f}<extra></extra>")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_ins2:
        daily_agg = df_filtered.groupby("date")[selected_metric].mean().reset_index()
        daily_agg = daily_agg.sort_values("date")
        
        fig_line = go.Figure()
        
        # Main line
        fig_line.add_trace(go.Scatter(
            x=daily_agg["date"],
            y=daily_agg[selected_metric],
            mode='lines',
            name=selected_metric_label,
            line=dict(color=COLOR_SCHEME['primary'], width=2),
            hovertemplate="<b>Date:</b> %{x}<br><b>Value:</b> %{y:.2f}<extra></extra>"
        ))
        
        # Moving average line
        if show_trend_line and f"{moving_avg_window}d_moving_avg" in df_filtered.columns:
            ma_agg = df_filtered.groupby("date")[f"{moving_avg_window}d_moving_avg"].mean().reset_index()
            ma_agg = ma_agg.sort_values("date")
            fig_line.add_trace(go.Scatter(
                x=ma_agg["date"],
                y=ma_agg[f"{moving_avg_window}d_moving_avg"],
                mode='lines',
                name=f'{moving_avg_window}-Day Moving Avg',
                line=dict(color=COLOR_SCHEME['secondary'], width=2, dash='dash'),
                hovertemplate="<b>Date:</b> %{x}<br><b>MA:</b> %{y:.2f}<extra></extra>"
            ))
        
        # Confidence band (optional)
        if show_confidence_band:
            std_agg = df_filtered.groupby("date")[selected_metric].std().reset_index()
            std_agg = std_agg.sort_values("date")
            upper_bound = daily_agg[selected_metric] + 1.96 * std_agg[selected_metric]
            lower_bound = daily_agg[selected_metric] - 1.96 * std_agg[selected_metric]
            
            fig_line.add_trace(go.Scatter(
                x=daily_agg["date"],
                y=upper_bound,
                mode='lines',
                name='Upper 95% CI',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            fig_line.add_trace(go.Scatter(
                x=daily_agg["date"],
                y=lower_bound,
                mode='lines',
                name='Lower 95% CI',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(31, 119, 180, 0.2)',
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Anomaly markers
        if enable_anomaly_detection and "is_anomaly_high" in df_filtered.columns:
            anomalies_high = df_filtered[df_filtered["is_anomaly_high"]].groupby("date")[selected_metric].mean().reset_index()
            if not anomalies_high.empty:
                fig_line.add_trace(go.Scatter(
                    x=anomalies_high["date"],
                    y=anomalies_high[selected_metric],
                    mode='markers',
                    name='High Anomalies',
                    marker=dict(color=COLOR_SCHEME['warning'], size=8, symbol='triangle-up'),
                    hovertemplate="<b>Anomaly:</b> %{y:.2f}<br>Date: %{x}<extra></extra>"
                ))
        
        fig_line.update_layout(
            title=f"Global Daily Trend: {selected_metric_label}",
            xaxis_title="Date",
            yaxis_title=selected_metric_label,
            height=400,
            hovermode='x unified',
            font=dict(family="Arial, sans-serif", size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_line, use_container_width=True)

# ------------------------- PAGE 2: STATISTICAL ANALYSIS -------------------------
elif nav_selection == "Statistical Analysis":
    st.title("📈 Statistical Analysis")
    st.markdown("Comprehensive statistical insights and relationships in climate data")

    # --- Descriptive Statistics (FIXED) ---
    st.subheader("📊 Descriptive Statistics")

    stats_cols = [
    selected_metric,
    "precip_mm",
    "humidity",
    "wind_kph",
    "pressure_mb",
    "temperature_celsius"
    ]

    available_stats_cols = [c for c in stats_cols if c in df_filtered.columns]

    stats_df = df_filtered[available_stats_cols].describe().T.round(2)

   # Add extra statistics
    stats_df["Skewness"] = df_filtered[available_stats_cols].skew().round(2)
    stats_df["Kurtosis"] = df_filtered[available_stats_cols].kurtosis().round(2)

   # 🔑 CRITICAL FIX (DO NOT SKIP)
    stats_df = stats_df.reset_index().rename(columns={"index": "Metric"})
    stats_df.columns = pd.Index(stats_df.columns).astype(str)

   # Safe display in Streamlit
    st.dataframe(
     stats_df.style
        .background_gradient(cmap="Blues", subset=["mean", "std", "min", "max"])
        .format(precision=2),
     use_container_width=True,
     hide_index=True
    )
    # Statistical insights
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.metric("Total Observations", f"{len(df_filtered):,}")
    with col_stat2:
        st.metric("Variables Analyzed", len(available_stats_cols))
    with col_stat3:
        missing_pct = (df_filtered[available_stats_cols].isna().sum().sum() / (len(df_filtered) * len(available_stats_cols))) * 100
        st.metric("Data Completeness", f"{100 - missing_pct:.1f}%")

    st.markdown("---")

    # --- Enhanced Correlation Matrix ---
    st.subheader("🔗 Correlation Analysis")
    corr_cols = ["temperature_celsius", "humidity", "precip_mm", "wind_kph", "pressure_mb", "uv_index", "heat_index_c", "wind_chill_c"]
    present_cols = [c for c in corr_cols if c in df_filtered.columns]
    corr_matrix = df_filtered[present_cols].corr()
    
    # Create custom labels
    label_dict = {
        "temperature_celsius": "Temperature",
        "humidity": "Humidity",
        "precip_mm": "Precipitation",
        "wind_kph": "Wind Speed",
        "pressure_mb": "Pressure",
        "uv_index": "UV Index",
        "heat_index_c": "Heat Index",
        "wind_chill_c": "Wind Chill"
    }
    corr_matrix.index = [label_dict.get(col, col) for col in corr_matrix.index]
    corr_matrix.columns = [label_dict.get(col, col) for col in corr_matrix.columns]
    
    fig_corr = px.imshow(
        corr_matrix, 
        text_auto='.2f', 
        aspect="auto", 
        color_continuous_scale=PLOTLY_CORR_COLORSCALE,
        title="Correlation Heatmap - Climate Variables",
        labels=dict(color="Correlation Coefficient")
    )
    fig_corr.update_layout(
        height=600,
        font=dict(family="Arial, sans-serif", size=11),
        title_font_size=16
    )
    fig_corr.update_traces(hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>")
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Highlight strongest correlations
    st.markdown("**💡 Strongest Correlations:**")
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append({
                'Variable 1': corr_matrix.columns[i],
                'Variable 2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })
    corr_df = pd.DataFrame(corr_pairs)
    corr_df['Abs Correlation'] = corr_df['Correlation'].abs()
    top_corr = corr_df.nlargest(5, 'Abs Correlation')[['Variable 1', 'Variable 2', 'Correlation']]
    st.dataframe(top_corr.style.background_gradient(cmap="RdBu_r", subset=['Correlation']), hide_index=True)

    st.markdown("---")

    # --- Enhanced Variable Relationships ---
    st.subheader("🔍 Variable Relationships & Scatter Analysis")
    
    with st.expander("⚙️ Scatter Plot Controls", expanded=True):
        col_sc1, col_sc2, col_sc3, col_sc4 = st.columns(4)
        with col_sc1:
            x_axis = st.selectbox("X-Axis Variable", present_cols, index=0, key="scatter_x")
        with col_sc2:
            y_axis = st.selectbox("Y-Axis Variable", present_cols, index=1, key="scatter_y")
        with col_sc3:
            color_by = st.selectbox("Color By", ["season", "country", "None"], index=0, key="scatter_color")
        with col_sc4:
            size_by = st.selectbox("Size By", ["wind_kph", "precip_mm", "humidity", "None"], index=0, key="scatter_size")
    
    # Prepare data for scatter
    scatter_df = df_filtered.copy()
    if color_by == "None":
        color_by = None
    if size_by == "None":
        size_by = None
        
    fig_scatter = px.scatter(
        scatter_df, 
        x=x_axis, 
        y=y_axis, 
        color=color_by,
        size=size_by,
        hover_data=["country", "location_name", "date"],
        title=f"{label_dict.get(x_axis, x_axis)} vs {label_dict.get(y_axis, y_axis)}",
        color_discrete_map=SEASON_COLORS if color_by == "season" else None,
        labels={x_axis: label_dict.get(x_axis, x_axis), y_axis: label_dict.get(y_axis, y_axis)}
    )
    
    # Add trend line
    if show_trend_line:
        try:
            # Remove NaN values for trend line calculation
            valid_data = scatter_df[[x_axis, y_axis]].dropna()
            if len(valid_data) > 1:
                z = np.polyfit(valid_data[x_axis], valid_data[y_axis], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(valid_data[x_axis].min(), valid_data[x_axis].max(), 100)
                fig_scatter.add_trace(go.Scatter(
                    x=x_trend,
                    y=p(x_trend),
                    mode='lines',
                    name='Trend Line',
                    line=dict(color=COLOR_SCHEME['warning'], width=2, dash='dash')
                ))
        except Exception as e:
            # Silently fail if trend line cannot be calculated
            pass
    
    fig_scatter.update_layout(
        height=500,
        font=dict(family="Arial, sans-serif", size=11),
        title_font_size=16,
        legend=dict(bgcolor='rgba(255,255,255,0.8)', bordercolor='rgba(0,0,0,0.2)')
    )
    fig_scatter.update_traces(marker=dict(opacity=0.6, line=dict(width=0.5, color='white')))
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Distribution comparison
    st.subheader("📉 Distribution Comparison")
    dist_col1, dist_col2 = st.columns(2)
    
    with dist_col1:
        selected_dist_var = st.selectbox("Select Variable for Distribution", present_cols, key="dist_var")
        fig_hist = px.histogram(
            df_filtered,
            x=selected_dist_var,
            color="season",
            nbins=30,
            title=f"Distribution of {label_dict.get(selected_dist_var, selected_dist_var)} by Season",
            color_discrete_map=SEASON_COLORS,
            labels={selected_dist_var: label_dict.get(selected_dist_var, selected_dist_var)}
        )
        fig_hist.update_layout(height=400, font=dict(family="Arial, sans-serif", size=11))
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with dist_col2:
        fig_density = px.density_heatmap(
            df_filtered,
            x=x_axis,
            y=y_axis,
            title=f"Density Heatmap: {label_dict.get(x_axis, x_axis)} vs {label_dict.get(y_axis, y_axis)}",
            labels={x_axis: label_dict.get(x_axis, x_axis), y_axis: label_dict.get(y_axis, y_axis)},
            color_continuous_scale=PLOTLY_COLORSCALE
        )
        fig_density.update_layout(height=400, font=dict(family="Arial, sans-serif", size=11))
        st.plotly_chart(fig_density, use_container_width=True)

# ------------------------- PAGE 3: CLIMATE TRENDS -------------------------
elif nav_selection == "Climate Trends":
    st.title("📉 Climate Trends & Distributions")
    st.markdown("Analyze temporal patterns, seasonal variations, and distribution characteristics")

    with st.expander("⚙️ Chart Settings", expanded=True):
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        agg_type = col_ctrl1.selectbox("Time Aggregation", ["Daily", "Weekly", "Monthly", "Seasonal", "Yearly"], key="trend_agg")
        normalize = col_ctrl2.checkbox("Normalize Data (Min-Max Scaling)", value=False, key="trend_norm")
        show_multiple_metrics = col_ctrl3.checkbox("Show Multiple Metrics", value=False, key="trend_multi")

    # Prepare trend data based on aggregation type
    if agg_type == "Monthly":
        trend_df = df_filtered.groupby(["month_name", "month_num", "season"])[selected_metric].mean().reset_index().sort_values("month_num")
        x_col = "month_name"
        x_title = "Month"
    elif agg_type == "Seasonal":
        trend_df = df_filtered.groupby("season")[selected_metric].mean().reset_index()
        x_col = "season"
        x_title = "Season"
    elif agg_type == "Yearly":
        trend_df = df_filtered.groupby("year")[selected_metric].mean().reset_index()
        x_col = "year"
        x_title = "Year"
    elif agg_type == "Weekly":
        df_filtered["week"] = df_filtered["date"].dt.isocalendar().week
        df_filtered["year_week"] = df_filtered["year"].astype(str) + "-W" + df_filtered["week"].astype(str).str.zfill(2)
        trend_df = df_filtered.groupby("year_week")[selected_metric].mean().reset_index()
        x_col = "year_week"
        x_title = "Week"
    else:  # Daily
        trend_df = df_filtered.groupby("date")[selected_metric].mean().reset_index()
        x_col = "date"
        x_title = "Date"

    if normalize:
        trend_df[selected_metric] = (trend_df[selected_metric] - trend_df[selected_metric].min()) / (trend_df[selected_metric].max() - trend_df[selected_metric].min())

    # Main trend visualization
    c1, c2 = st.columns(2)
    
    with c1:
        if show_multiple_metrics:
            # Multi-metric comparison
            metrics_to_show = st.multiselect(
                "Select Metrics to Compare",
                options=list(metric_options.keys())[:5],  # Limit to first 5 for clarity
                default=[selected_metric_label],
                key="multi_metrics"
            )
            
            if metrics_to_show:
                fig_trend = go.Figure()
                for metric_label in metrics_to_show:
                    metric_col = metric_options[metric_label]
                    if agg_type == "Daily":
                        metric_trend = df_filtered.groupby("date")[metric_col].mean().reset_index()
                        x_vals = metric_trend["date"]
                    elif agg_type == "Monthly":
                        metric_trend = df_filtered.groupby(["month_name", "month_num"])[metric_col].mean().reset_index().sort_values("month_num")
                        x_vals = metric_trend["month_name"]
                    else:
                        metric_trend = df_filtered.groupby("season")[metric_col].mean().reset_index()
                        x_vals = metric_trend["season"]
                    
                    fig_trend.add_trace(go.Scatter(
                        x=x_vals,
                        y=metric_trend[metric_col],
                        mode='lines+markers',
                        name=metric_label,
                        line=dict(width=2)
                    ))
                
                fig_trend.update_layout(
                    title=f"{agg_type} Trend Comparison",
                    xaxis_title=x_title,
                    yaxis_title="Value",
                    height=450,
                    hovermode='x unified',
                    font=dict(family="Arial, sans-serif", size=11),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_trend, use_container_width=True)
        else:
            fig_trend = px.area(
                trend_df, 
                x=x_col, 
                y=selected_metric, 
                title=f"{agg_type} Trend ({selected_metric_label})",
                labels={selected_metric: selected_metric_label, x_col: x_title}
            )
            fig_trend.update_traces(
                fill='tonexty',
                fillcolor=f'rgba(31, 119, 180, 0.3)',
                line=dict(color=COLOR_SCHEME['primary'], width=2)
            )
            fig_trend.update_layout(
                height=450,
                font=dict(family="Arial, sans-serif", size=11),
                title_font_size=16
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        
    with c2:
        fig_box = px.box(
            df_filtered, 
            x="season", 
            y=selected_metric, 
            color="season",
            title=f"Seasonal Distribution: {selected_metric_label}",
            color_discrete_map=SEASON_COLORS,
            labels={selected_metric: selected_metric_label, "season": "Season"}
        )
        fig_box.update_layout(
            height=450,
            showlegend=False,
            font=dict(family="Arial, sans-serif", size=11),
            title_font_size=16
        )
        fig_box.update_traces(boxmean='sd')
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")

    # Advanced Distributions
    st.subheader("📊 Distribution Analysis")
    c3, c4 = st.columns(2)

    with c3:
        fig_violin = px.violin(
            df_filtered, 
            y=selected_metric, 
            x="season", 
            box=True, 
            points="outliers",
            title=f"Violin Plot: {selected_metric_label} by Season",
            color="season",
            color_discrete_map=SEASON_COLORS,
            labels={selected_metric: selected_metric_label, "season": "Season"}
        )
        fig_violin.update_layout(
            height=450,
            showlegend=False,
            font=dict(family="Arial, sans-serif", size=11),
            title_font_size=16
        )
        st.plotly_chart(fig_violin, use_container_width=True)
    
    with c4:
        # Enhanced radar chart
        seasonal_means = df_filtered.groupby("season")[["temperature_celsius", "humidity", "wind_kph", "precip_mm"]].mean().reset_index()
        
        # Normalize for radar chart (0-1 scale)
        metrics_for_radar = ["temperature_celsius", "humidity", "wind_kph", "precip_mm"]
        for metric in metrics_for_radar:
            if metric in seasonal_means.columns:
                min_val = seasonal_means[metric].min()
                max_val = seasonal_means[metric].max()
                if max_val > min_val:
                    seasonal_means[f"{metric}_norm"] = (seasonal_means[metric] - min_val) / (max_val - min_val)
                else:
                    seasonal_means[f"{metric}_norm"] = 0.5
        
        radar_df = pd.melt(
            seasonal_means, 
            id_vars=["season"], 
            value_vars=[f"{m}_norm" for m in metrics_for_radar if f"{m}_norm" in seasonal_means.columns],
            var_name="Metric", 
            value_name="Value"
        )
        radar_df["Metric"] = radar_df["Metric"].str.replace("_norm", "").str.replace("_", " ").str.title()
        
        fig_radar = px.line_polar(
            radar_df, 
            r="Value", 
            theta="Metric", 
            color="season", 
            line_close=True, 
            title="Seasonal Climate Profile (Normalized)",
            color_discrete_map=SEASON_COLORS
        )
        fig_radar.update_layout(
            height=450,
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            font=dict(family="Arial, sans-serif", size=11),
            title_font_size=16
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Additional trend analysis
    st.subheader("📈 Comparative Trend Analysis")
    trend_col1, trend_col2 = st.columns(2)
    
    with trend_col1:
        # Year-over-year comparison
        if "year" in df_filtered.columns and len(df_filtered["year"].unique()) > 1:
            yearly_comparison = df_filtered.groupby(["year", "season"])[selected_metric].mean().reset_index()
            fig_yearly = px.line(
                yearly_comparison,
                x="season",
                y=selected_metric,
                color="year",
                markers=True,
                title=f"Year-over-Year Seasonal Comparison: {selected_metric_label}",
                labels={selected_metric: selected_metric_label, "season": "Season", "year": "Year"}
            )
            fig_yearly.update_layout(
                height=400,
                font=dict(family="Arial, sans-serif", size=11),
                title_font_size=14
            )
            st.plotly_chart(fig_yearly, use_container_width=True)
    
    with trend_col2:
        # Location comparison
        top_locations = df_filtered.groupby("location_name")[selected_metric].mean().nlargest(5).index
        location_trends = df_filtered[df_filtered["location_name"].isin(top_locations)]
        
        if agg_type == "Monthly":
            loc_trend_df = location_trends.groupby(["location_name", "month_name", "month_num"])[selected_metric].mean().reset_index().sort_values("month_num")
            x_loc_col = "month_name"
        else:
            loc_trend_df = location_trends.groupby(["location_name", "date"])[selected_metric].mean().reset_index()
            x_loc_col = "date"
        
        fig_loc_trend = px.line(
            loc_trend_df,
            x=x_loc_col,
            y=selected_metric,
            color="location_name",
            title=f"Top 5 Locations Trend: {selected_metric_label}",
            labels={selected_metric: selected_metric_label}
        )
        fig_loc_trend.update_layout(
            height=400,
            font=dict(family="Arial, sans-serif", size=11),
            title_font_size=14,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_loc_trend, use_container_width=True)

# ------------------------- PAGE 4: EXTREME EVENTS -------------------------
elif nav_selection == "Extreme Events":
    st.title("⚡ Extreme Weather Events")
    st.markdown("Identify and analyze extreme weather patterns and anomalies")

    st.sidebar.markdown("### ⚙️ Extreme Thresholds")
    threshold_method = st.sidebar.radio(
        "Threshold Method",
        ["Manual Input", "Percentile-Based"],
        key="thresh_method"
    )
    
    if threshold_method == "Manual Input":
        temp_thresh = st.sidebar.number_input("Extreme Temp (> X °C)", value=35.0, min_value=0.0, step=0.5, key="temp_man")
        wind_thresh = st.sidebar.number_input("Extreme Wind (> X kph)", value=40.0, min_value=0.0, step=1.0, key="wind_man")
        precip_thresh = st.sidebar.number_input("Extreme Precip (> X mm)", value=20.0, min_value=0.0, step=1.0, key="precip_man")
    else:
        percentile_val = st.sidebar.slider("Percentile Threshold", min_value=90, max_value=99, value=95, step=1, key="percentile")
        temp_thresh = df_filtered["temperature_celsius"].quantile(percentile_val / 100)
        wind_thresh = df_filtered["wind_kph"].quantile(percentile_val / 100)
        precip_thresh = df_filtered["precip_mm"].quantile(percentile_val / 100)
        st.sidebar.info(f"**Auto-calculated:**\n- Temp: {temp_thresh:.1f}°C\n- Wind: {wind_thresh:.1f} kph\n- Precip: {precip_thresh:.1f} mm")

    # Extreme event summary cards
    extremes_heat = df_filtered[df_filtered["temperature_celsius"] > temp_thresh]
    extremes_wind = df_filtered[df_filtered["wind_kph"] > wind_thresh]
    extremes_precip = df_filtered[df_filtered["precip_mm"] > precip_thresh]
    
    ex_col1, ex_col2, ex_col3, ex_col4 = st.columns(4)
    ex_col1.metric("🔥 Heat Events", len(extremes_heat), delta=f"{len(extremes_heat)/len(df_filtered)*100:.1f}%")
    ex_col2.metric("💨 Wind Events", len(extremes_wind), delta=f"{len(extremes_wind)/len(df_filtered)*100:.1f}%")
    ex_col3.metric("🌧️ Precip Events", len(extremes_precip), delta=f"{len(extremes_precip)/len(df_filtered)*100:.1f}%")
    ex_col4.metric("📊 Total Extremes", len(extremes_heat) + len(extremes_wind) + len(extremes_precip))

    st.markdown("---")

    # Hall of Extremes with enhanced display
    st.subheader("🏆 Hall of Extremes")
    col_ex1, col_ex2, col_ex3, col_ex4 = st.columns(4)

    with col_ex1:
        st.markdown("**🔥 Hottest Days**")
        hottest = df_filtered.nlargest(5, "temperature_celsius")[["date", "location_name", "country", "temperature_celsius"]].copy()
        hottest["temperature_celsius"] = hottest["temperature_celsius"].round(1)
        st.dataframe(
            hottest.style.background_gradient(cmap="Reds", subset=["temperature_celsius"]),
            hide_index=True,
            use_container_width=True
        )

    with col_ex2:
        st.markdown("**❄️ Coldest Days**")
        coldest = df_filtered.nsmallest(5, "temperature_celsius")[["date", "location_name", "country", "temperature_celsius"]].copy()
        coldest["temperature_celsius"] = coldest["temperature_celsius"].round(1)
        st.dataframe(
            coldest.style.background_gradient(cmap="Blues", subset=["temperature_celsius"]),
            hide_index=True,
            use_container_width=True
        )

    with col_ex3:
        st.markdown("**💨 Windiest Days**")
        windiest = df_filtered.nlargest(5, "wind_kph")[["date", "location_name", "country", "wind_kph"]].copy()
        windiest["wind_kph"] = windiest["wind_kph"].round(1)
        st.dataframe(
            windiest.style.background_gradient(cmap="Oranges", subset=["wind_kph"]),
            hide_index=True,
            use_container_width=True
        )
    
    with col_ex4:
        st.markdown("**🌧️ Wettest Days**")
        wettest = df_filtered.nlargest(5, "precip_mm")[["date", "location_name", "country", "precip_mm"]].copy()
        wettest["precip_mm"] = wettest["precip_mm"].round(1)
        st.dataframe(
            wettest.style.background_gradient(cmap="Blues", subset=["precip_mm"]),
            hide_index=True,
            use_container_width=True
        )

    st.markdown("---")

    # Extreme Event Analysis
    st.subheader("📊 Extreme Event Analysis")
    
    extremes = df_filtered[
        (df_filtered["temperature_celsius"] > temp_thresh) |
        (df_filtered["wind_kph"] > wind_thresh) |
        (df_filtered["precip_mm"] > precip_thresh)
    ].copy()

    if not extremes.empty:
        def tag_extreme(row):
            tags = []
            if row["temperature_celsius"] > temp_thresh: tags.append("Heatwave")
            if row["wind_kph"] > wind_thresh: tags.append("High Wind")
            if row["precip_mm"] > precip_thresh: tags.append("Heavy Rain")
            return ", ".join(tags) if tags else "None"

        extremes["Event Type"] = extremes.apply(tag_extreme, axis=1)
        extremes = extremes[extremes["Event Type"] != "None"]
        
        if not extremes.empty:
            # Event frequency visualization
            event_counts = extremes["Event Type"].str.split(", ", expand=True).stack().value_counts().reset_index()
            event_counts.columns = ["Event Type", "Count"]
            
            fig_freq = px.bar(
                event_counts, 
                x="Event Type", 
                y="Count", 
                color="Event Type",
                title="Frequency of Extreme Events",
                color_discrete_sequence=[COLOR_SCHEME['warning'], COLOR_SCHEME['secondary'], COLOR_SCHEME['info']],
                labels={"Count": "Number of Events", "Event Type": "Event Type"}
            )
            fig_freq.update_layout(
                height=400,
                showlegend=False,
                font=dict(family="Arial, sans-serif", size=11),
                title_font_size=16
            )
            fig_freq.update_traces(hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>")
            st.plotly_chart(fig_freq, use_container_width=True)

            # Temporal distribution of extremes
            st.subheader("📅 Temporal Distribution of Extreme Events")
            ext_col1, ext_col2 = st.columns(2)
            
            with ext_col1:
                extremes["month"] = extremes["date"].dt.month_name()
                monthly_extremes = extremes.groupby(["month", "Event Type"]).size().reset_index(name="Count")
                monthly_extremes["month_num"] = monthly_extremes["month"].map({
                    "January": 1, "February": 2, "March": 3, "April": 4,
                    "May": 5, "June": 6, "July": 7, "August": 8,
                    "September": 9, "October": 10, "November": 11, "December": 12
                })
                monthly_extremes = monthly_extremes.sort_values("month_num")
                
                fig_monthly = px.bar(
                    monthly_extremes,
                    x="month",
                    y="Count",
                    color="Event Type",
                    title="Extreme Events by Month",
                    labels={"Count": "Number of Events", "month": "Month"},
                    color_discrete_sequence=[COLOR_SCHEME['warning'], COLOR_SCHEME['secondary'], COLOR_SCHEME['info']]
                )
                fig_monthly.update_layout(
                    height=400,
                    font=dict(family="Arial, sans-serif", size=11),
                    title_font_size=14,
                    xaxis={'categoryorder': 'array', 'categoryarray': monthly_extremes["month"].unique()}
                )
                st.plotly_chart(fig_monthly, use_container_width=True)
            
            with ext_col2:
                seasonal_extremes = extremes.groupby(["season", "Event Type"]).size().reset_index(name="Count")
                fig_seasonal = px.bar(
                    seasonal_extremes,
                    x="season",
                    y="Count",
                    color="Event Type",
                    title="Extreme Events by Season",
                    labels={"Count": "Number of Events", "season": "Season"},
                    color_discrete_sequence=[COLOR_SCHEME['warning'], COLOR_SCHEME['secondary'], COLOR_SCHEME['info']]
                )
                fig_seasonal.update_layout(
                    height=400,
                    font=dict(family="Arial, sans-serif", size=11),
                    title_font_size=14
                )
                st.plotly_chart(fig_seasonal, use_container_width=True)

            # Regional breakdown
            st.subheader("🌍 Regional Breakdown of Extremes")
            regional_ext = extremes.groupby(["country", "Event Type"]).size().reset_index(name="Count")
            
            fig_reg_scat = px.scatter(
                regional_ext, 
                x="country", 
                y="Event Type", 
                size="Count", 
                color="Count",
                title="Regional Extreme Event Intensity",
                color_continuous_scale="Reds",
                labels={"Count": "Number of Events", "country": "Country", "Event Type": "Event Type"},
                hover_data=["Count"]
            )
            fig_reg_scat.update_layout(
                height=500,
                font=dict(family="Arial, sans-serif", size=11),
                title_font_size=16
            )
            st.plotly_chart(fig_reg_scat, use_container_width=True)
            
            # Location-level heatmap
            st.subheader("📍 Location-Level Extreme Event Heatmap")
            location_ext = extremes.groupby(["location_name", "Event Type"]).size().reset_index(name="Count")
            location_pivot = location_ext.pivot(index="location_name", columns="Event Type", values="Count").fillna(0)
            
            if not location_pivot.empty:
                fig_heatmap = px.imshow(
                    location_pivot.head(20),  # Top 20 locations
                    labels=dict(x="Event Type", y="Location", color="Count"),
                    title="Top 20 Locations - Extreme Event Frequency",
                    color_continuous_scale="Reds",
                    aspect="auto"
                )
                fig_heatmap.update_layout(
                    height=600,
                    font=dict(family="Arial, sans-serif", size=10),
                    title_font_size=16
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("No extreme events found with the current thresholds.")
    else:
        st.info("No extreme events found with the current thresholds. Try adjusting the threshold values in the sidebar.")

# ------------------------- PAGE 5: HELP & USER GUIDE -------------------------
elif nav_selection == "Help & User Guide":
    st.title("📘 User Guide & Documentation")
    st.markdown("Comprehensive guide to using the ClimateScope Interactive Dashboard")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Getting Started", "Features", "Visualizations", "Tips & Tricks"])
    
    with tab1:
        st.header("🚀 Getting Started")
        st.markdown("""
        ### Welcome to ClimateScope!
        
        ClimateScope is an interactive dashboard for analyzing climate and weather data. Follow these steps to get started:
        
        #### 1. **Navigation**
        - Use the sidebar radio buttons to switch between different analysis pages:
          - **Executive Dashboard**: High-level overview and KPIs
          - **Statistical Analysis**: Detailed statistical insights and correlations
          - **Climate Trends**: Temporal patterns and seasonal analysis
          - **Extreme Events**: Identification and analysis of extreme weather
          - **Help & User Guide**: This page
        
        #### 2. **Global Filters** (Sidebar)
        - **Country Selection**: Toggle "Select All Countries" or choose specific countries
        - **Location Selection**: Filter by specific locations/cities
        - **Date Range**: Select the time period for analysis
        - **Primary Metric**: Choose the main variable to analyze (Temperature, Precipitation, etc.)
        
        #### 3. **Advanced Controls**
        - **Visualization Settings**: Adjust moving averages, trend lines, and confidence bands
        - **Threshold Settings**: Configure percentile thresholds for anomaly detection
        
        #### 4. **Interacting with Charts**
        - **Zoom**: Click and drag to zoom into specific areas
        - **Pan**: Click and drag to move around the chart
        - **Reset**: Double-click to reset the view
        - **Hover**: Hover over data points to see detailed information
        - **Download**: Use the camera icon to download charts as PNG images
        """)
    
    with tab2:
        st.header("✨ Key Features")
        
        col_feat1, col_feat2 = st.columns(2)
        
        with col_feat1:
            st.subheader("📊 Interactive Visualizations")
            st.markdown("""
            - **Interactive Maps**: Choropleth maps showing global distribution
            - **Dynamic Charts**: All charts update based on your filters
            - **Multiple Chart Types**: Line, bar, scatter, box, violin, radar, and more
            - **Real-time Updates**: Changes reflect immediately when filters are adjusted
            """)
            
            st.subheader("🔍 Advanced Analytics")
            st.markdown("""
            - **Moving Averages**: Configurable time windows (1-30 days)
            - **Anomaly Detection**: Automatic identification of outliers
            - **Correlation Analysis**: Heatmaps showing variable relationships
            - **Trend Analysis**: Multiple aggregation levels (Daily, Weekly, Monthly, Seasonal, Yearly)
            """)
        
        with col_feat2:
            st.subheader("🌍 Geographic Analysis")
            st.markdown("""
            - **Country-level Filtering**: Analyze specific regions
            - **Location-level Details**: Drill down to city/location level
            - **Regional Comparisons**: Compare patterns across different areas
            - **Global Distribution Maps**: Visualize spatial patterns
            """)
            
            st.subheader("⚡ Extreme Event Detection")
            st.markdown("""
            - **Customizable Thresholds**: Manual or percentile-based
            - **Event Classification**: Automatic tagging of heatwaves, high winds, heavy rain
            - **Temporal Analysis**: Monthly and seasonal breakdowns
            - **Regional Mapping**: Identify hotspots for extreme events
            """)
    
    with tab3:
        st.header("📈 Visualization Guide")
        
        st.subheader("Executive Dashboard")
        st.markdown("""
        - **KPI Cards**: Key performance indicators with delta indicators
        - **Global Map**: Choropleth map showing average values by country
        - **Top Locations**: Bar chart of locations with highest values
        - **Trend Line**: Time series with optional moving averages and confidence bands
        - **Insights Panel**: Automated findings and statistical highlights
        """)
        
        st.subheader("Statistical Analysis")
        st.markdown("""
        - **Descriptive Statistics**: Comprehensive statistical summary table
        - **Correlation Matrix**: Heatmap showing relationships between variables
        - **Scatter Plots**: Interactive scatter plots with customizable axes and coloring
        - **Distribution Charts**: Histograms and density plots
        - **Trend Lines**: Optional regression lines on scatter plots
        """)
        
        st.subheader("Climate Trends")
        st.markdown("""
        - **Time Aggregation**: Daily, Weekly, Monthly, Seasonal, or Yearly views
        - **Multi-metric Comparison**: Compare multiple variables simultaneously
        - **Seasonal Analysis**: Box plots and violin plots by season
        - **Radar Charts**: Normalized seasonal climate profiles
        - **Year-over-Year**: Compare trends across different years
        - **Location Trends**: Track top locations over time
        """)
        
        st.subheader("Extreme Events")
        st.markdown("""
        - **Hall of Extremes**: Tables showing hottest, coldest, windiest, wettest days
        - **Event Frequency**: Bar charts of extreme event counts
        - **Temporal Distribution**: Monthly and seasonal breakdowns
        - **Regional Analysis**: Scatter plots and heatmaps by country/location
        - **Threshold Configuration**: Manual or percentile-based thresholds
        """)
    
    with tab4:
        st.header("💡 Tips & Best Practices")
        
        st.subheader("Optimizing Your Analysis")
        st.markdown("""
        1. **Start Broad, Then Narrow**: Begin with all countries/locations, then filter down
        2. **Use Date Ranges Wisely**: Longer periods show trends, shorter periods show details
        3. **Compare Metrics**: Use the multi-metric feature to understand relationships
        4. **Check Anomalies**: Enable anomaly detection to find outliers automatically
        5. **Explore Correlations**: Use the correlation matrix to identify related variables
        """)
        
        st.subheader("Interpreting Results")
        st.markdown("""
        - **Moving Averages**: Smooth out daily fluctuations to see underlying trends
        - **Confidence Bands**: Show the range of expected values (95% confidence interval)
        - **Percentile Thresholds**: Use for anomaly detection based on historical data
        - **Seasonal Patterns**: Look for consistent patterns across seasons
        - **Regional Differences**: Compare countries to identify climate variations
        """)
        
        st.subheader("Performance Tips")
        st.markdown("""
        - **Filter Early**: Apply filters before loading large datasets
        - **Limit Locations**: When analyzing many locations, consider filtering to top performers
        - **Use Aggregations**: Monthly/Seasonal views are faster than daily for large datasets
        - **Disable Optional Features**: Turn off confidence bands and trend lines if not needed
        """)
        
        st.subheader("Troubleshooting")
        st.markdown("""
        - **No Data**: Check that your filters aren't too restrictive
        - **Charts Not Loading**: Ensure the data file exists and is properly formatted
        - **Missing Values**: Some metrics may have missing data - check the data completeness metric
        - **Slow Performance**: Reduce date range or number of locations selected
        """)
    
    st.markdown("---")
    st.markdown("### 📞 Need More Help?")
    st.info("""
    For additional support or to report issues, please refer to the project documentation or contact the development team.
    All visualizations are powered by Plotly and are fully interactive. Explore the data and discover insights!
    """)