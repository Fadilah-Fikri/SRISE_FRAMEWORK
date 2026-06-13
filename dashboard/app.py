import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="SRISE Framework Dashboard",
    layout="wide"
)

st.title("🌍 SRISE Framework")
st.subheader("Seismic Risk Segmentation & Anomaly Detection Indonesia")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("outputs/anomalies.csv")
    return df

df = load_data()

# =========================
# SIDEBAR FILTER
# =========================
st.sidebar.header("Filter")

min_mag = st.sidebar.slider(
    "Minimum Magnitude",
    float(df["magnitude"].min()),
    float(df["magnitude"].max()),
    3.0
)

filtered_df = df[
    df["magnitude"] >= min_mag
]

# =========================
# METRICS
# =========================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Earthquakes", len(filtered_df))
col2.metric("Clusters", filtered_df["cluster"].nunique())
col3.metric(
    "Anomalies",
    (filtered_df["anomaly_score"] == -1).sum()
)
col4.metric(
    "Avg Risk",
    round(filtered_df["risk_score"].mean(), 3)
)

# =========================
# CLUSTER VISUALIZATION
# =========================
st.subheader("DBSCAN Cluster Segmentation")

fig_cluster = px.scatter(
    filtered_df,
    x="longitude",
    y="latitude",
    color="cluster",
    hover_data=["magnitude", "risk_score"]
)

st.plotly_chart(
    fig_cluster,
    use_container_width=True
)

# =========================
# RISK DISTRIBUTION
# =========================
st.subheader("Risk Score Distribution")

fig_risk = px.histogram(
    filtered_df,
    x="risk_score"
)

st.plotly_chart(
    fig_risk,
    use_container_width=True
)

# =========================
# TOP RISK AREAS
# =========================
st.subheader("Top 10 Highest Risk Areas")

top_risk = filtered_df.sort_values(
    "risk_score",
    ascending=False
).head(10)

st.dataframe(
    top_risk[
        [
            "location",
            "magnitude",
            "depth",
            "risk_score"
        ]
    ]
)

# =========================
# INTERACTIVE MAP
# =========================
st.subheader("Anomaly Risk Map")

m = folium.Map(
    location=[-2.5, 118],
    zoom_start=5
)

for _, row in filtered_df.iterrows():
    if row["anomaly_score"] == -1:
        folium.CircleMarker(
            location=[
                row["latitude"],
                row["longitude"]
            ],
            radius=4,
            popup=(
                f"Location: {row['location']}<br>"
                f"Magnitude: {row['magnitude']}<br>"
                f"Risk: {row['risk_score']:.2f}"
            )
        ).add_to(m)

st_folium(
    m,
    width=1200,
    height=600
)