import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
import shutil
from pathlib import Path
import datetime

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Dashboard SRISE Framework",
    layout="wide"
)

st.title("🌍 SRISE Framework")
st.subheader("Seismic Risk & Anomaly Detection in Indonesia")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("outputs/anomalies.csv")
    return df

df = load_data()

EXPECTED_COLUMNS = {
    "id",
    "eventid",
    "date",
    "time",
    "latitude",
    "longitude",
    "magnitude",
    "mag_type",
    "depth",
    "phasecount",
    "azimuth_gap",
    "location",
    "datetime",
    "location_frequency",
    "cluster",
    "density_score",
    "mag_norm",
    "depth_norm",
    "freq_norm",
    "density_norm",
    "risk_score",
    "anomaly_score",
}

def validate_anomaly_schema(df):
    columns_lower = {c.strip().lower(): c for c in df.columns}
    missing = [c for c in EXPECTED_COLUMNS if c not in columns_lower]
    return missing, columns_lower

missing_columns, _ = validate_anomaly_schema(df)
if missing_columns:
    st.error(
        "Dataset outputs/anomalies.csv tidak lengkap atau formatnya salah. "
        "Pastikan file memiliki kolom yang sama seperti format anomaly dataset sebelumnya."
    )

# =========================
# SIDEBAR FILTER
# =========================
st.sidebar.header("Filter")

min_mag = st.sidebar.slider(
    "Magnitudo Minimum",
    float(df["magnitude"].min()),
    float(df["magnitude"].max()),
    3.0
)

filtered_df = df[
    df["magnitude"] >= min_mag
]

# -------------------------
# Dataset upload (optional)
# -------------------------
uploaded_file = st.sidebar.file_uploader("Unggah dataset CSV (opsional)", type=["csv"])
if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"Gagal membaca CSV: {e}")
        uploaded_df = None

    if uploaded_df is not None:
        st.sidebar.markdown("**Pratinjau file yang diunggah**")
        st.sidebar.write(f"Baris: {uploaded_df.shape[0]}, Kolom: {uploaded_df.shape[1]}")
        st.sidebar.dataframe(uploaded_df.head())

        upload_action = st.sidebar.radio(
            "Aksi unggah",
            ("Hanya Pratinjau", "Ganti dataset (timpa)", "Tambahkan ke dataset")
        )

        do_backup = st.sidebar.checkbox("Backup dataset lama sebelum terapkan", value=True)

        required_cols = {
            "latitude",
            "longitude",
            "magnitude",
            "risk_score",
            "anomaly_score",
            "location",
            "cluster",
            "date",
            "time",
            "depth",
        }

        cols_lower = {c.strip().lower(): c for c in uploaded_df.columns}
        missing_cols = [c for c in required_cols if c not in cols_lower]
        if missing_cols:
            st.sidebar.error(
                f"Kolom wajib hilang: {', '.join(missing_cols)}. File tidak akan mengganti atau menambahkan dataset."
            )

        if st.sidebar.button("Terapkan Unggahan"):
            out_path = Path("outputs") / "anomalies.csv"
            if upload_action == "Hanya Pratinjau":
                st.sidebar.info("Mode pratinjau: tidak ada perubahan yang dilakukan.")

            elif upload_action == "Ganti dataset (timpa)":
                if missing_cols:
                    st.sidebar.error("Tidak dapat mengganti dataset: file tidak lengkap atau format salah.")
                else:
                    if out_path.exists() and do_backup:
                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        bak = out_path.parent / f"anomalies_backup_{ts}.csv"
                        try:
                            shutil.copy(out_path, bak)
                        except Exception:
                            pass
                    try:
                        uploaded_df.to_csv(out_path, index=False)
                        st.sidebar.success("Dataset berhasil diganti — memuat ulang aplikasi...")
                        try:
                            st.cache_data.clear()
                        except Exception:
                            pass
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"Gagal menyimpan dataset: {e}")

            elif upload_action == "Tambahkan ke dataset":
                if missing_cols:
                    st.sidebar.error("Tidak dapat menambahkan dataset: file tidak lengkap atau format salah.")
                else:
                    if out_path.exists():
                        existing = pd.read_csv(out_path)
                    else:
                        existing = pd.DataFrame()

                    combined = pd.concat([existing, uploaded_df], ignore_index=True)
                    # try to drop duplicates if identifier available (case-insensitive)
                    id_col = None
                    for cand in ("eventid", "id"):
                        for c in combined.columns:
                            if c.strip().lower() == cand:
                                id_col = c
                                break
                        if id_col:
                            break
                    if id_col:
                        combined = combined.drop_duplicates(subset=[id_col], keep="last")

                    if out_path.exists() and do_backup:
                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        bak = out_path.parent / f"anomalies_backup_{ts}.csv"
                        try:
                            shutil.copy(out_path, bak)
                        except Exception:
                            pass

                    try:
                        combined.to_csv(out_path, index=False)
                        st.sidebar.success("Data berhasil ditambahkan — memuat ulang aplikasi...")
                        try:
                            st.cache_data.clear()
                        except Exception:
                            pass
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"Gagal menambahkan dan menyimpan dataset: {e}")


# -------------------------
# Optional reclustering UI
# -------------------------
st.sidebar.header("Reclustering (DBSCAN)")
do_recluster = st.sidebar.checkbox("Enable reclustering with DBSCAN", value=False)
if do_recluster:
    eps = st.sidebar.slider("DBSCAN eps", 0.01, 5.0, 0.5, step=0.01)
    min_samples = st.sidebar.slider("DBSCAN min_samples", 1, 50, 5)
    # run DBSCAN on lon/lat
    coords = filtered_df[["longitude", "latitude"]].to_numpy()
    try:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(coords)
        display_df = filtered_df.copy()
        display_df["cluster"] = labels
    except Exception:
        display_df = filtered_df.copy()
        st.sidebar.error("DBSCAN failed with these parameters")
else:
    display_df = filtered_df.copy()

# =========================
# Risk category based on risk_score

def categorize_risk(score):
    if score >= 0.70:
        return "bahaya"
    if score >= 0.55:
        return "cukup bahaya"
    if score >= 0.45:
        return "sedang"
    return "tidak bahaya"

# Add risk category labels to display_df
display_df = display_df.copy()
display_df["risk_category"] = display_df["risk_score"].apply(categorize_risk)

risk_category_values = {
    "tidak bahaya": 0,
    "sedang": 1,
    "cukup bahaya": 2,
    "bahaya": 3,
}
display_df["risk_category_value"] = display_df["risk_category"].map(risk_category_values)

display_df["risk_category"] = pd.Categorical(
    display_df["risk_category"],
    categories=["bahaya", "cukup bahaya", "sedang", "tidak bahaya"],
    ordered=True,
)

color_by = st.sidebar.radio(
    "Warna berdasarkan",
    ("DBSCAN cluster", "Risk category"),
    index=1
)
color_col = "cluster" if color_by == "DBSCAN cluster" else "risk_category"

# =========================
# METRICS
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Earthquakes", len(filtered_df))
col2.metric(
    "Groups",
    display_df["risk_category"].nunique() if color_by == "Risk category" else display_df["cluster"].nunique()
)
col3.metric(
    "Anomalies",
    (display_df["anomaly_score"] == -1).sum()
)
col4.metric(
    "Avg Risk",
    round(display_df["risk_score"].mean(), 3)
)

# Compute silhouette score using longitude/latitude for non-noise points
def _compute_silhouette(df):
    if "cluster" not in df.columns:
        return None
    labels = df["cluster"]
    # exclude DBSCAN noise labeled as -1
    try:
        mask = labels >= 0
    except Exception:
        return None
    if mask.sum() == 0:
        return None
    n_clusters = labels[mask].nunique()
    if n_clusters < 2:
        return None
    X = df.loc[mask, ["longitude", "latitude"]].to_numpy()
    try:
        return round(float(silhouette_score(X, labels[mask])), 3)
    except Exception:
        return None

silhouette = _compute_silhouette(display_df)
col5.metric("Silhouette Score", silhouette if silhouette is not None else "N/A")

# =========================
# Risk category summary
risk_counts = display_df["risk_category"].value_counts().reindex([
    "bahaya",
    "cukup bahaya",
    "sedang",
    "tidak bahaya"
], fill_value=0)
summary_df = risk_counts.reset_index()
summary_df.columns = ["risk_category", "count"]
summary_df["risk_category_value"] = summary_df["risk_category"].map(risk_category_values)
summary_df = summary_df.sort_values(by="risk_category_value", ascending=False).reset_index(drop=True)
summary_df = summary_df[["risk_category", "risk_category_value", "count"]]
summary_df = summary_df.rename(columns={
    "risk_category": "Kategori Risiko",
    "risk_category_value": "Nilai Risiko",
    "count": "Jumlah"
})

st.subheader("Risk Category Summary")
st.dataframe(summary_df, use_container_width=True)

# =========================
# CLUSTER VISUALIZATION
# =========================
st.subheader("DBSCAN Cluster Segmentation")

fig_cluster = px.scatter(
    display_df,
    x="longitude",
    y="latitude",
    color=color_col,
    hover_data=["magnitude", "risk_score", "risk_category"],
    category_orders={"risk_category": ["bahaya", "cukup bahaya", "sedang", "tidak bahaya"]}
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
    display_df,
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

top_risk = display_df.sort_values(
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
st.subheader("Risk Map")

risk_colors = {
    "bahaya": "red",
    "cukup bahaya": "orange",
    "sedang": "yellow",
    "tidak bahaya": "green",
}

m = folium.Map(
    location=[-2.5, 118],
    zoom_start=5
)

for _, row in display_df.iterrows():
    color = risk_colors.get(row["risk_category"], "blue")
    folium.CircleMarker(
        location=[
            row["latitude"],
            row["longitude"]
        ],
        radius=3,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
        popup=(
            f"Location: {row['location']}<br>"
            f"Magnitude: {row['magnitude']}<br>"
            f"Risk: {row['risk_score']:.2f}<br>"
            f"Kategori: {row['risk_category']}"
        )
    ).add_to(m)

st_folium(
    m,
    width=1200,
    height=600
)

st.subheader("📋 Filtered Earthquake Anomaly Events")

anomaly_df = display_df[
    filtered_df["anomaly_score"] == -1
]

st.metric(
    "Filtered Anomaly Events",
    len(anomaly_df)
)

st.dataframe(
    anomaly_df[
        [
            "date",
            "time",
            "location",
            "magnitude",
            "depth",
            "risk_score"
        ]
    ],
    use_container_width=True
)