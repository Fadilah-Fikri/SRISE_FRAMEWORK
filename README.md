# SRISE Framework (Seismic Risk Index & Segmentation Engine)

SRISE Framework is a multi-stage geospatial analytics framework designed for seismic risk segmentation and anomaly detection in Indonesia using historical earthquake catalog data.

This framework integrates clustering, spatial density estimation, spatial autocorrelation, multi-factor risk scoring, and anomaly detection into one unified analytical pipeline.

## Core Methodology

* **DBSCAN** → Spatial clustering for seismic zone segmentation
* **Kernel Density Estimation (KDE)** → Earthquake density hotspot modeling
* **Moran’s I Spatial Autocorrelation** → Global spatial dependency analysis
* **Risk Index Scoring** → Composite seismic risk quantification
* **Isolation Forest** → Detection of abnormal seismic events
* **GeoPandas + Folium** → Interactive spatial visualization
* **Streamlit Dashboard** → Real-time risk monitoring dashboard

## Dataset

Source:
Earthquakes in Indonesia Catalog V2 (BMKG-based historical seismic records)

Dataset contains:

* Latitude
* Longitude
* Magnitude
* Depth
* Location
* Event metadata

Total records: **102,515+ earthquake events**

## Project Structure

```bash
SRISE-HKI/
├── dataset/
├── notebooks/
├── dashboard/
├── models/
├── outputs/
└── requirements.txt
```

## Research Contribution

This project proposes a **novel multi-stage seismic segmentation framework** by integrating spatial clustering, density-based hotspot analysis, and anomaly detection for earthquake risk intelligence.

Potential applications:

* Disaster mitigation
* Seismic hotspot monitoring
* Risk zoning
* Early warning intelligence
* Spatial anomaly detection

## Deployment

Run dashboard:

```bash
streamlit run dashboard/app.py
```

## Output

* Cluster segmentation maps
* Density hotspot maps
* Risk-scored earthquake zones
* High-risk anomaly maps
* Interactive monitoring dashboard

## Intellectual Property (HKI)

This framework is developed as part of an Intellectual Property (HKI) submission and academic publication pipeline.
