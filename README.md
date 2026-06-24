# SRISE Framework (Seismic Risk Index & Segmentation Engine)

SRISE Framework is a multi-stage geospatial analytics framework designed for seismic risk segmentation and anomaly detection in Indonesia using historical earthquake catalog data.

This framework integrates clustering, spatial density estimation, spatial autocorrelation, multi-factor risk scoring, and anomaly detection into one unified analytical pipeline.

## Core Methodology

- **DBSCAN** → Spatial clustering for seismic zone segmentation
- **Kernel Density Estimation (KDE)** → Earthquake density hotspot modeling
- **Moran’s I Spatial Autocorrelation** → Global spatial dependency analysis
- **Risk Index Scoring** → Composite seismic risk quantification
- **Isolation Forest** → Detection of abnormal seismic events
- **GeoPandas + Folium** → Interactive spatial visualization
- **Streamlit Dashboard** → Real-time risk monitoring dashboard

## Dataset

Source:
Earthquakes in Indonesia Catalog V2 (BMKG-based historical seismic records)

Dataset contains:

- Latitude
- Longitude
- Magnitude
- Depth
- Location
- Event metadata

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

- Disaster mitigation
- Seismic hotspot monitoring
- Risk zoning
- Early warning intelligence
- Spatial anomaly detection

## Deployment

Run dashboard:

```bash
streamlit run dashboard/app.py
```

## Output

- Cluster segmentation maps
- Density hotspot maps
- Risk-scored earthquake zones
- High-risk anomaly maps
- Interactive monitoring dashboard

## Intellectual Property (HKI)

This framework is developed as part of an Intellectual Property (HKI) submission and academic publication pipeline.

## Dataset Upload — Skenario & Instruksi

Dashboard menyertakan fitur upload dataset CSV untuk memperbarui data `outputs/anomalies.csv` secara interaktif.
Berikut skenario yang mungkin terjadi ketika pengguna mengunggah file baru dan tindakan yang direkomendasikan:

- **Preview only**
  - Aksi: file hanya dibaca dan ditampilkan preview di sidebar.
  - Efek: tidak ada perubahan pada file `outputs/anomalies.csv`.
  - Rekomendasi: selalu gunakan mode ini pertama kali untuk verifikasi struktur kolom dan isi singkat.

- **Replace dataset (overwrite)**
  - Aksi: file upload akan menggantikan `outputs/anomalies.csv`.
  - Backup: jika opsi backup dicentang, file lama disalin ke `outputs/anomalies_backup_YYYYMMDD_HHMMSS.csv` sebelum overwrite.
  - Reload: aplikasi memanggil reload. Catatan penting: fungsi load menggunakan `@st.cache_data` — jika cache tidak dibersihkan, dashboard mungkin masih menampilkan data lama. Disarankan menambahkan `st.cache_data.clear()` atau memicu invalidasi cache sebelum `st.experimental_rerun()` agar file baru segera terbaca.
  - Dampak: semua metrik, peta dan cluster yang ditampilkan akan berubah sesuai file baru. Jika file baru tidak berisi kolom `cluster`, gunakan opsi _Enable reclustering with DBSCAN_ di sidebar untuk menghitung label cluster baru.

- **Append to dataset**
  - Aksi: konten upload dikombinasikan (concatenate) dengan `outputs/anomalies.csv` dan disimpan kembali.
  - Deduplikasi: jika kolom `eventID` atau `id` tersedia, sistem akan drop-duplicate berdasarkan kolom tersebut (keputusan: keep last). Jika tidak ada identifier, duplikat tidak dihapus secara otomatis.
  - Backup & Reload: backup dapat dibuat sebelum penulisan; aplikasi akan mencoba reload setelah penyimpanan (lihat catatan cache di atas).
  - Dampak: jumlah baris bertambah; cluster lama dan baru mungkin tidak konsisten — disarankan menjalankan reclustering DBSCAN atau sweep parameter setelah append.

## Validasi & Edge-cases

- **Kolom wajib**: setidaknya `latitude`, `longitude`, `magnitude` disarankan. Jika kolom tidak ada atau tipe non-numerik, beberapa visualisasi/algoritma bisa gagal.
- **File besar**: operasi append/dedupe dan reload dapat memakan memori dan waktu; pertimbangkan memproses offline untuk dataset sangat besar.
- **Cache stale**: jika dashboard masih menampilkan data lama setelah perubahan, panggil `st.cache_data.clear()` sebelum `st.experimental_rerun()` atau restart Streamlit.

## Rekomendasi alur kerja singkat

1. Pilih `Preview only` dan periksa header/5 baris pertama.
2. Aktifkan `Backup existing dataset` sebelum melakukan `Replace` atau `Append`.
3. Setelah mengganti/menambah data, jalankan `Enable reclustering with DBSCAN` di sidebar atau jalankan sweep parameter dengan `scripts/sweep_dbscan.py` untuk menemukan parameter DBSCAN yang cocok.

Contoh perintah sweep DBSCAN (dari root workspace):

```bash
python scripts/sweep_dbscan.py --min-mag 3.0 --eps 0.01 0.02 0.05 0.1 --min-samples 3 5 10
```

Contoh menjalankan dashboard:

```bash
streamlit run dashboard/app.py
```

## Pemulihan (restore)

Jika backup dibuat, Anda dapat mengembalikan file lama dengan menyalin file `outputs/anomalies_backup_YYYYMMDD_HHMMSS.csv` ke `outputs/anomalies.csv` dan me-restart dashboard.

---

_Dokumentasi ini menjelaskan perilaku yang diimplementasikan di `dashboard/app.py` (uploader, opsi backup, dan perilaku reload)._
