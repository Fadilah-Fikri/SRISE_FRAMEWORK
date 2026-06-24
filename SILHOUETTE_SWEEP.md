# SILHOUETTE & DBSCAN Parameter Sweep

Dokumen ini menjelaskan skenario sweep parameter DBSCAN untuk mencari konfigurasi yang menghasilkan setidaknya dua cluster non-noise sehingga Silhouette Score dapat dihitung, plus cara menafsirkan hasil.

## Tujuan

- Menemukan kombinasi parameter `eps` dan `min_samples` untuk DBSCAN yang menghasilkan >= 2 cluster (label >= 0) pada dataset `outputs/anomalies.csv` (opsional menggunakan filter magnitude).
- Hitung Silhouette Score untuk kombinasi valid dan pilih konfigurasi dengan Silhouette tertinggi sebagai kandidat.

## Prasyarat

- Python environment dengan dependensi dari `requirements.txt` (termasuk `scikit-learn`, `pandas`, `numpy`).
- File dataset: `outputs/anomalies.csv` tersedia.

## Skrip sweep

Terdapat skrip contoh: `scripts/sweep_dbscan.py`.

Contoh menjalankan sweep kecil:

```bash
cd /Users/user/Documents/Portofolio/srise_framework
python scripts/sweep_dbscan.py --min-mag 3.0 --eps 0.01 0.02 0.05 0.1 --min-samples 3 5 10
```

Argumen penting:

- `--min-mag`: filter magnitude sebelum clustering (default 3.0).
- `--eps`: daftar `eps` (jarak) DBSCAN yang ingin diuji. Unit adalah derajat lintang/bujur (0.01 ≈ 1.1 km pada ekuator).
- `--min-samples`: daftar nilai `min_samples` yang ingin diuji.
- `--input`: path CSV input (default `outputs/anomalies.csv`).
- `--output`: CSV output (default `outputs/cluster_sweep_results.csv`).

Hasil ditulis ke `outputs/cluster_sweep_results.csv` berisi kolom:

- `eps`, `min_samples`, `non_noise` (jumlah titik yang bukan noise), `clusters` (jumlah cluster non-noise), `silhouette` (nilai Silhouette atau kosong jika tidak bisa dihitung).

## Interpretasi hasil

- `clusters` = 0: semua titik dianggap noise oleh DBSCAN.
- `clusters` = 1: ada hanya satu cluster non-noise (Silhouette tidak dapat dihitung → N/A).
- `clusters` >= 2: Silhouette dihitung pada titik non-noise.

Silhouette:

- Rentang [-1, 1].
- Dekati 1: cluster rapih dan terpisah.
- Sekitar 0: cluster tumpang tindih.
- Negatif: titik lebih dekat ke cluster lain dari pada cluster-nya sendiri.

Catatan DBSCAN dan penyesuaian parameter:

- `eps` kecil → banyak titik jadi noise.
- `eps` besar → cluster akan menyatu menjadi sedikit cluster (mungkin 1 saja).
- `min_samples` besar → butuh lebih banyak tetangga agar titik jadi core → lebih banyak noise.

Praktik: jalankan grid coarse dahulu (mis. `eps` dari 0.01 sampai 0.5, `min_samples` 3,5,10). Jika menemukan konfigurasi yang menghasilkan beberapa cluster, lakukan grid lebih halus di sekitar `eps` tersebut.

## Tips cepat untuk wilayah geospasial

- Jika ingin `eps` dalam meter, konversi ke derajat: 1 derajat ≈ 111 km (lintang), jadi 1 km ≈ 0.009°. Contoh: `eps=0.01` ≈ 1.11 km.

## Langkah selanjutnya

- Jalankan skrip sweep.
- Pilih baris hasil dengan `clusters >= 2` dan Silhouette tertinggi.
- Periksa hasil clustering (peta, cluster counts) di `dashboard/app.py` dengan mengaktifkan `Enable reclustering with DBSCAN` dan memasukkan parameter pilihan.

---

File terkait:

- `scripts/sweep_dbscan.py` (skrip sweep)
- `dashboard/app.py` (dashboard interaktif; telah ditambahkan kontrol reclustering)
