"""sweep_dbscan.py
Run a small parameter sweep for DBSCAN on outputs/anomalies.csv and
report number of clusters, non-noise points, and silhouette score.

Usage:
    python scripts/sweep_dbscan.py --min-mag 3.0 --eps 0.01 0.02 0.05 --min-samples 3 5 10

Outputs to stdout and writes results to outputs/cluster_sweep_results.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


def run_sweep(csv_path, min_mag, eps_list, min_samples_list, out_path):
    df = pd.read_csv(csv_path)
    df = df[df['magnitude'] >= min_mag]
    coords = df[['longitude', 'latitude']].to_numpy()

    results = []
    for eps in eps_list:
        for ms in min_samples_list:
            db = DBSCAN(eps=eps, min_samples=ms)
            labels = db.fit_predict(coords)
            non_noise = int((labels >= 0).sum())
            clusters = int(len(set(labels[labels >= 0])))
            sil = None
            if clusters >= 2:
                try:
                    sil = float(silhouette_score(coords[labels >= 0], labels[labels >= 0]))
                except Exception as e:
                    sil = str(e)
            results.append({
                'eps': eps,
                'min_samples': ms,
                'non_noise': non_noise,
                'clusters': clusters,
                'silhouette': sil
            })
            print(f'eps={eps}, min_samples={ms} -> non_noise={non_noise}, clusters={clusters}, silhouette={sil}')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['eps','min_samples','non_noise','clusters','silhouette'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f'Wrote results to {out_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--min-mag', type=float, default=3.0)
    p.add_argument('--eps', type=float, nargs='+', default=[0.01, 0.02, 0.05, 0.1, 0.5])
    p.add_argument('--min-samples', type=int, nargs='+', default=[3,5,10])
    p.add_argument('--input', type=str, default='outputs/anomalies.csv')
    p.add_argument('--output', type=str, default='outputs/cluster_sweep_results.csv')
    args = p.parse_args()

    run_sweep(Path(args.input), args.min_mag, args.eps, args.min_samples, Path(args.output))
