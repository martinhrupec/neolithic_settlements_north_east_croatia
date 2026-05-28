"""
Filter top 10% cluster-a po confidence_score
==============================================

Cilj: izdvojiti najjace discovery kandidate u ZASEBNI layer.
User onda u QGIS-u samo doda ovaj layer bez ikakvog stiliziranja —
sve sto je tu su top 10% high-confidence zone.

Input:  clusters_thr_85_with_confidence.geojson  (1480 cluster-a)
Output: clusters_thr_85_top10pct.geojson         (~148 cluster-a, top 10%)
        clusters_thr_85_top10pct.csv             ranked tablica
"""

import os
import json
import numpy as np
import pandas as pd


ROOT        = r"c:\Users\Martin\Desktop\skripte_za_diplomski"
CLUSTER_DIR = os.path.join(ROOT, "statisticki_testovi", "13_cluster_zones")

INPUT_GEOJSON  = os.path.join(CLUSTER_DIR, "clusters_thr_85_with_confidence.geojson")
OUTPUT_GEOJSON = os.path.join(CLUSTER_DIR, "clusters_thr_85_top10pct.geojson")
OUTPUT_CSV     = os.path.join(CLUSTER_DIR, "clusters_thr_85_top10pct.csv")

TOP_PERCENTAGE = 10.0   # top 10%


def main():
    print("=" * 70)
    print("FILTER TOP 10% by confidence_score")
    print("=" * 70)

    # ----- Ucitaj enriched layer -----
    print(f"\n[1] Ucitavam: {INPUT_GEOJSON}")
    with open(INPUT_GEOJSON, "r", encoding="utf-8") as fh:
        fc = json.load(fh)
    n_total = len(fc["features"])
    print(f"    n_total = {n_total}")

    # ----- Extract confidence_score -----
    scores = np.array([f["properties"]["confidence_score"]
                       for f in fc["features"]])
    print(f"    confidence_score range: {scores.min():.4f} - {scores.max():.4f}")
    print(f"    median: {np.median(scores):.4f}")

    # ----- Threshold za top 10% -----
    threshold = float(np.percentile(scores, 100 - TOP_PERCENTAGE))
    n_keep = int((scores >= threshold).sum())
    print(f"\n[2] Threshold (P{100-TOP_PERCENTAGE:.0f}): "
          f"confidence_score >= {threshold:.4f}")
    print(f"    -> zadrzano {n_keep} cluster-a od {n_total}  "
          f"({100*n_keep/n_total:.1f}%)")

    # ----- Filtriraj features -----
    keep_features = [
        f for f, s in zip(fc["features"], scores) if s >= threshold
    ]
    # Sortiraj descending by confidence_score
    keep_features.sort(
        key=lambda f: f["properties"]["confidence_score"],
        reverse=True,
    )

    # ----- Spremi novi GeoJSON -----
    out_fc = {
        "type":     "FeatureCollection",
        "name":     "clusters_thr_85_top10pct",
        "crs":      fc.get("crs"),
        "features": keep_features,
    }
    with open(OUTPUT_GEOJSON, "w", encoding="utf-8") as fh:
        json.dump(out_fc, fh, ensure_ascii=False)
    print(f"\n[3] -> {OUTPUT_GEOJSON}")

    # ----- Spremi CSV -----
    rows = [f["properties"] for f in keep_features]
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"    -> {OUTPUT_CSV}")

    # ----- Sazetak top 10 -----
    print(f"\n[4] Top 10 najbolja:")
    cols_show = ["cluster_id", "area_km2", "mean_prob_poly", "mean_unc",
                 "confidence_score", "dist_to_known_m", "n_known_inside"]
    cols_show = [c for c in cols_show if c in df.columns]
    print(df[cols_show].head(10).to_string(index=False))

    print(f"\n[5] Karakteristike svih {n_keep} top-10% cluster-a:")
    print(f"    mean area_km2         : {df['area_km2'].mean():.3f}")
    print(f"    mean confidence_score : {df['confidence_score'].mean():.4f}")
    print(f"    mean dist_to_known_m  : {df['dist_to_known_m'].mean():.0f}")
    print(f"    cluster-a koji sadrze poznati: "
          f"{(df['contains_known']==1).sum()} / {n_keep}")

    print(f"\nSpremljeno u: {CLUSTER_DIR}")


if __name__ == "__main__":
    main()
