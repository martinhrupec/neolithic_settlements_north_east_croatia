# -*- coding: utf-8 -*-
"""
05_COMBINE_PROB_UNCERTAINTY — kombinirana confidence analiza
==============================================================

Spaja `probability_neolitik.tif` + `uncertainty_neolitik.tif` u:

  1. confident_neolitik.tif      = clip(prob - std, 0, 1)
                                   Single raster, high vrijednost = visoka
                                   vjerojatnost AND niska nesigurnost.

  2. clusters_thr_85_with_confidence.geojson
                                   Postojeci cluster polygoni + dodani atributi:
                                   - mean_prob_poly       (preracunato iz rastera)
                                   - mean_unc             prosjecna std u polygonu
                                   - mean_confident_prob  prosjecni (prob - std)
                                   - confidence_score     = mean_prob - mean_unc
                                   - is_high_confidence   1 ako mean_prob >= 0.85
                                                            i mean_unc < 0.10
                                                          0 inace

  3. clusters_thr_85_top_discoveries.csv
                                   Top 100 cluster-a sortiranih po confidence_score.
                                   Spreman za diplomski tekst.

Dependencije: rasterio, shapely (vec instalirano).
"""

import os
import json
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import shape, mapping


ROOT = r"c:\Users\Martin\Desktop\skripte_za_diplomski"

HEATMAP_DIR = os.path.join(ROOT, "12_heatmap_qgis")
CLUSTER_DIR = os.path.join(ROOT, "statisticki_testovi", "13_cluster_zones")

PROB_PATH    = os.path.join(HEATMAP_DIR, "probability_neolitik.tif")
UNC_PATH     = os.path.join(HEATMAP_DIR, "uncertainty_neolitik.tif")
GEOJSON_IN   = os.path.join(CLUSTER_DIR, "clusters_thr_85.geojson")

OUT_CONF_TIF = os.path.join(HEATMAP_DIR, "confident_neolitik.tif")
OUT_GEOJSON  = os.path.join(CLUSTER_DIR, "clusters_thr_85_with_confidence.geojson")
OUT_CSV      = os.path.join(CLUSTER_DIR, "clusters_thr_85_top_discoveries.csv")

# Threshold za is_high_confidence flag
HIGH_PROB_THR = 0.85
LOW_STD_THR   = 0.10

NODATA_OUT = -9999.0


def main():
    print("=" * 70, flush=True)
    print("COMBINE PROB + UNCERTAINTY", flush=True)
    print("=" * 70, flush=True)

    # ------------------------------------------------------------
    # 1) Ucitaj rastere i izgradi confident raster
    # ------------------------------------------------------------
    print(f"\n[1] Ucitavam {PROB_PATH}", flush=True)
    with rasterio.open(PROB_PATH) as ds_p:
        prob = ds_p.read(1)
        prob_nd = ds_p.nodata
        transform = ds_p.transform
        crs = ds_p.crs
        height = ds_p.height
        width = ds_p.width

    print(f"[2] Ucitavam {UNC_PATH}", flush=True)
    with rasterio.open(UNC_PATH) as ds_u:
        unc = ds_u.read(1)
        unc_nd = ds_u.nodata

    valid = (
        (prob != (prob_nd if prob_nd is not None else -9999.0)) &
        (unc  != (unc_nd  if unc_nd  is not None else -9999.0)) &
        np.isfinite(prob) & np.isfinite(unc)
    )
    n_valid = int(valid.sum())
    print(f"    valid piksela: {n_valid:,} / {valid.size:,}", flush=True)

    print(f"\n[3] Racunam confident_neolitik = clip(prob - std, 0, 1)...", flush=True)
    confident = np.full(prob.shape, NODATA_OUT, dtype=np.float32)
    confident[valid] = np.clip(prob[valid] - unc[valid], 0.0, 1.0)

    with rasterio.open(
        OUT_CONF_TIF, "w",
        driver="GTiff", width=width, height=height, count=1,
        dtype="float32", crs=crs, transform=transform,
        nodata=NODATA_OUT, compress="LZW", tiled=True,
        BIGTIFF="IF_SAFER",
    ) as ds_out:
        ds_out.write(confident, 1)
    print(f"    -> {OUT_CONF_TIF}", flush=True)

    # Statistika confident
    c_valid = confident[valid]
    print(f"\n    Statistika confident:", flush=True)
    print(f"      mean   = {c_valid.mean():.4f}", flush=True)
    print(f"      median = {np.median(c_valid):.4f}", flush=True)
    print(f"      p>=0.5 = {(c_valid>=0.5).sum():,} "
          f"({100*(c_valid>=0.5).sum()/n_valid:.1f}%)", flush=True)
    print(f"      p>=0.7 = {(c_valid>=0.7).sum():,} "
          f"({100*(c_valid>=0.7).sum()/n_valid:.1f}%)", flush=True)
    print(f"      p>=0.85= {(c_valid>=0.85).sum():,} "
          f"({100*(c_valid>=0.85).sum()/n_valid:.1f}%)", flush=True)

    # ------------------------------------------------------------
    # 2) Enrich cluster polygons
    # ------------------------------------------------------------
    print(f"\n[4] Ucitavam {GEOJSON_IN}", flush=True)
    with open(GEOJSON_IN, "r", encoding="utf-8") as fh:
        fc = json.load(fh)
    n_features = len(fc["features"])
    print(f"    n_polygons = {n_features}", flush=True)

    print(f"\n[5] Obogacujem polygone s uncertainty stats...", flush=True)
    enriched_features = []
    skip_count = 0

    with rasterio.open(PROB_PATH) as ds_p, rasterio.open(UNC_PATH) as ds_u:
        for i, feat in enumerate(fc["features"]):
            geom = shape(feat["geometry"])
            try:
                p_arr, _ = rio_mask(ds_p, [geom], crop=True, filled=False)
                u_arr, _ = rio_mask(ds_u, [geom], crop=True, filled=False)
            except Exception:
                skip_count += 1
                continue

            p_vals = p_arr[0].compressed()
            u_vals = u_arr[0].compressed()
            if len(p_vals) == 0 or len(u_vals) == 0:
                skip_count += 1
                continue

            mean_prob_poly = float(p_vals.mean())
            mean_unc       = float(u_vals.mean())
            mean_conf_prob = float(np.clip(p_vals - u_vals, 0.0, 1.0).mean())
            confidence_score = mean_prob_poly - mean_unc
            is_high_conf = int(
                (mean_prob_poly >= HIGH_PROB_THR) and
                (mean_unc < LOW_STD_THR)
            )

            props = dict(feat["properties"])
            props["mean_prob_poly"]      = round(mean_prob_poly, 4)
            props["mean_unc"]            = round(mean_unc, 4)
            props["mean_confident_prob"] = round(mean_conf_prob, 4)
            props["confidence_score"]    = round(confidence_score, 4)
            props["is_high_confidence"]  = is_high_conf

            enriched_features.append({
                "type":       "Feature",
                "geometry":   feat["geometry"],
                "properties": props,
            })

            if (i + 1) % 200 == 0:
                print(f"    {i+1}/{n_features}...", flush=True)

    print(f"    enriched: {len(enriched_features)}  skipped: {skip_count}", flush=True)

    # ------------------------------------------------------------
    # 3) Spremi novi GeoJSON
    # ------------------------------------------------------------
    out_fc = {
        "type":     "FeatureCollection",
        "name":     "clusters_thr_85_with_confidence",
        "crs":      fc.get("crs"),
        "features": enriched_features,
    }
    with open(OUT_GEOJSON, "w", encoding="utf-8") as fh:
        json.dump(out_fc, fh, ensure_ascii=False)
    print(f"\n[6] -> {OUT_GEOJSON}", flush=True)

    # ------------------------------------------------------------
    # 4) Top discoveries CSV
    # ------------------------------------------------------------
    print(f"\n[7] Top discoveries CSV...", flush=True)
    rows = [feat["properties"] for feat in enriched_features]
    df = pd.DataFrame(rows)
    df_sorted = df.sort_values("confidence_score", ascending=False)
    df_sorted.head(100).to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"    -> {OUT_CSV}  (top 100 by confidence_score)", flush=True)

    # ------------------------------------------------------------
    # 5) Sazetak
    # ------------------------------------------------------------
    n_hc = int((df["is_high_confidence"] == 1).sum())
    print(f"\n{'='*70}", flush=True)
    print(f"SAZETAK", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Total cluster-a obradeno  : {len(df)}", flush=True)
    print(f"  High-confidence cluster-a : {n_hc}  "
          f"(mean_prob>={HIGH_PROB_THR}, mean_unc<{LOW_STD_THR})",
          flush=True)
    print(f"  Top confidence_score      : {df['confidence_score'].max():.4f}", flush=True)
    print(f"  Median confidence_score   : {df['confidence_score'].median():.4f}", flush=True)
    print(f"  Mean mean_unc             : {df['mean_unc'].mean():.4f}", flush=True)

    print(f"\n  Top 10 by confidence_score:", flush=True)
    cols_show = ["cluster_id", "area_km2", "mean_prob_poly", "mean_unc",
                 "confidence_score", "is_high_confidence",
                 "dist_to_known_m", "n_known_inside"]
    cols_show = [c for c in cols_show if c in df.columns]
    print(df_sorted[cols_show].head(10).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
