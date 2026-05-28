# -*- coding: utf-8 -*-
"""
04_PREDICT_UNCERTAINTY — generira uncertainty raster
======================================================

Za svaki piksel racuna std (varijancu) po 500 stabala u RF ensemble-u.

Logika:
  - probability_neolitik.tif  = mean(500 stabla)  (vec napravljeno u 02)
  - uncertainty_neolitik.tif  = std(500 stabla)   (ovaj script)

Interpretacija std-a:
  - std ~ 0    : sva stabla se slazu (siguran model)
  - std ~ 0.2+ : stabla se snazno ne slazu (model se koleba)

Korisno kombinirati:
  - high prob + low std  = "snazna i pouzdana neolitik predikcija"
  - high prob + high std = "model misli neolitik, ali nije siguran"

Dependancije:  pip install rasterio scikit-learn joblib pandas
Pokretanje:    python 04_predict_uncertainty.py
"""

import os
import json
import time
import numpy as np
import pandas as pd
import joblib
import rasterio


HEATMAP_DIR = r"c:\Users\Martin\Desktop\skripte_za_diplomski\12_heatmap_qgis"
RASTERS_DIR = os.path.join(HEATMAP_DIR, "rasters")
META_PATH   = os.path.join(RASTERS_DIR, "metadata.json")
MODEL_PATH  = r"c:\Users\Martin\Desktop\skripte_za_diplomski\statisticki_testovi\10_random_forest\model_OPTIMALNI_5.joblib"
OUT_PATH    = os.path.join(HEATMAP_DIR, "uncertainty_neolitik.tif")

# Manji batch nego u 02 jer drzimo 500 tree predictions istovremeno
BATCH_SIZE = 50_000

NODATA_UNC = -9999.0


def read_raster(path, band_idx=1):
    with rasterio.open(path) as ds:
        arr = ds.read(band_idx)
        tr  = ds.transform
        crs = ds.crs
        nd  = ds.nodata
    return arr, tr, crs, nd


def save_raster(arr, transform, crs, out_path, dtype, nodata=None):
    rows, cols = arr.shape
    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        width=cols, height=rows, count=1,
        dtype=dtype,
        crs=crs, transform=transform,
        nodata=nodata,
        compress="LZW", tiled=True,
        BIGTIFF="IF_SAFER",
    ) as ds:
        ds.write(arr, 1)


def main():
    print("=" * 70, flush=True)
    print("PREDICT UNCERTAINTY -> uncertainty_neolitik.tif", flush=True)
    print("=" * 70, flush=True)

    # ------------------------------------------------------------
    # 1) Metadata + model
    # ------------------------------------------------------------
    print(f"\n[1] Ucitavam metadata: {META_PATH}", flush=True)
    with open(META_PATH, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    shape = tuple(meta["shape"])
    soil_types_int_to_name = {int(k): v for k, v in meta["soil_types"].items()}
    soil_types_int_to_name[0] = "Nepoznato"
    print(f"    shape = {shape}", flush=True)

    print(f"\n[2] Ucitavam model: {MODEL_PATH}", flush=True)
    t0 = time.time()
    model_dict = joblib.load(MODEL_PATH)
    pipe       = model_dict["pipeline"]
    features   = model_dict["features"]
    cat_cols   = features["categorical"]
    num_cols   = features["numerical"]
    all_cols   = cat_cols + num_cols

    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    n_trees = len(clf.estimators_)
    print(f"    {n_trees} stabala u ensemble-u  (ucitavanje {time.time()-t0:.1f}s)",
          flush=True)

    # ------------------------------------------------------------
    # 2) Ucitaj 5 rastera (isto kao 02)
    # ------------------------------------------------------------
    print(f"\n[3] Ucitavam rastere...", flush=True)
    t0 = time.time()
    rel,   tr_ref, crs_ref, rel_nd   = read_raster(os.path.join(RASTERS_DIR, meta["files"]["rel_vis_100_250"]))
    vtt,   _,      _,       vtt_nd   = read_raster(os.path.join(RASTERS_DIR, meta["files"]["vtt_r100"]))
    dist,  _,      _,       dist_nd  = read_raster(os.path.join(RASTERS_DIR, meta["files"]["dist_rijeka_korig"]))
    strah, _,      _,       strah_nd = read_raster(os.path.join(RASTERS_DIR, meta["files"]["strahler"]))
    apsv,  _,      _,       apsv_nd  = read_raster(meta["files"]["aps_vis"])
    print(f"    ucitano u {time.time()-t0:.1f}s", flush=True)

    for name, a in [("vtt", vtt), ("dist", dist), ("strah", strah), ("apsv", apsv)]:
        if a.shape != shape:
            raise RuntimeError(f"Shape mismatch: {name} = {a.shape}, ocekivao {shape}")

    # ------------------------------------------------------------
    # 3) Valid mask + feature vectors
    # ------------------------------------------------------------
    print(f"\n[4] Build feature matrix...", flush=True)
    rel_f  = rel.astype(np.float32)
    dist_f = dist.astype(np.float32)
    apsv_f = apsv.astype(np.float32)

    def _valid(arr, nd, fallback_nd):
        check_nd = nd if nd is not None else fallback_nd
        return (arr != check_nd) & np.isfinite(arr)

    rel_valid   = _valid(rel_f,  rel_nd,  -9999.0)
    dist_valid  = _valid(dist_f, dist_nd, -9999.0)
    apsv_valid  = _valid(apsv_f, apsv_nd, -9999.0)
    strah_valid = (strah > 0)
    vtt_valid   = (vtt > 0)

    valid = rel_valid & dist_valid & apsv_valid & strah_valid & vtt_valid
    n_valid = int(valid.sum())
    print(f"    valid piksela: {n_valid:,} / {valid.size:,}", flush=True)

    rel_v   = rel_f[valid]
    dist_v  = dist_f[valid]
    apsv_v  = apsv_f[valid]
    strah_v = strah[valid].astype(np.int32)
    vtt_v   = vtt[valid].astype(np.int32)
    vtt_str = np.array([soil_types_int_to_name.get(int(v), "Nepoznato") for v in vtt_v])

    # ------------------------------------------------------------
    # 4) Predict per-tree, compute std per batch
    # ------------------------------------------------------------
    print(f"\n[5] Per-tree predikcije + std u batchovima od {BATCH_SIZE:,}...", flush=True)
    print(f"    (drzimo {n_trees} tree predictions × batch u memoriji)", flush=True)
    std_flat = np.empty(n_valid, dtype=np.float32)

    t0 = time.time()
    n_batches = (n_valid + BATCH_SIZE - 1) // BATCH_SIZE
    for b in range(n_batches):
        a = b * BATCH_SIZE
        z = min(a + BATCH_SIZE, n_valid)
        X = pd.DataFrame({
            "vtt_r100":          vtt_str[a:z],
            "rel_vis_100_250":   rel_v[a:z],
            "dist_rijeka_korig": dist_v[a:z],
            "strahler":          strah_v[a:z],
            "aps_vis":           apsv_v[a:z],
        })[all_cols]

        # Transform once kroz preprocessor (vtt OHE), pa svako stablo dobije OHE matrix
        X_trans = pre.transform(X)

        # Stack tree predictions
        tree_probs = np.empty((n_trees, z - a), dtype=np.float32)
        for ti, tree in enumerate(clf.estimators_):
            tree_probs[ti] = tree.predict_proba(X_trans)[:, 1].astype(np.float32)

        std_flat[a:z] = tree_probs.std(axis=0)

        elapsed = time.time() - t0
        eta = elapsed / (b + 1) * (n_batches - b - 1)
        print(f"    batch {b+1}/{n_batches}  ({z:,} piksela)  "
              f"elapsed {elapsed:.1f}s  ETA {eta:.1f}s",
              flush=True)

    # ------------------------------------------------------------
    # 5) Pack + save
    # ------------------------------------------------------------
    print(f"\n[6] Pack u 2D + spremam -> {OUT_PATH}", flush=True)
    std_2d = np.full(shape, NODATA_UNC, dtype=np.float32)
    std_2d[valid] = std_flat
    save_raster(std_2d, tr_ref, crs_ref, OUT_PATH, "float32", nodata=NODATA_UNC)

    # ------------------------------------------------------------
    # 6) Statistika
    # ------------------------------------------------------------
    s_valid = std_2d[valid]
    print(f"\n{'='*70}", flush=True)
    print(f"STATISTIKA STD (uncertainty):", flush=True)
    print(f"  min    = {s_valid.min():.4f}", flush=True)
    print(f"  max    = {s_valid.max():.4f}", flush=True)
    print(f"  mean   = {s_valid.mean():.4f}", flush=True)
    print(f"  median = {np.median(s_valid):.4f}", flush=True)
    print(f"  p90    = {np.percentile(s_valid, 90):.4f}", flush=True)
    print(f"  std<0.10 = {(s_valid<0.10).sum():,} ({100*(s_valid<0.10).sum()/n_valid:.1f}%)",
          flush=True)
    print(f"  std<0.15 = {(s_valid<0.15).sum():,} ({100*(s_valid<0.15).sum()/n_valid:.1f}%)",
          flush=True)
    print(f"  std>=0.20 = {(s_valid>=0.20).sum():,} ({100*(s_valid>=0.20).sum()/n_valid:.1f}%)",
          flush=True)
    print(f"{'='*70}", flush=True)
    print(f"GOTOVO -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
