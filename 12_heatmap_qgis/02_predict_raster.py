# -*- coding: utf-8 -*-
"""
02_PREDICT_RASTER — apliciraj OPTIMALNI_5 model na svaki piksel
================================================================

Cilj: za svaki piksel master grida izracunaj predict_proba neolitik prisutnosti.

Input: 12_heatmap_qgis/rasters/metadata.json (generira ga 01_prep_rasters.py)

Pipeline:
  1. Ucitaj 5 rastera u memoriju (rel_vis, vtt_r100, dist_rijeka_korig, strahler, aps_vis)
  2. Mapiraj vtt_r100 integer raster -> string imena (Cambisols, Fluvisols, ...)
  3. Filter samo valjane piksele (svi features finite, vtt != Nepoznato)
  4. Predict u batchovima od ~200k piksela
  5. Spremi probability.tif (Float32, 0..1; nodata = -9999)

Dependancije:  pip install rasterio scikit-learn joblib pandas
Pokretanje:    python 02_predict_raster.py    (lokalno, ne u QGIS-u)
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
OUT_PATH    = os.path.join(HEATMAP_DIR, "probability_neolitik.tif")

BATCH_SIZE = 200_000
NODATA_PROB = -9999.0


def read_raster(path, band_idx=1):
    """Vrati (arr, transform, crs, nodata) iz rasterskog fajla."""
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
    print("PREDICT RASTER -> probability_neolitik.tif", flush=True)
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
    print(f"    shape = {shape}, res = {meta['res_x']:.1f}m", flush=True)

    print(f"\n[2] Ucitavam model: {MODEL_PATH}", flush=True)
    t0 = time.time()
    model_dict = joblib.load(MODEL_PATH)
    pipe       = model_dict["pipeline"]
    features   = model_dict["features"]
    cat_cols   = features["categorical"]
    num_cols   = features["numerical"]
    all_cols   = cat_cols + num_cols
    print(f"    set_name = {model_dict['set_name']}  "
          f"(ucitavanje {time.time()-t0:.1f}s)", flush=True)
    print(f"    features = {all_cols}", flush=True)

    # ------------------------------------------------------------
    # 2) Ucitaj svih 5 rastera
    # ------------------------------------------------------------
    print(f"\n[3] Ucitavam rastere...", flush=True)
    t0 = time.time()
    rel,   tr_ref, crs_ref, rel_nd   = read_raster(os.path.join(RASTERS_DIR, meta["files"]["rel_vis_100_250"]))
    vtt,   _,      _,       vtt_nd   = read_raster(os.path.join(RASTERS_DIR, meta["files"]["vtt_r100"]))
    dist,  _,      _,       dist_nd  = read_raster(os.path.join(RASTERS_DIR, meta["files"]["dist_rijeka_korig"]))
    strah, _,      _,       strah_nd = read_raster(os.path.join(RASTERS_DIR, meta["files"]["strahler"]))
    apsv,  _,      _,       apsv_nd  = read_raster(meta["files"]["aps_vis"])
    print(f"    ucitano u {time.time()-t0:.1f}s", flush=True)

    for name, a, nd in [
        ("rel_vis_100_250",   rel,   rel_nd),
        ("vtt_r100",          vtt,   vtt_nd),
        ("dist_rijeka_korig", dist,  dist_nd),
        ("strahler",          strah, strah_nd),
        ("aps_vis",           apsv,  apsv_nd),
    ]:
        print(f"    {name:<20}  shape={a.shape}  dtype={a.dtype}  nodata={nd}",
              flush=True)

    for name, a in [("vtt", vtt), ("dist", dist), ("strah", strah), ("apsv", apsv)]:
        if a.shape != shape:
            raise RuntimeError(
                f"Shape mismatch: {name} = {a.shape}, ocekivao {shape}. "
                f"Pokreni 01_prep_rasters.py ponovno."
            )

    # ------------------------------------------------------------
    # 3) Build feature matrix flat-style + valid mask
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
    pct = 100 * n_valid / valid.size
    print(f"    valid piksela: {n_valid:,} / {valid.size:,}  ({pct:.1f}%)",
          flush=True)

    if n_valid == 0:
        raise RuntimeError("Nijedan piksel nije valjan — provjeri input rastere.")

    rel_v   = rel_f[valid]
    dist_v  = dist_f[valid]
    apsv_v  = apsv_f[valid]
    strah_v = strah[valid].astype(np.int32)
    vtt_v   = vtt[valid].astype(np.int32)
    vtt_str = np.array([soil_types_int_to_name.get(int(v), "Nepoznato") for v in vtt_v])

    # ------------------------------------------------------------
    # 4) Predict u batchovima
    # ------------------------------------------------------------
    print(f"\n[5] Predict u batchovima od {BATCH_SIZE:,}...", flush=True)
    proba_flat = np.empty(n_valid, dtype=np.float32)

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
        proba_flat[a:z] = pipe.predict_proba(X)[:, 1].astype(np.float32)

        elapsed = time.time() - t0
        eta = elapsed / (b + 1) * (n_batches - b - 1)
        print(f"    batch {b+1}/{n_batches}  ({z:,} piksela)  "
              f"elapsed {elapsed:.1f}s  ETA {eta:.1f}s",
              flush=True)

    # ------------------------------------------------------------
    # 5) Pack natrag u 2D + spremi
    # ------------------------------------------------------------
    print(f"\n[6] Pack u 2D + spremam -> {OUT_PATH}", flush=True)
    proba_2d = np.full(shape, NODATA_PROB, dtype=np.float32)
    proba_2d[valid] = proba_flat
    save_raster(proba_2d, tr_ref, crs_ref, OUT_PATH, "float32", nodata=NODATA_PROB)

    # ------------------------------------------------------------
    # 6) Statistika
    # ------------------------------------------------------------
    p_valid = proba_2d[valid]
    print(f"\n{'='*70}", flush=True)
    print(f"STATISTIKA PROBABILITY:", flush=True)
    print(f"  min    = {p_valid.min():.4f}", flush=True)
    print(f"  max    = {p_valid.max():.4f}", flush=True)
    print(f"  mean   = {p_valid.mean():.4f}", flush=True)
    print(f"  median = {np.median(p_valid):.4f}", flush=True)
    print(f"  p>=0.3 = {(p_valid>=0.3).sum():,} ({100*(p_valid>=0.3).sum()/n_valid:.1f}%)",
          flush=True)
    print(f"  p>=0.5 = {(p_valid>=0.5).sum():,} ({100*(p_valid>=0.5).sum()/n_valid:.1f}%)",
          flush=True)
    print(f"  p>=0.7 = {(p_valid>=0.7).sum():,} ({100*(p_valid>=0.7).sum()/n_valid:.1f}%)",
          flush=True)
    print(f"{'='*70}", flush=True)
    print(f"GOTOVO -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
