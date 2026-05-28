# -*- coding: utf-8 -*-
"""
01_PREP_RASTERS — generiranje 4 deriviranih rastera za heatmap
================================================================

Cilj: za svaki piksel master grida (= nadmorska_visina raster) izracunaj
cetiri feature varijable koje koristi RF model OPTIMALNI_5.

Slijedi LOGIKU iz postojecih skripti:
  - rel_vis_100_250   iz relativna_visina.py
                      (inner mean krug 100 m  MINUS  outer mean prsten 100-250 m)
  - vtt_r100          iz vecinski_tip_tla.py
                      (modus tipa tla u krugu 100 m)
  - dist_rijeka_korig iz tekucice.py
                      (nearest distance to tekucice_copy)
  - strahler          iz tekucice.py
                      (STRAHLER atribut najblize tekucice)

Aps_vis = sam nadmorska_visina raster (NE generiramo poseban, koristimo izravno).

Implementacija: GDAL + numpy + scipy.signal.fftconvolve.
Za 2M piksela trajanje je 1-3 minute, ovisno o rezoluciji.

Output u OUT_DIR (svi rasteri su LZW-tiled GeoTIFF, isti grid kao DEM):
  - rel_vis_100_250.tif      (Float32, nodata = -9999)
  - vtt_r100.tif             (Int16, vrijednosti su raster kodovi iz SOIL_TYPES)
  - dist_rijeka_korig.tif    (Float32, distanca u metrima)
  - strahler.tif             (Int16, 1-7)

Pokretanje: Otvori QGIS s loadanim slojevima -> Python Console -> Open Script.
"""

import os
import json
import tempfile
import uuid
import numpy as np
from scipy import signal, ndimage
from osgeo import gdal, ogr

from qgis.core import QgsProject


def _tmp_tif(tag):
    """Generira unique temp .tif path na disku (umjesto /vsimem/ koji moze biti problematican u QGIS-u)."""
    return os.path.join(tempfile.gettempdir(),
                        f"qgis_prep_{uuid.uuid4().hex[:10]}_{tag}.tif")


# ============================================================
#  POSTAVKE
# ============================================================

OUT_DIR = r"C:\Users\Martin\Desktop\skripte_za_diplomski\12_heatmap_qgis\rasters"

DEM_LAYER    = "nadmorska_visina"
SOIL_LAYER   = "tipovi_tla"
RIVERS_LAYER = "tekucice_copy"

INNER_R = 100    # m  (krug)
OUTER_R = 250    # m  (vanjski radius prstena)

STRAHLER_FIELD = "STRAHLER"

# Mapa raster value -> soil name  (mora odgovarati vecinski_tip_tla.py)
SOIL_TYPES = {
    2:  "Alisols",
    4:  "Arenosols",
    5:  "Calcisols",
    6:  "Cambisols",
    7:  "Chernozems",
    11: "Fluvisols",
    12: "Gleysols",
    16: "Leptosols",
    18: "Luvisols",
    20: "Pheozems",
    24: "Regosols",
    29: "Vertisols",
}


# ============================================================
#  POMOCNE FUNKCIJE
# ============================================================

def get_layer(name):
    layers = QgsProject.instance().mapLayersByName(name)
    if not layers:
        raise ValueError(f"Sloj '{name}' nije pronaden u projektu.")
    return layers[0]


def open_raster(layer, band_idx=1):
    """Otvori QGIS rasterski sloj i vrati (arr, geotransform, projection, nodata)."""
    path = layer.source()
    ds = gdal.Open(path)
    if ds is None:
        raise RuntimeError(f"Ne mogu otvoriti raster: {path}")
    band = ds.GetRasterBand(band_idx)
    arr  = band.ReadAsArray()
    gt   = ds.GetGeoTransform()
    proj = ds.GetProjection()
    nd   = band.GetNoDataValue()
    ds   = None
    return arr, gt, proj, nd


def split_vector_source(source_str):
    """
    QGIS vector source moze imati oblik:
        /path/file.gpkg|layername=ime
        /path/file.shp
    Vrati (path, layer_name_or_None).
    """
    if "|" in source_str:
        parts = source_str.split("|")
        path = parts[0]
        layer_name = None
        for p in parts[1:]:
            if p.startswith("layername="):
                layer_name = p[len("layername="):]
                break
            if p.startswith("layerid="):
                layer_name = None  # cemo koristiti indeks
                break
        return path, layer_name
    return source_str, None


def open_ogr_layer(layer):
    """Otvori QGIS vektorski sloj kroz OGR — vraca (ogr_ds, ogr_lyr)."""
    src = layer.source()
    path, lname = split_vector_source(src)
    ds = ogr.Open(path)
    if ds is None:
        raise RuntimeError(f"OGR ne moze otvoriti: {path}")
    if lname:
        lyr = ds.GetLayerByName(lname)
        if lyr is None:
            raise RuntimeError(f"OGR layer '{lname}' nije u {path}. "
                               f"Dostupni: {[ds.GetLayer(i).GetName() for i in range(ds.GetLayerCount())]}")
    else:
        lyr = ds.GetLayer(0)
    return ds, lyr


def save_raster(arr, gt, proj, out_path, dtype, nodata=None):
    rows, cols = arr.shape
    drv = gdal.GetDriverByName("GTiff")
    ds  = drv.Create(out_path, cols, rows, 1, dtype,
                     options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"])
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    band = ds.GetRasterBand(1)
    band.WriteArray(arr)
    if nodata is not None:
        band.SetNoDataValue(float(nodata))
    band.FlushCache()
    ds = None


def circular_kernel(radius_pix):
    """Boolean kernel za krug radijusa `radius_pix` piksela (centered)."""
    r = max(int(radius_pix), 1)
    y, x = np.ogrid[-r:r+1, -r:r+1]
    return (x*x + y*y) <= r*r


def annulus_kernel(inner_pix, outer_pix):
    """Boolean kernel za prsten: inner_pix < d <= outer_pix."""
    r = max(int(outer_pix), 1)
    y, x = np.ogrid[-r:r+1, -r:r+1]
    d2 = x*x + y*y
    return (d2 <= outer_pix*outer_pix) & (d2 > inner_pix*inner_pix)


def focal_mean(arr, kernel, valid_mask):
    """
    Mean unutar `kernel`, ignorirajuci nodata.
    Vraca NaN za piksele gdje nijedan susjed nije valjan.
    """
    arr_f    = arr.astype(np.float32)
    vmask_f  = valid_mask.astype(np.float32)
    arr0     = arr_f * vmask_f
    k_f      = kernel.astype(np.float32)

    val_count = signal.fftconvolve(vmask_f, k_f, mode="same")
    val_sum   = signal.fftconvolve(arr0,    k_f, mode="same")

    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(val_count > 0.5, val_sum / val_count, np.nan)
    return out.astype(np.float32)


def focal_mode_categorical(cat_arr, valid_mask, kernel, categories):
    """
    Za svaki piksel: najcesca kategorija u kernelu.
    Vraca int array; 0 = "Nepoznato" (nema valjanih susjeda).
    """
    rows, cols = cat_arr.shape
    k_f = kernel.astype(np.float32)
    best_count = np.zeros((rows, cols), dtype=np.float32)
    best_cat   = np.zeros((rows, cols), dtype=np.int32)
    total      = np.zeros((rows, cols), dtype=np.float32)

    for c in categories:
        m = ((cat_arr == c) & valid_mask).astype(np.float32)
        cnt = signal.fftconvolve(m, k_f, mode="same")
        total += cnt
        win = cnt > best_count
        best_count = np.where(win, cnt, best_count)
        best_cat   = np.where(win, c,   best_cat)

    best_cat[total <= 0.5] = 0
    return best_cat


def warp_to_grid(src_layer, ref_gt, ref_shape, ref_proj, resample_alg="near"):
    """Resamplira `src_layer` na master grid (gt+shape+proj).  Vraca numpy array."""
    rows, cols = ref_shape
    xmin = ref_gt[0]
    ymax = ref_gt[3]
    xres = ref_gt[1]
    yres = ref_gt[5]                       # uglavnom negativan
    xmax = xmin + cols * xres
    ymin = ymax + rows * yres

    tmp = _tmp_tif("warped")
    # gdal.Warp prima raster path; za rastere QGIS source string je obicno cisti path
    src_path = src_layer.source()
    if "|" in src_path:
        src_path = src_path.split("|", 1)[0]
    gdal.Warp(
        tmp, src_path,
        format="GTiff",
        xRes=abs(xres), yRes=abs(yres),
        outputBounds=(xmin, ymin, xmax, ymax),
        dstSRS=ref_proj,
        resampleAlg=resample_alg,
    )
    ds = gdal.Open(tmp)
    arr = ds.GetRasterBand(1).ReadAsArray()
    nd  = ds.GetRasterBand(1).GetNoDataValue()
    ds  = None
    try:
        os.remove(tmp)
    except OSError:
        pass
    return arr, nd


# ============================================================
#  GLAVNA LOGIKA
# ============================================================

def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("PREP RASTERS - generiranje 4 deriviranih rastera za heatmap")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1) Ucitaj DEM (master grid)
    # ------------------------------------------------------------
    print(f"\n[1] Ucitavam {DEM_LAYER}...")
    dem_layer = get_layer(DEM_LAYER)
    dem, gt, proj, dem_nodata = open_raster(dem_layer)
    rows, cols = dem.shape
    res_x = abs(gt[1])
    res_y = abs(gt[5])
    print(f"    shape = {rows} x {cols}   res = {res_x:.1f} x {res_y:.1f} m")
    print(f"    extent x = {gt[0]:.0f}..{gt[0]+cols*res_x:.0f}")
    print(f"    extent y = {gt[3]-rows*res_y:.0f}..{gt[3]:.0f}")
    print(f"    nodata = {dem_nodata}")

    if dem_nodata is not None:
        dem_valid = (dem != dem_nodata) & np.isfinite(dem)
    else:
        dem_valid = np.isfinite(dem)
    print(f"    valid piksela: {int(dem_valid.sum()):,} / {dem.size:,}")

    # Kerneli za 100/250 m
    inner_pix = int(round(INNER_R / res_x))
    outer_pix = int(round(OUTER_R / res_x))
    if inner_pix < 1:
        inner_pix = 1
    if outer_pix <= inner_pix:
        outer_pix = inner_pix + 1
    k_inner   = circular_kernel(inner_pix)
    k_annulus = annulus_kernel(inner_pix, outer_pix)
    print(f"    kernel inner = {k_inner.shape} (radius {inner_pix} px)")
    print(f"    kernel annulus = {k_annulus.shape} (inner {inner_pix} px, outer {outer_pix} px)")
    print(f"    inner krug ima {int(k_inner.sum())} piksela, prsten ima {int(k_annulus.sum())} piksela")

    # ------------------------------------------------------------
    # 2) REL_VIS_100_250
    # ------------------------------------------------------------
    print(f"\n[2] Racunam rel_vis_{INNER_R}_{OUTER_R}...")
    print("    focal mean inner...")
    mean_inner = focal_mean(dem, k_inner,   dem_valid)
    print("    focal mean annulus...")
    mean_outer = focal_mean(dem, k_annulus, dem_valid)

    rel_vis = (mean_inner - mean_outer).astype(np.float32)
    rel_vis_nd = np.float32(-9999.0)
    rel_vis = np.where(dem_valid & np.isfinite(rel_vis), rel_vis, rel_vis_nd)

    out_path = os.path.join(OUT_DIR, "rel_vis_100_250.tif")
    save_raster(rel_vis, gt, proj, out_path, gdal.GDT_Float32, nodata=rel_vis_nd)
    print(f"    -> {out_path}")

    del mean_inner, mean_outer, rel_vis

    # ------------------------------------------------------------
    # 3) VTT_R100  (modus tipa tla u krugu 100 m)
    # ------------------------------------------------------------
    print(f"\n[3] Racunam vtt_r{INNER_R}...")
    soil_layer = get_layer(SOIL_LAYER)
    soil_arr, soil_nd_orig = warp_to_grid(soil_layer, gt, dem.shape, proj, "near")
    print(f"    soil shape = {soil_arr.shape}   nodata = {soil_nd_orig}")

    if soil_nd_orig is not None:
        soil_valid = (soil_arr != soil_nd_orig) & dem_valid
    else:
        soil_valid = dem_valid

    cats = sorted(SOIL_TYPES.keys())
    print(f"    kategorije ({len(cats)}): {cats}")
    print("    focal mode (12 konvolucija)...")
    vtt = focal_mode_categorical(soil_arr.astype(np.int32), soil_valid, k_inner, cats)
    vtt[~dem_valid] = 0

    out_path = os.path.join(OUT_DIR, "vtt_r100.tif")
    save_raster(vtt.astype(np.int16), gt, proj, out_path, gdal.GDT_Int16, nodata=0)
    print(f"    -> {out_path}")
    print(f"       mapping: 0 = Nepoznato, ostale vrijednosti vidi SOIL_TYPES dict")

    del soil_arr, vtt

    # ------------------------------------------------------------
    # 4) DIST_RIJEKA_KORIG (proximity)
    # ------------------------------------------------------------
    print(f"\n[4] Racunam dist_rijeka_korig (proximity to {RIVERS_LAYER})...")
    rivers_layer = get_layer(RIVERS_LAYER)

    # Rasterize linije u binarni raster na master gridu
    drv = gdal.GetDriverByName("GTiff")
    burn_path = _tmp_tif("rivers_burn")
    burn_ds = drv.Create(burn_path, cols, rows, 1, gdal.GDT_Byte)
    burn_ds.SetGeoTransform(gt)
    burn_ds.SetProjection(proj)
    burn_ds.GetRasterBand(1).Fill(0)

    ogr_ds, ogr_lyr = open_ogr_layer(rivers_layer)
    gdal.RasterizeLayer(burn_ds, [1], ogr_lyr, burn_values=[1],
                        options=["ALL_TOUCHED=TRUE"])
    burn_ds.FlushCache()
    burn_ds = None
    ogr_ds = None

    prox_path = os.path.join(OUT_DIR, "dist_rijeka_korig.tif")
    in_ds  = gdal.Open(burn_path)
    out_ds = drv.Create(prox_path, cols, rows, 1, gdal.GDT_Float32,
                        options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"])
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    gdal.ComputeProximity(in_ds.GetRasterBand(1), out_ds.GetRasterBand(1),
                          options=["DISTUNITS=GEO", "VALUES=1"])
    in_ds  = None
    out_ds = None
    try:
        os.remove(burn_path)
    except OSError:
        pass
    print(f"    -> {prox_path}")

    # ------------------------------------------------------------
    # 5) STRAHLER raster (nearest strahler od najblize rijeke)
    # ------------------------------------------------------------
    print(f"\n[5] Racunam strahler raster (atribut '{STRAHLER_FIELD}')...")

    str_burn_path = _tmp_tif("strahler_burn")
    sds = drv.Create(str_burn_path, cols, rows, 1, gdal.GDT_Int16)
    sds.SetGeoTransform(gt)
    sds.SetProjection(proj)
    sds.GetRasterBand(1).Fill(0)

    ogr_ds, ogr_lyr = open_ogr_layer(rivers_layer)
    # Provjera da atribut postoji
    field_names = [ogr_lyr.GetLayerDefn().GetFieldDefn(i).GetName()
                   for i in range(ogr_lyr.GetLayerDefn().GetFieldCount())]
    if STRAHLER_FIELD not in field_names:
        print(f"    UPOZORENJE: atribut '{STRAHLER_FIELD}' nije pronaden u {RIVERS_LAYER}.")
        print(f"               dostupni atributi: {field_names}")
        print(f"    -> svi pikseli rijeka dobit ce strahler = 1 (fallback).")
        gdal.RasterizeLayer(sds, [1], ogr_lyr, burn_values=[1],
                            options=["ALL_TOUCHED=TRUE"])
    else:
        gdal.RasterizeLayer(sds, [1], ogr_lyr,
                            options=[f"ATTRIBUTE={STRAHLER_FIELD}",
                                     "ALL_TOUCHED=TRUE"])
    sds.FlushCache()
    s_arr = sds.GetRasterBand(1).ReadAsArray()
    sds   = None
    ogr_ds = None
    try:
        os.remove(str_burn_path)
    except OSError:
        pass

    river_mask = s_arr > 0
    print(f"    piksela s rijekom: {int(river_mask.sum()):,}")
    if river_mask.sum() == 0:
        raise RuntimeError("Nijedan piksel nije pogoden rijekama. Provjeri rasterize.")

    print("    distance_transform_edt + nearest index...")
    _, (yi, xi) = ndimage.distance_transform_edt(~river_mask, return_indices=True)
    strahler_nearest = s_arr[yi, xi].astype(np.int16)
    strahler_nearest[~dem_valid] = 0

    out_path = os.path.join(OUT_DIR, "strahler.tif")
    save_raster(strahler_nearest, gt, proj, out_path, gdal.GDT_Int16, nodata=0)
    print(f"    -> {out_path}")

    # ------------------------------------------------------------
    # 6) Metadata za sljedecu skriptu
    # ------------------------------------------------------------
    meta = {
        "dem_path":     dem_layer.source(),
        "out_dir":      OUT_DIR,
        "shape":        [rows, cols],
        "geotransform": list(gt),
        "projection":   proj,
        "res_x":        res_x,
        "res_y":        res_y,
        "soil_types":   {int(k): v for k, v in SOIL_TYPES.items()},
        "files": {
            "rel_vis_100_250":   "rel_vis_100_250.tif",
            "vtt_r100":          "vtt_r100.tif",
            "dist_rijeka_korig": "dist_rijeka_korig.tif",
            "strahler":          "strahler.tif",
            "aps_vis":           dem_layer.source(),
        },
    }
    meta_path = os.path.join(OUT_DIR, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(f"\n[6] Metadata -> {meta_path}")

    # ------------------------------------------------------------
    # GOTOVO
    # ------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"GOTOVO! Sve spremljeno u: {OUT_DIR}")
    print(f"  rel_vis_100_250.tif    (Float32, nodata=-9999)")
    print(f"  vtt_r100.tif           (Int16, mapping = SOIL_TYPES; 0=Nepoznato)")
    print(f"  dist_rijeka_korig.tif  (Float32, metri)")
    print(f"  strahler.tif           (Int16, 1-7)")
    print(f"  metadata.json          (sve sto 02_predict treba)")
    print(f"  + aps_vis = sam {DEM_LAYER} (path u metadati)")
    print(f"{'='*70}")


run()
