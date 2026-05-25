# -*- coding: utf-8 -*-
"""
TEKUCICE - Izvoz podataka o tekucicama (QGIS skripta)
======================================================
Izvozi 3 tipa varijabli za 8 tockastih slojeva:

  1. UDALJENOST  (dist_rijeka, vec postoji u sloju) — samo CSV export
     → dist_rijeka_/csv_output/{layer}.csv  (fid, dist_rijeka)

  2. GUSTOCA  km rijeke po km² u kruznici 1000 m i 2000 m
     → gustoca_rijeka_/csv_output/{layer}_gr_{radius}.csv  (fid, gustoca_km_km2)
     + background_gustoca.csv  (total_km, area_km2, gustoca_km_km2)

  3. STRAHLER  red najblize tekucice (1-7)
     → strahler_/csv_output/{layer}.csv  (fid, strahler)
     + background_strahler.csv  (strahler, duljina_km, postotak)

Provjeri POSTAVKE — posebno STRAHLER_FIELD i RIVERS_LAYER_NAME.

Pokretanje: Otvori QGIS → Plugins → Python Console → Run Script
"""

from qgis.core import (
    QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry,
    QgsPointXY, QgsField, QgsFields, QgsVectorFileWriter,
    QgsWkbTypes, QgsCoordinateTransform, QgsSpatialIndex,
)
from PyQt5.QtCore import QVariant
import processing
import os, csv, math
from collections import defaultdict

# ============================================================
#  POSTAVKE
# ============================================================

POINT_LAYERS = [
    "random_ceste_biased",
    "nasumicni_lokaliteti_umjetno_generirani",
    "neolitik_svi_odredeni",
    "neolitik_c_starcevacka",
    "neolitik_c_sop_kor_len",
    "kontinuirana_naselja",
    "samo_rani",
    "samo_srednji_kasni",
]

RIVERS_LAYER_NAME      = "tekucice"        # naziv sloja tekucica u QGIS projektu
RIVERS_COPY_LAYER_NAME = "tekucice_copy"   # korigirani sloj (rucno dodane rijeke)
STRAHLER_FIELD         = "STRAHLER"        # !! provjeri naziv atributa Strahlerovog reda
DIST_FIELD             = "dist_rijeka"     # atribut udaljenosti koji vec postoji u tockovnim slojevima
REFERENCE_RASTER       = "nadmorska_visina"

GUSTOCA_RADII = [1000, 2000]

BASE_DIR          = r"C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna"
DIST_DIR          = os.path.join(BASE_DIR, r"dist_rijeka_\csv_output")
DIST_COPY_DIR     = os.path.join(BASE_DIR, r"dist_rijeka_korig_\csv_output")
GUSTOCA_DIR       = os.path.join(BASE_DIR, r"gustoca_rijeka_\csv_output")
STRAHLER_DIR      = os.path.join(BASE_DIR, r"strahler_\csv_output")

# ============================================================
#  POMOCNE FUNKCIJE
# ============================================================

def get_layer(name):
    layers = QgsProject.instance().mapLayersByName(name)
    if not layers:
        raise ValueError(f"Sloj '{name}' nije pronađen. Provjeri naziv u QGIS-u.")
    return layers[0]


def get_transform(src_layer, dst_layer):
    if src_layer.crs() == dst_layer.crs():
        return None
    return QgsCoordinateTransform(src_layer.crs(), dst_layer.crs(), QgsProject.instance())


def make_buffer_layer(pt_layer, radius_m, tag):
    import tempfile, uuid
    tmp_path = os.path.join(
        tempfile.gettempdir(),
        f"qgis_tek_{uuid.uuid4().hex[:8]}_{tag}_{radius_m}m.gpkg"
    )
    fields = QgsFields()
    fields.append(QgsField("orig_fid", QVariant.Int))
    writer = QgsVectorFileWriter(
        tmp_path, "UTF-8", fields, QgsWkbTypes.Polygon, pt_layer.crs(), "GPKG"
    )
    for feat in pt_layer.getFeatures():
        geom = feat.geometry()
        if geom.isEmpty():
            continue
        circle = QgsGeometry.fromPointXY(QgsPointXY(geom.asPoint())).buffer(radius_m, 32)
        out = QgsFeature()
        out.setGeometry(circle)
        out.setAttributes([feat.id()])
        writer.addFeature(out)
    del writer
    return tmp_path


# ============================================================
#  1. UDALJENOST OD RIJEKE
# ============================================================

def export_dist_rijeka():
    os.makedirs(DIST_DIR, exist_ok=True)
    print(f"\n{'='*60}")
    print("1. UDALJENOST — izvoz atributa dist_rijeka")
    print(f"{'='*60}")

    for layer_name in POINT_LAYERS:
        try:
            pt_layer = get_layer(layer_name)
        except ValueError as e:
            print(f"  PRESKAČEM {layer_name}: {e}")
            continue

        if pt_layer.fields().indexFromName(DIST_FIELD) == -1:
            print(f"  {layer_name}: atribut '{DIST_FIELD}' ne postoji — preskačem")
            continue

        csv_path = os.path.join(DIST_DIR, f"{layer_name}.csv")
        n = 0
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["fid", DIST_FIELD])
            for feat in pt_layer.getFeatures():
                val = feat[DIST_FIELD]
                w.writerow([feat.id(), val if val is not None else ""])
                n += 1
        print(f"  {layer_name}.csv  (n={n})")


# ============================================================
#  2. GUSTOCA RIJEKA (km/km²)
# ============================================================

def calc_gustoca_layer(layer_name, rivers_layer, radius):
    pt_layer = get_layer(layer_name)

    buf_path = make_buffer_layer(pt_layer, radius, layer_name)
    buf_lyr  = QgsVectorLayer(buf_path, "buf", "ogr")
    if not buf_lyr.isValid():
        raise RuntimeError(f"Buffer layer nije validan: {buf_path}")

    result = processing.run("qgis:sumlinelengths", {
        "POLYGONS":    buf_lyr,
        "LINES":       rivers_layer,
        "LEN_FIELD":   "sum_len",
        "COUNT_FIELD": "cnt",
        "OUTPUT":      "memory:",
    })
    out_lyr = result["OUTPUT"]

    area_km2 = math.pi * (radius / 1000.0) ** 2

    fid_gustoca = {}
    for feat in out_lyr.getFeatures():
        orig_fid = feat["orig_fid"]
        sum_len  = feat["sum_len"]
        length_km = (sum_len / 1000.0) if sum_len else 0.0
        fid_gustoca[orig_fid] = round(length_km / area_km2, 6)

    try:
        os.remove(buf_path)
    except Exception:
        pass

    return fid_gustoca


def export_gustoca():
    os.makedirs(GUSTOCA_DIR, exist_ok=True)
    print(f"\n{'='*60}")
    print("2. GUSTOCA RIJEKA — km/km²")
    print(f"{'='*60}")

    rivers = get_layer(RIVERS_LAYER_NAME)

    for layer_name in POINT_LAYERS:
        print(f"\n  {'─'*50}")
        print(f"  {layer_name}")

        for radius in GUSTOCA_RADII:
            print(f"  r={radius}m ... ", end="", flush=True)
            try:
                fid_g = calc_gustoca_layer(layer_name, rivers, radius)
            except Exception as e:
                print(f"GREŠKA: {e}")
                continue

            csv_path = os.path.join(GUSTOCA_DIR, f"{layer_name}_gr_{radius}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.writer(fh)
                w.writerow(["fid", "gustoca_km_km2"])
                for fid, val in fid_g.items():
                    w.writerow([fid, val])
            print(f"OK  ({len(fid_g)} točaka)")


def export_background_gustoca(rivers_layer):
    """Ukupna duljina tekucica / povrsina studijskog podrucja (bounding box rastera)."""
    total_km = sum(f.geometry().length() for f in rivers_layer.getFeatures()) / 1000.0

    try:
        raster  = get_layer(REFERENCE_RASTER)
        ex      = raster.extent()
        area_km2 = (ex.width() * ex.height()) / 1_000_000.0
    except ValueError:
        print("  UPOZORENJE: referentni raster nije pronađen, koristim extent tekucica")
        ex       = rivers_layer.extent()
        area_km2 = (ex.width() * ex.height()) / 1_000_000.0

    gustoca = total_km / area_km2
    out_path = os.path.join(GUSTOCA_DIR, "background_gustoca.csv")
    with open(out_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["total_km", "area_km2", "gustoca_km_km2"])
        w.writerow([round(total_km, 2), round(area_km2, 2), round(gustoca, 6)])
    print(f"  background_gustoca.csv  "
          f"({total_km:.1f} km / {area_km2:.1f} km² = {gustoca:.4f} km/km²)")


# ============================================================
#  3. STRAHLER RED NAJBLIZE RIJEKE
# ============================================================

def build_river_index(rivers_layer):
    idx      = QgsSpatialIndex()
    feat_map = {}
    for feat in rivers_layer.getFeatures():
        idx.insertFeature(feat)
        feat_map[feat.id()] = feat
    return idx, feat_map


def nearest_strahler(pt_geom, river_index, feat_map, n_candidates=10):
    candidate_ids = river_index.nearestNeighbor(pt_geom.asPoint(), n_candidates)
    if not candidate_ids:
        return None
    min_dist = float("inf")
    result   = None
    for rid in candidate_ids:
        dist = pt_geom.distance(feat_map[rid].geometry())
        if dist < min_dist:
            min_dist = dist
            result   = feat_map[rid][STRAHLER_FIELD]
    return result


def export_strahler():
    os.makedirs(STRAHLER_DIR, exist_ok=True)
    print(f"\n{'='*60}")
    print("3. STRAHLER RED NAJBLIZE RIJEKE")
    print(f"{'='*60}")

    rivers = get_layer(RIVERS_LAYER_NAME)
    print(f"  Gradim prostorni indeks za {rivers.featureCount()} rijeka...")
    r_idx, r_map = build_river_index(rivers)
    print("  Indeks gotov.")

    for layer_name in POINT_LAYERS:
        print(f"\n  {'─'*50}")
        print(f"  {layer_name}")

        try:
            pt_layer = get_layer(layer_name)
        except ValueError as e:
            print(f"  PRESKAČEM: {e}")
            continue

        transform  = get_transform(pt_layer, rivers)
        fid_str    = {}
        null_count = 0

        for feat in pt_layer.getFeatures():
            geom = QgsGeometry(feat.geometry())
            if transform:
                geom.transform(transform)
            s = nearest_strahler(geom, r_idx, r_map)
            fid_str[feat.id()] = s
            if s is None:
                null_count += 1

        csv_path = os.path.join(STRAHLER_DIR, f"{layer_name}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["fid", "strahler"])
            for fid, val in fid_str.items():
                w.writerow([fid, val if val is not None else ""])

        print(f"  Točaka: {len(fid_str)}   None: {null_count}   → {layer_name}.csv")


def export_background_strahler(rivers_layer):
    """Ukupna duljina po Strahlerovom redu (km i %)."""
    lengths = defaultdict(float)
    for feat in rivers_layer.getFeatures():
        val = feat[STRAHLER_FIELD]
        if val is not None:
            lengths[int(val)] += feat.geometry().length() / 1000.0

    total    = sum(lengths.values())
    out_path = os.path.join(STRAHLER_DIR, "background_strahler.csv")
    with open(out_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["strahler", "duljina_km", "postotak"])
        for order in sorted(lengths.keys()):
            km = lengths[order]
            w.writerow([order, round(km, 3), round(km / total * 100, 4)])

    print(f"  background_strahler.csv  ({len(lengths)} redova, ukupno {total:.1f} km)")
    for o in sorted(lengths.keys()):
        print(f"    Strahler {o}: {lengths[o]:.1f} km  ({lengths[o]/total*100:.1f}%)")


# ============================================================
#  UDALJENOST — KORIGIRANI SLOJ (tekucice_copy)
# ============================================================

def export_dist_copy():
    """
    Izracunaj udaljenost od najblize rijeke u korigiranom sloju tekucice_copy.
    Za razliku od export_dist_rijeka() (koja cita gotov atribut),
    ova funkcija gradi prostorni indeks i mjeri stvarnu udaljenost za svaku tocku.
    Izvoz: DIST_COPY_DIR/{layer}.csv  (fid, dist_rijeka_korig)
    """
    os.makedirs(DIST_COPY_DIR, exist_ok=True)
    print(f"\n{'='*60}")
    print("UDALJENOST — tekucice_copy (korigirani sloj)")
    print(f"{'='*60}")

    try:
        rivers = get_layer(RIVERS_COPY_LAYER_NAME)
    except ValueError as e:
        print(f"  GREŠKA: {e}")
        return

    print(f"  Gradim prostorni indeks za {rivers.featureCount()} rijeka (copy)...")
    r_idx, r_map = build_river_index(rivers)
    print("  Indeks gotov.")

    for layer_name in POINT_LAYERS:
        try:
            pt_layer = get_layer(layer_name)
        except ValueError as e:
            print(f"  PRESKAČEM {layer_name}: {e}")
            continue

        transform  = get_transform(pt_layer, rivers)
        fid_dist   = {}
        null_count = 0

        for feat in pt_layer.getFeatures():
            geom = QgsGeometry(feat.geometry())
            if transform:
                geom.transform(transform)

            candidate_ids = r_idx.nearestNeighbor(geom.asPoint(), 10)
            if not candidate_ids:
                fid_dist[feat.id()] = None
                null_count += 1
                continue

            min_dist = float("inf")
            for rid in candidate_ids:
                dist = geom.distance(r_map[rid].geometry())
                if dist < min_dist:
                    min_dist = dist

            fid_dist[feat.id()] = round(min_dist, 2)

        csv_path = os.path.join(DIST_COPY_DIR, f"{layer_name}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["fid", "dist_rijeka_korig"])
            for fid, val in fid_dist.items():
                w.writerow([fid, val if val is not None else ""])

        print(f"  {layer_name}.csv  (n={len(fid_dist)}, null={null_count})")


# ============================================================
#  POKRENI
# ============================================================

def run_all():
    print("=" * 60)
    print("TEKUCICE - početak analize")
    print("=" * 60)

    # udaljenost — originalna mreža (dist_rijeka)  →  dist_rijeka_/
    export_dist_rijeka()

    # udaljenost — korigirana mreža (dist_rijeka_korig)  →  dist_rijeka_korig_/
    export_dist_copy()

    # gustoca i strahler — uvijek po originalnoj mreži tekucice
    export_gustoca()
    rivers = get_layer(RIVERS_LAYER_NAME)
    export_background_gustoca(rivers)

    export_strahler()
    rivers = get_layer(RIVERS_LAYER_NAME)
    export_background_strahler(rivers)

    print(f"\n{'='*60}")
    print(f"GOTOVO!  Izlazi: {DIST_DIR}")
    print(f"         {DIST_COPY_DIR}")
    print(f"         {GUSTOCA_DIR}")
    print(f"         {STRAHLER_DIR}")
    print(f"{'='*60}")


run_all()
