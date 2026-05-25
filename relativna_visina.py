"""
RELATIVNA VISINA NASELJA - QGIS Python skripta
================================================
Za svaku točku računa:
  - prosječnu visinu unutarnjeg diska (polumjer x)
  - prosječnu visinu vanjskog prstena/annulusa (polumjer y MINUS polumjer x)
  - rezultat: inner_avg - outer_ring_avg  →  "relativna prominencija"

Pokretanje: Otvori QGIS → Plugins → Python Console → Run Script
"""

from qgis.core import (
    QgsProject,
    QgsVectorLayer,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsField,
    QgsFields,
    QgsVectorFileWriter,
    QgsWkbTypes,
)
from qgis.analysis import QgsZonalStatistics
from PyQt5.QtCore import QVariant
import os
import csv

# ============================================================
#  POSTAVKE - OVDJE MIJENJAJ PARAMETRE
# ============================================================

# Slojevi za koje TREBA pokrenuti analizu
# (neolitik_svi_odredeni je već obrađen — samo se izvozi)
ANALYSE_LAYERS = [
    "random_ceste_biased",
    "nasumicni_lokaliteti_umjetno_generirani",
    "neolitik_c_starcevacka",
    "neolitik_c_sop_kor_len",
    "kontinuirana_naselja",
    "samo_rani",
    "samo_srednji_kasni",
]

# Svih 8 slojeva — za CSV izvoz
ALL_LAYERS = [
    "random_ceste_biased",
    "nasumicni_lokaliteti_umjetno_generirani",
    "neolitik_svi_odredeni",
    "neolitik_c_starcevacka",
    "neolitik_c_sop_kor_len",
    "kontinuirana_naselja",
    "samo_rani",
    "samo_srednji_kasni",
]

RASTER_LAYER_NAME = "nadmorska_visina"

RADIUS_COMBINATIONS = [
    (100,  250),
    (100,  500),
    (100, 1000),
    (200,  500),
    (200, 1000),
    (500, 1000),
]

OUTPUT_CSV_DIR = r"C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna\relativna_visina_\csv_output"

# ============================================================
#  POMOĆNE FUNKCIJE
# ============================================================

def get_layer(name):
    layers = QgsProject.instance().mapLayersByName(name)
    if not layers:
        raise ValueError(f"Sloj '{name}' nije pronađen. Provjeri naziv u QGIS-u.")
    return layers[0]


def make_circle_layer(point_layer, radius_m, tag):
    import tempfile, uuid
    tmp_path = os.path.join(
        tempfile.gettempdir(),
        f"qgis_rv_{uuid.uuid4().hex[:8]}_{tag}_{radius_m}m.gpkg"
    )
    fields = QgsFields()
    fields.append(QgsField("orig_fid", QVariant.Int))
    writer = QgsVectorFileWriter(
        tmp_path, "UTF-8", fields, QgsWkbTypes.Polygon, point_layer.crs(), "GPKG"
    )
    for feat in point_layer.getFeatures():
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


def make_annulus_layer(point_layer, inner_r, outer_r, tag):
    import tempfile, uuid
    tmp_path = os.path.join(
        tempfile.gettempdir(),
        f"qgis_rv_{uuid.uuid4().hex[:8]}_{tag}_{inner_r}_{outer_r}m.gpkg"
    )
    fields = QgsFields()
    fields.append(QgsField("orig_fid", QVariant.Int))
    writer = QgsVectorFileWriter(
        tmp_path, "UTF-8", fields, QgsWkbTypes.Polygon, point_layer.crs(), "GPKG"
    )
    for feat in point_layer.getFeatures():
        geom = feat.geometry()
        if geom.isEmpty():
            continue
        pt = QgsGeometry.fromPointXY(QgsPointXY(geom.asPoint()))
        annulus = pt.buffer(outer_r, 32).difference(pt.buffer(inner_r, 32))
        out = QgsFeature()
        out.setGeometry(annulus)
        out.setAttributes([feat.id()])
        writer.addFeature(out)
    del writer
    return tmp_path


def zonal_mean(tmp_path, raster_layer, prefix):
    poly = QgsVectorLayer(tmp_path, "tmp", "ogr")
    if not poly.isValid():
        raise RuntimeError(f"Ne mogu učitati: {tmp_path}")
    QgsZonalStatistics(poly, raster_layer, prefix, 1, QgsZonalStatistics.Mean).calculateStatistics(None)
    poly.reload()
    mean_field = next(
        (f.name() for f in poly.fields()
         if f.name().lower().startswith(prefix.lower()) and "mean" in f.name().lower()),
        None
    )
    if mean_field is None:
        raise RuntimeError(f"Mean polje s prefiksom '{prefix}' nije pronađeno. Polja: {[f.name() for f in poly.fields()]}")
    return {feat["orig_fid"]: (float(feat[mean_field]) if feat[mean_field] is not None else None)
            for feat in poly.getFeatures()}


# ============================================================
#  ANALIZA JEDNOG SLOJA
# ============================================================

def analyse_layer(layer_name, raster_layer):
    print(f"\n{'─'*55}")
    print(f"  {layer_name}")
    pt_layer = get_layer(layer_name)
    print(f"  Točaka: {pt_layer.featureCount()}")

    fid_list = [f.id() for f in pt_layer.getFeatures()]
    results  = {fid: {} for fid in fid_list}

    for (x, y) in RADIUS_COMBINATIONS:
        print(f"  inner={x}m  outer={y}m ... ", end="", flush=True)

        inner_path   = make_circle_layer(pt_layer, x,    f"i{x}")
        annulus_path = make_annulus_layer(pt_layer, x, y, f"a{x}{y}")

        inner_means = zonal_mean(inner_path,   raster_layer, f"i{x}_")
        outer_means = zonal_mean(annulus_path, raster_layer, f"o{x}{y}_")

        for fid in fid_list:
            iv = inner_means.get(fid)
            ov = outer_means.get(fid)
            results[fid][f"rel_{x}_{y}m"]   = round(iv - ov, 3) if (iv is not None and ov is not None) else None
            results[fid][f"inner_{x}m"]      = round(iv, 3) if iv is not None else None
            results[fid][f"outer_{x}_{y}m"]  = round(ov, 3) if ov is not None else None

        for p in [inner_path, annulus_path]:
            try: os.remove(p)
            except: pass

        print("OK")

    # Dodaj atribute u sloj
    pt_layer.startEditing()
    for (x, y) in RADIUS_COMBINATIONS:
        for col in [f"rel_{x}_{y}m", f"inner_{x}m", f"outer_{x}_{y}m"]:
            if pt_layer.fields().indexFromName(col) == -1:
                pt_layer.addAttribute(QgsField(col, QVariant.Double))
    pt_layer.updateFields()

    for feat in pt_layer.getFeatures():
        fid = feat.id()
        for col, val in results[fid].items():
            idx = pt_layer.fields().indexFromName(col)
            if idx != -1:
                pt_layer.changeAttributeValue(fid, idx, val)

    pt_layer.commitChanges()
    print(f"  Atributi zapisani.")


# ============================================================
#  CSV IZVOZ ZA SVIH 8 SLOJEVA
# ============================================================

def export_csvs():
    os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
    total = 0

    for layer_name in ALL_LAYERS:
        try:
            pt_layer = get_layer(layer_name)
        except ValueError as e:
            print(f"PRESKAČEM izvoz {layer_name}: {e}")
            continue

        for (x, y) in RADIUS_COMBINATIONS:
            col      = f"rel_{x}_{y}m"
            col_idx  = pt_layer.fields().indexFromName(col)
            if col_idx == -1:
                print(f"  {layer_name}: atribut '{col}' nedostaje — preskačem")
                continue

            csv_path = os.path.join(OUTPUT_CSV_DIR, f"{layer_name}_rv_{x}_{y}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
                writer = csv.writer(fh)
                writer.writerow(["fid", col])
                for feat in pt_layer.getFeatures():
                    val = feat[col]
                    writer.writerow([feat.id(), val if val is not None else ""])

            total += 1
            print(f"  CSV: {layer_name}_rv_{x}_{y}.csv")

    print(f"\nIzvezeno {total} CSV datoteka → {OUTPUT_CSV_DIR}")


# ============================================================
#  POKRENI
# ============================================================

def run_all():
    print("=" * 60)
    print("RELATIVNA VISINA - početak analize")
    print("=" * 60)

    raster_layer = get_layer(RASTER_LAYER_NAME)
    print(f"Raster: {raster_layer.name()}   CRS: {raster_layer.crs().authid()}")
    print(f"Analiziram {len(ANALYSE_LAYERS)} slojeva, izvozim za {len(ALL_LAYERS)}.")

    for layer_name in ANALYSE_LAYERS:
        try:
            analyse_layer(layer_name, raster_layer)
        except Exception as e:
            print(f"\nGRESKA za {layer_name}: {e}")

    print(f"\n{'='*60}")
    print("Izvoz CSV-ova za svih 8 slojeva...")
    print(f"{'='*60}")
    export_csvs()

    print(f"\n{'='*60}")
    print("GOTOVO!")
    print(f"{'='*60}")


run_all()
