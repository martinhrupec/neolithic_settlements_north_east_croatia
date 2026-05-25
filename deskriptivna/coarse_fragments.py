# -*- coding: utf-8 -*-
"""
COARSE_FRAGMENTS - Ocitavanje rasterske vrijednosti na tocki
=============================================================
Za svaki od 8 tockastih slojeva:
  1. Uzorkuje vrijednost rasterskog sloja "coarse_fragments" na lokaciji tocke
  2. Dodaje atribut "c_frag" u svaki sloj u QGIS projektu
  3. Izvozi 8 CSV datoteka  →  {OUTPUT_DIR}/{naziv_sloja}.csv

Svaki CSV: fid, c_frag
"""

from qgis.core import (
    QgsProject,
    QgsField,
    QgsCoordinateTransform,
)
from PyQt5.QtCore import QVariant
import processing
import os
import csv

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

COARSE_RASTER_LAYER = "coarse_fragments"
ATTR_NAME           = "c_frag"          # 6 znakova - sigurno za shapefile i GPKG

OUTPUT_DIR = r"C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna\coarse_fragments_\csv_output"

# ============================================================
#  POMOCNE FUNKCIJE
# ============================================================

def get_layer(name):
    layers = QgsProject.instance().mapLayersByName(name)
    if not layers:
        raise ValueError(f"Sloj '{name}' nije pronađen. Provjeri naziv u QGIS-u.")
    return layers[0]


def sample_raster(raster_layer, feat, transform=None):
    """
    Uzorkuje vrijednost rastera na lokaciji tocke.
    Vraca float ili None ako je tocka izvan rastera / NoData.
    """
    geom = feat.geometry()
    pt   = geom.asPoint()
    if transform:
        pt = transform.transform(pt)
    val, ok = raster_layer.dataProvider().sample(pt, 1)
    return val if ok else None


# ============================================================
#  GLAVNA ANALIZA
# ============================================================

def run_coarse_fragments():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        raster_layer = get_layer(COARSE_RASTER_LAYER)
    except ValueError as e:
        print(f"GRESKA: {e}")
        return

    print("=" * 60)
    print("COARSE FRAGMENTS - poceta analize")
    print("=" * 60)
    print(f"Raster: {raster_layer.name()}   CRS: {raster_layer.crs().authid()}")
    print(f"Izlaz CSV: {OUTPUT_DIR}")
    print()

    for layer_name in POINT_LAYERS:
        print(f"{'─'*55}")
        print(f"   {layer_name}")

        try:
            pt_layer = get_layer(layer_name)
        except ValueError as e:
            print(f"   PRESKACM: {e}")
            continue

        # CRS transformacija ako se razlikuju
        transform = None
        if pt_layer.crs() != raster_layer.crs():
            print("   (CRS transformacija aktiva)")
            transform = QgsCoordinateTransform(
                pt_layer.crs(),
                raster_layer.crs(),
                QgsProject.instance(),
            )

        # Uzorkovanje
        fid_val = {}
        null_count = 0
        for feat in pt_layer.getFeatures():
            val = sample_raster(raster_layer, feat, transform)
            fid_val[feat.id()] = val
            if val is None:
                null_count += 1

        n = len(fid_val)
        print(f"   Tocaka: {n}   NoData/izvan rastera: {null_count}")

        # Dodaj atribut u sloj
        pt_layer.startEditing()
        if pt_layer.fields().indexFromName(ATTR_NAME) == -1:
            pt_layer.addAttribute(QgsField(ATTR_NAME, QVariant.Double))
        pt_layer.updateFields()

        idx = pt_layer.fields().indexFromName(ATTR_NAME)
        for feat in pt_layer.getFeatures():
            val = fid_val.get(feat.id())
            pt_layer.changeAttributeValue(feat.id(), idx, val)

        pt_layer.commitChanges()
        print(f"   Atribut '{ATTR_NAME}' zapisan OK")

        # Izvoz CSV
        csv_path = os.path.join(OUTPUT_DIR, f"{layer_name}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh)
            writer.writerow(["fid", ATTR_NAME])
            for fid, val in fid_val.items():
                writer.writerow([fid, val if val is not None else ""])

        print(f"   CSV: {layer_name}.csv")

    print()
    print("=" * 60)
    print("GOTOVO!")
    print("=" * 60)


# ============================================================
#  POKRENI
# ============================================================
run_coarse_fragments()
