# -*- coding: utf-8 -*-
"""
TRI - Terrain Ruggedness Index (QGIS skripta)
==============================================
Za svaki od 8 tockastih slojeva:
  1. Uzorkuje vrijednost rastera "TRI" na lokaciji tocke
  2. Dodaje atribut "tri" u sloj
  3. Izvozi CSV  →  {OUTPUT_DIR}/{naziv_sloja}.csv  (fid, tri)

Uz to izvozi pozadinski histogram:
  4. background_tri.csv  →  tri_value, n_piksela, postotak
     (histogram cijelog rastera, N_HIST_BINS stupaca)

Pokretanje: Otvori QGIS → Plugins → Python Console → Run Script
"""

from qgis.core import (
    QgsProject,
    QgsField,
    QgsCoordinateTransform,
    QgsRasterBandStats,
)
from PyQt5.QtCore import QVariant
import os, csv

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

RASTER_LAYER_NAME = "TRI"
ATTR_NAME         = "tri"
N_HIST_BINS       = 200

OUTPUT_DIR = r"C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna\tri_\csv_output"

# ============================================================
#  POMOCNE FUNKCIJE
# ============================================================

def get_layer(name):
    layers = QgsProject.instance().mapLayersByName(name)
    if not layers:
        raise ValueError(f"Sloj '{name}' nije pronađen. Provjeri naziv u QGIS-u.")
    return layers[0]


def sample_raster(raster_layer, feat, transform=None):
    geom = feat.geometry()
    pt   = geom.asPoint()
    if transform:
        pt = transform.transform(pt)
    val, ok = raster_layer.dataProvider().sample(pt, 1)
    return val if ok else None


# ============================================================
#  UZORKOVANJE TOCAKA
# ============================================================

def sample_layers(raster_layer):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for layer_name in POINT_LAYERS:
        print(f"{'─'*55}")
        print(f"   {layer_name}")

        try:
            pt_layer = get_layer(layer_name)
        except ValueError as e:
            print(f"   PRESKACM: {e}")
            continue

        transform = None
        if pt_layer.crs() != raster_layer.crs():
            print("   (CRS transformacija aktiva)")
            transform = QgsCoordinateTransform(
                pt_layer.crs(),
                raster_layer.crs(),
                QgsProject.instance(),
            )

        fid_val    = {}
        null_count = 0
        for feat in pt_layer.getFeatures():
            val = sample_raster(raster_layer, feat, transform)
            fid_val[feat.id()] = val
            if val is None:
                null_count += 1

        print(f"   Točaka: {len(fid_val)}   NoData/izvan rastera: {null_count}")

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

        csv_path = os.path.join(OUTPUT_DIR, f"{layer_name}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh)
            writer.writerow(["fid", ATTR_NAME])
            for fid, val in fid_val.items():
                writer.writerow([fid, val if val is not None else ""])

        print(f"   CSV: {layer_name}.csv")


# ============================================================
#  POZADINSKI HISTOGRAM
# ============================================================

def export_background(raster_layer):
    """
    Izvozi histogram TRI rastera.
    Koristi QgsRasterBandStats + provider.histogram() za N_HIST_BINS stupaca.
    """
    provider = raster_layer.dataProvider()

    print(f"\nRacunam statistike rastera '{raster_layer.name()}'...")
    st = provider.bandStatistics(1, QgsRasterBandStats.All)

    tri_min = st.minimumValue
    tri_max = st.maximumValue
    print(f"  Min={tri_min:.2f}  Max={tri_max:.2f}  Mean={st.mean:.2f}  SD={st.stdDev:.2f}")

    print(f"  Racunam histogram ({N_HIST_BINS} stupaca)...")
    hist      = provider.histogram(1, N_HIST_BINS, tri_min, tri_max)
    hist_vec  = hist.histogramVector
    bin_width = (tri_max - tri_min) / N_HIST_BINS
    total     = sum(hist_vec)

    out_path = os.path.join(OUTPUT_DIR, "background_tri.csv")
    with open(out_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["tri_value", "n_piksela", "postotak"])
        for i, count in enumerate(hist_vec):
            bin_center = tri_min + (i + 0.5) * bin_width
            w.writerow([round(bin_center, 3), int(count), round(count / total * 100, 6)])

    print(f"  Exportano: background_tri.csv  ({N_HIST_BINS} stupaca, {total:,} piksela)")
    print(f"  Mean={st.mean:.2f}  SD={st.stdDev:.2f}")


# ============================================================
#  POKRENI
# ============================================================

def run_all():
    print("=" * 60)
    print("TRI - početak analize")
    print("=" * 60)

    try:
        raster_layer = get_layer(RASTER_LAYER_NAME)
    except ValueError as e:
        print(f"GRESKA: {e}")
        return

    print(f"Raster: {raster_layer.name()}   CRS: {raster_layer.crs().authid()}")

    print("\nUzorkovanje točaka...")
    sample_layers(raster_layer)

    print("\nIzvoz pozadinskog histograma...")
    export_background(raster_layer)

    print("\n" + "=" * 60)
    print("GOTOVO!")
    print(f"Izlaz: {OUTPUT_DIR}")
    print("=" * 60)


run_all()
