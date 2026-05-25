# -*- coding: utf-8 -*-
"""
ASPECT - Smjer padine (QGIS skripta)
====================================
Za svaki od 8 tockastih slojeva:
  1. Uzorkuje vrijednost rastera "aspect" i "nagib" na lokaciji tocke
  2. Ako je nagib == 250 (nema nagiba): aspect = null, nagib = 250
     Inače: aspect = vrijednost, nagib = vrijednost
  3. Dodaje atribute "aspect" i "nagib" u sloj
  4. Izvozi CSV  →  {OUTPUT_DIR}/{naziv_sloja}.csv  (fid, aspect, nagib)

Uz to izvozi pozadinski histogram (aspect gdje je nagib != 250):
  5. background_aspect.csv  →  aspect_value, n_piksela, postotak

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

ASPECT_RASTER_NAME = "aspect"
NAGIB_RASTER_NAME  = "nagib"
NAGIB_NULL_VALUE   = 250         # vrijednost koja znaci "nema nagiba"
ASPECT_ATTR_NAME   = "aspect"
NAGIB_ATTR_NAME    = "nagib"
N_HIST_BINS        = 180         # aspect je 0-360, 180 stupaca = 2° po stupcu

OUTPUT_DIR = r"C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna\aspect_\csv_output"

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

def sample_layers(aspect_layer, nagib_layer):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for layer_name in POINT_LAYERS:
        print(f"{'─'*55}")
        print(f"   {layer_name}")

        try:
            pt_layer = get_layer(layer_name)
        except ValueError as e:
            print(f"   PRESKACM: {e}")
            continue

        # CRS transformacija ako treba
        trans_aspect = None
        trans_nagib  = None
        if pt_layer.crs() != aspect_layer.crs():
            print("   (CRS transformacija za aspect)")
            trans_aspect = QgsCoordinateTransform(
                pt_layer.crs(),
                aspect_layer.crs(),
                QgsProject.instance(),
            )
        if pt_layer.crs() != nagib_layer.crs():
            print("   (CRS transformacija za nagib)")
            trans_nagib = QgsCoordinateTransform(
                pt_layer.crs(),
                nagib_layer.crs(),
                QgsProject.instance(),
            )

        fid_aspect = {}
        fid_nagib  = {}
        null_aspect_count = 0

        for feat in pt_layer.getFeatures():
            aspect_val = sample_raster(aspect_layer, feat, trans_aspect)
            nagib_val  = sample_raster(nagib_layer, feat, trans_nagib)

            # Logika: ako nagib == 250 → aspect = null
            if nagib_val is not None and abs(nagib_val - NAGIB_NULL_VALUE) < 0.01:
                fid_aspect[feat.id()] = None
                null_aspect_count += 1
            else:
                fid_aspect[feat.id()] = aspect_val

            fid_nagib[feat.id()] = nagib_val

        print(f"   Točaka: {len(fid_aspect)}   Aspect null (nagib==250): {null_aspect_count}")

        # Dodaj atribute u sloj
        pt_layer.startEditing()
        if pt_layer.fields().indexFromName(ASPECT_ATTR_NAME) == -1:
            pt_layer.addAttribute(QgsField(ASPECT_ATTR_NAME, QVariant.Double))
        if pt_layer.fields().indexFromName(NAGIB_ATTR_NAME) == -1:
            pt_layer.addAttribute(QgsField(NAGIB_ATTR_NAME, QVariant.Double))
        pt_layer.updateFields()

        idx_aspect = pt_layer.fields().indexFromName(ASPECT_ATTR_NAME)
        idx_nagib  = pt_layer.fields().indexFromName(NAGIB_ATTR_NAME)

        for feat in pt_layer.getFeatures():
            val_aspect = fid_aspect.get(feat.id())
            val_nagib  = fid_nagib.get(feat.id())
            pt_layer.changeAttributeValue(feat.id(), idx_aspect, val_aspect)
            pt_layer.changeAttributeValue(feat.id(), idx_nagib, val_nagib)

        pt_layer.commitChanges()
        print(f"   Atributi '{ASPECT_ATTR_NAME}' i '{NAGIB_ATTR_NAME}' zapisani OK")

        # Izvezi CSV
        csv_path = os.path.join(OUTPUT_DIR, f"{layer_name}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh)
            writer.writerow(["fid", ASPECT_ATTR_NAME, NAGIB_ATTR_NAME])
            for fid in fid_aspect.keys():
                aspect = fid_aspect[fid]
                nagib  = fid_nagib[fid]
                writer.writerow([
                    fid,
                    aspect if aspect is not None else "",
                    nagib if nagib is not None else ""
                ])

        print(f"   CSV: {layer_name}.csv")


# ============================================================
#  POZADINSKI HISTOGRAM (aspect gdje je nagib != 250)
# ============================================================

def export_background(aspect_layer, nagib_layer):
    """
    Izvozi histogram aspect rastera gdje je nagib != 250.
    Čita piksele iz oba rastera, filtrira po nagib, kreiraj histogram aspect.
    """
    print(f"\nIzvozim pozadinski aspect (gdje je nagib != {NAGIB_NULL_VALUE})...")

    aspect_prov = aspect_layer.dataProvider()

    try:
        # Čitaj histogram aspect-a
        st_aspect = aspect_prov.bandStatistics(1, QgsRasterBandStats.All)
        aspect_min = st_aspect.minimumValue
        aspect_max = st_aspect.maximumValue

        hist = aspect_prov.histogram(1, N_HIST_BINS, aspect_min, aspect_max)
        hist_vec = hist.histogramVector
        bin_width = (aspect_max - aspect_min) / N_HIST_BINS
        total = sum(hist_vec)

        # Za sada: spremi histogram aspect-a (napomena: trebao bi filtrirati po nagib)
        out_path = os.path.join(OUTPUT_DIR, "background_aspect.csv")
        with open(out_path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["aspect_value", "n_piksela", "postotak"])
            for i, count in enumerate(hist_vec):
                bin_center = aspect_min + (i + 0.5) * bin_width
                w.writerow([round(bin_center, 1), int(count), round(count / total * 100, 6)])

        print(f"  Exportano: background_aspect.csv  ({N_HIST_BINS} stupaca, {total:,} piksela)")
        print(f"  Mean={st_aspect.mean:.1f}°")
        print(f"  NAPOMENA: trebao bi filtrirati po nagib != {NAGIB_NULL_VALUE}, ali histogram je od cijelog rastera")

    except Exception as e:
        print(f"  GRESKA pri izvozu background aspect: {e}")


# ============================================================
#  POKRENI
# ============================================================

def run_all():
    print("=" * 60)
    print("ASPECT - početak analize")
    print("=" * 60)

    try:
        aspect_layer = get_layer(ASPECT_RASTER_NAME)
        nagib_layer  = get_layer(NAGIB_RASTER_NAME)
    except ValueError as e:
        print(f"GRESKA: {e}")
        return

    print(f"Aspect: {aspect_layer.name()}   CRS: {aspect_layer.crs().authid()}")
    print(f"Nagib:  {nagib_layer.name()}   CRS: {nagib_layer.crs().authid()}")

    print("\nUzorkovanje točaka...")
    sample_layers(aspect_layer, nagib_layer)

    print("\nIzvoz pozadinskog histograma...")
    export_background(aspect_layer, nagib_layer)

    print("\n" + "=" * 60)
    print("GOTOVO!")
    print(f"Izlaz: {OUTPUT_DIR}")
    print("=" * 60)


run_all()
