"""
VECINSKI_TIP_TLA - Dominantni tip tla oko točkastih lokaliteta
================================================================
Za svaki od 8 točkastih slojeva i 4 polumjera (100, 250, 500, 1000 m):
  1. Kružni buffer → zonalni histogram → argmax → dominantni WRB tip tla
  2. Dodaje atribute  vtl_100 / vtl_250 / vtl_500 / vtl_1000  u svaki sloj
  3. Izvozi 32 CSV datoteka  →  {OUTPUT_DIR}/{naziv_sloja}_r{polumjer}.csv

Svaki CSV sadrži sva originalna polja sloja + stupac "vecinski_tip_tla"
za taj polumjer + stupce "naziv_sloja" i "polumjer_m".
"""

from qgis.core import QgsProject, QgsField
from PyQt5.QtCore import QVariant
import processing
import os
import csv

# ============================================================
#  POSTAVKE  (mijenjaj samo ovdje)
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

# Provjeri naziv rasterskog sloja u QGIS-u (Layer Properties → naziv)
SOIL_RASTER_LAYER = "tipovi_tla"

RADII = [100, 250, 500, 1000]   # metri

# Mapa s CSV-ovima — kreira se automatski
OUTPUT_DIR = r"C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna\vecinski_tip_tla_\csv_output"

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
#  POMOĆNE FUNKCIJE
# ============================================================

def get_layer(name):
    layers = QgsProject.instance().mapLayersByName(name)
    if not layers:
        raise ValueError(f"Sloj '{name}' nije pronađen. Provjeri naziv u QGIS-u.")
    return layers[0]


def vtl_attr(radius):
    """Naziv atributa za dani polumjer, npr. vtl_100."""
    return f"vtl_{radius}"


def dominant_soil(hist_feat, value_to_col):
    """Vrati naziv dominantnog tipa tla za jedan feature histograma."""
    best_val   = None
    best_count = -1
    for soil_val, col in value_to_col.items():
        count = hist_feat[col]
        if count is None:
            count = 0
        if count > best_count:
            best_count = count
            best_val   = soil_val
    if best_val is None or best_count == 0:
        return "Nepoznato"
    return SOIL_TYPES.get(best_val, f"Val_{best_val}")


def build_value_col_map(hist_fields, prefix):
    """Poveži rasterske vrijednosti (iz SOIL_TYPES) s imenima stupaca histograma."""
    mapping = {}
    for val in SOIL_TYPES:
        for col in hist_fields:
            if col.startswith(prefix):
                suffix = col[len(prefix):]
                try:
                    if int(float(suffix)) == val:
                        mapping[val] = col
                        break
                except ValueError:
                    pass
    return mapping


# ============================================================
#  GLAVNA ANALIZA
# ============================================================

def run_dominant_soil_analysis():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        raster_layer = get_layer(SOIL_RASTER_LAYER)
    except ValueError as e:
        print(f"GREŠKA: {e}")
        print("Provjeri naziv rasterskog sloja u varijabli SOIL_RASTER_LAYER.")
        return

    print("=" * 65)
    print("VECINSKI TIP TLA — početak analize")
    print("=" * 65)
    print(f"Raster : {raster_layer.name()}   CRS: {raster_layer.crs().authid()}")
    print(f"Radijusi: {RADII} m")
    print(f"Slojevi ({len(POINT_LAYERS)}):")
    for name in POINT_LAYERS:
        print(f"  · {name}")
    print(f"Izlaz CSV: {OUTPUT_DIR}")
    print()

    total_csv = 0

    for layer_name in POINT_LAYERS:
        print(f"\n{'─'*60}")
        print(f"▶  {layer_name}")

        try:
            pt_layer = get_layer(layer_name)
        except ValueError as e:
            print(f"   PRESKAČEM: {e}")
            continue

        n = pt_layer.featureCount()
        print(f"   Točaka: {n}")

        if pt_layer.crs() != raster_layer.crs():
            print("   UPOZORENJE: CRS-ovi se razlikuju — buffer može biti netočan!")

        # FID redosljed — mora biti stabilan
        fid_list = [feat.id() for feat in pt_layer.getFeatures()]

        # results[fid][radius] = naziv dominantnog tipa
        results = {fid: {} for fid in fid_list}

        # ── Računanje po polumjeru ──────────────────────────────
        for radius in RADII:
            print(f"   Radijus {radius:4d} m ... ", end="", flush=True)

            buf = processing.run("native:buffer", {
                'INPUT':    pt_layer,
                'DISTANCE': radius,
                'SEGMENTS': 32,
                'DISSOLVE': False,
                'OUTPUT':   'memory:buf',
            })['OUTPUT']

            prefix = f"h{radius}_"
            hist = processing.run("native:zonalhistogram", {
                'INPUT_RASTER':  raster_layer,
                'RASTER_BAND':   1,
                'INPUT_VECTOR':  buf,
                'COLUMN_PREFIX': prefix,
                'OUTPUT':        'memory:hist',
            })['OUTPUT']

            hist_fields   = [f.name() for f in hist.fields()]
            value_to_col  = build_value_col_map(hist_fields, prefix)

            for fid, hfeat in zip(fid_list, hist.getFeatures()):
                results[fid][radius] = dominant_soil(hfeat, value_to_col)

            print("OK")

        # ── Dodaj / osvježi atribute u izvornom sloju ───────────
        print("   Zapisujem atribute u sloj...", end=" ", flush=True)
        pt_layer.startEditing()

        for radius in RADII:
            fname = vtl_attr(radius)
            if pt_layer.fields().indexFromName(fname) == -1:
                pt_layer.addAttribute(QgsField(fname, QVariant.String, "string", 50))

        pt_layer.updateFields()

        for feat in pt_layer.getFeatures():
            fid = feat.id()
            for radius in RADII:
                fname = vtl_attr(radius)
                idx   = pt_layer.fields().indexFromName(fname)
                if idx != -1:
                    pt_layer.changeAttributeValue(
                        fid, idx, results[fid].get(radius, "Nepoznato")
                    )

        pt_layer.commitChanges()
        print("OK")

        # ── Izvoz CSV-ova (jedan po polumjeru) ──────────────────
        for radius in RADII:
            csv_path  = os.path.join(OUTPUT_DIR, f"{layer_name}_vtt_r{radius}.csv")
            vtl_fname = vtl_attr(radius)

            with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
                writer = csv.writer(fh)
                writer.writerow(["fid", "vecinski_tip_tla"])
                for feat in pt_layer.getFeatures():
                    soil = feat[vtl_fname] or "Nepoznato"
                    writer.writerow([feat.id(), soil])

            total_csv += 1
            print(f"   CSV: {layer_name}_vtt_r{radius}.csv")

    print()
    print("=" * 65)
    print(f"GOTOVO!  Izvezeno {total_csv} CSV datoteka u:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 65)


# ============================================================
#  SAMO IZVOZ CSV (ako su vtl_* atributi već u slojevima)
# ============================================================

def export_csvs_only():
    """Čita vtl_* atribute koji su već u slojevima i izvozi 32 CSV-a."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total = 0
    for layer_name in POINT_LAYERS:
        try:
            pt_layer = get_layer(layer_name)
        except ValueError as e:
            print(f"PRESKAČEM {layer_name}: {e}")
            continue
        for radius in RADII:
            vtl_fname = vtl_attr(radius)
            if pt_layer.fields().indexFromName(vtl_fname) == -1:
                print(f"  {layer_name}: atribut '{vtl_fname}' nedostaje — pokreni run_dominant_soil_analysis()")
                continue
            csv_path = os.path.join(OUTPUT_DIR, f"{layer_name}_vtt_r{radius}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
                writer = csv.writer(fh)
                writer.writerow(["fid", "vecinski_tip_tla"])
                for feat in pt_layer.getFeatures():
                    soil = feat[vtl_fname] or "Nepoznato"
                    writer.writerow([feat.id(), soil])
            total += 1
            print(f"  {layer_name}_vtt_r{radius}.csv")
    print(f"\nIzvezeno {total} CSV datoteka → {OUTPUT_DIR}")


# ============================================================
#  POKRENI
# ============================================================
# Puna analiza (buffer + histogram + atributi + CSV):
# run_dominant_soil_analysis()

# Samo re-export CSV ako su vtl_* atributi već u slojevima:
export_csvs_only()



"""
Mod — koji tip tla se najčešće pojavljuje u skupu
Frekvencijska tablica — koliko točaka (n) i koliko % pripada svakom tipu
Stupčasti dijagram (bar chart) — vizualizacija frekvencija
Usporedba između skupova — npr. χ² (hi-kvadrat) test ili jednostavna vizualna usporedba frekvencija između neolitik_svi_odredeni i random_ceste_biased
"""