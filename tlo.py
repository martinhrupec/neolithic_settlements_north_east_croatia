"""
TLO - površina tipova tla oko naselja
======================================
Za svako naselje računa koliko km² svaki WRB tip tla zauzima
unutar zadanih polumjera (100, 250, 500, 1000 m).

REDOSLJED POKRETANJA:
  1. Pokreni discover_soil_types()  → vidi koje rasterske vrijednosti postoje
  2. Ispuni SOIL_TYPES rječnik dolje
  3. Pokreni run_analysis()         → dodaje atribute u sloj naselja

Nazivi atributa: tlo_{polumjer}_{naziv}   npr. tlo_100_Luvisol
"""

from qgis.core import (
    QgsProject,
    QgsField,
)
from PyQt5.QtCore import QVariant
import processing

# ============================================================
#  POSTAVKE
# ============================================================

SETTLEMENTS_LAYER = "neolitik_svi_odredeni"
SOIL_RASTER_LAYER = "tipovi_tla"               # <- naziv rasterskog sloja

RADII = [100, 250, 500, 1000]               # metri

# Korak 2: ispuni nakon što pokreneš discover_soil_types()
# Format:  rasterska_vrijednost: "NazivTla"
SOIL_TYPES = {
    2: "Alisols",
    4: "Arenosols",
    5: "Calcisols",
    6: "Cambisols",
    7: "Chernozems",
    11: "Fluvisols",
    12: "Gleysols",
    16: "Leptosols",
    18: "Luvisols",
    20: "Pheozems",
    24: "Regosols",
    29: "Vertisols"
}

# ============================================================
#  POMOĆNE FUNKCIJE
# ============================================================

def get_layer(name):
    layers = QgsProject.instance().mapLayersByName(name)
    if not layers:
        raise ValueError(f"Sloj '{name}' nije pronađen! Provjeri naziv u POSTAVKAMA.")
    return layers[0]


def field_name(radius, soil_name):
    """Pretvori polumjer i naziv tla u ispravan naziv atributa."""
    clean = soil_name.replace(" ", "_").replace("-", "_")
    return f"tlo_{radius}_{clean}"


# ============================================================
#  KORAK 1 - discover_soil_types()
# ============================================================

def discover_soil_types():
    """
    Ispiši sve jedinstvene vrijednosti u rasterskom sloju tla
    i njihovu ukupnu površinu. Pokreni ovo PRVO.
    """
    raster_layer = get_layer(SOIL_RASTER_LAYER)

    px = raster_layer.rasterUnitsPerPixelX()
    py = raster_layer.rasterUnitsPerPixelY()
    pixel_area_km2 = (px * py) / 1_000_000

    print(f"Raster: {raster_layer.name()}")
    print(f"CRS:    {raster_layer.crs().authid()}")
    print(f"Piksel: {px:.1f} × {py:.1f} m  ({pixel_area_km2*1e6:.0f} m²)")
    print()

    result = processing.run("native:rasterlayeruniquevaluesreport", {
        'INPUT': raster_layer,
        'BAND': 1,
        'OUTPUT_TABLE': 'memory:soil_values',
    })

    table = result['OUTPUT_TABLE']

    values = []
    rows = []
    for feat in table.getFeatures():
        val = int(float(feat['value']))
        count = int(feat['count'])
        area = count * pixel_area_km2
        values.append(val)
        rows.append((val, count, area))

    rows.sort()
    print(f"  {'Vrijednost':>12}  {'Piksela':>12}  {'Površina (km²)':>16}")
    print("  " + "-" * 46)
    for val, count, area in rows:
        print(f"  {val:>12}  {count:>12,}  {area:>16.2f}")

    print()
    print("─" * 50)
    print("Ispuni SOIL_TYPES u POSTAVKAMA:")
    print()
    print("SOIL_TYPES = {")
    for val, _, _ in rows:
        print(f"    {val}: \"???\",")
    print("}")
    print()
    print("Zamijeni '???' s pravim WRB nazivom, zatim pokreni run_analysis().")


# ============================================================
#  KORAK 3 - run_analysis()
# ============================================================

def run_analysis():
    if not SOIL_TYPES:
        print("GREŠKA: SOIL_TYPES je prazan — ispuni ga i pokušaj ponovo.")
        return

    settlements_layer = get_layer(SETTLEMENTS_LAYER)
    raster_layer      = get_layer(SOIL_RASTER_LAYER)

    px = raster_layer.rasterUnitsPerPixelX()
    py = raster_layer.rasterUnitsPerPixelY()
    pixel_area_km2 = (px * py) / 1_000_000

    print("=" * 60)
    print("TLO - početak analize")
    print("=" * 60)
    print(f"Naselja:  {settlements_layer.name()}  ({settlements_layer.featureCount()} točaka)")
    print(f"Raster:   {raster_layer.name()}")
    print(f"Piksel:   {px:.1f}×{py:.1f} m = {pixel_area_km2:.8f} km²")
    print(f"Polumjeri: {RADII} m")
    print(f"Tipovi tla ({len(SOIL_TYPES)}): {list(SOIL_TYPES.values())}")
    print()

    if settlements_layer.crs() != raster_layer.crs():
        print("UPOZORENJE: CRS se ne podudaraju! Površine mogu biti netočne.")

    # Pokupi originalne FID-ove u redoslijedu iteracije
    fid_list = [feat.id() for feat in settlements_layer.getFeatures()]

    # all_results[fid][naziv_atributa] = area_km2
    all_results = {fid: {} for fid in fid_list}

    for radius in RADII:
        print(f"Polumjer {radius} m ...")

        # Stvori buffer (in-memory)
        buf = processing.run("native:buffer", {
            'INPUT':       settlements_layer,
            'DISTANCE':    radius,
            'SEGMENTS':    32,
            'DISSOLVE':    False,
            'OUTPUT':      'memory:buf',
        })['OUTPUT']

        # Zonalni histogram — broji piksele po vrijednosti unutar svakog poligona
        prefix = f"s{radius}_"
        hist = processing.run("native:zonalhistogram", {
            'INPUT_RASTER':  raster_layer,
            'RASTER_BAND':   1,
            'INPUT_VECTOR':  buf,
            'COLUMN_PREFIX': prefix,
            'OUTPUT':        'memory:hist',
        })['OUTPUT']

        # Pronađi stvarna imena stupaca (vrijednost može biti "1" ili "1.0")
        hist_field_names = [f.name() for f in hist.fields()]
        value_to_col = {}
        for val in SOIL_TYPES:
            for col in hist_field_names:
                if col.startswith(prefix):
                    suffix = col[len(prefix):]
                    try:
                        if int(float(suffix)) == val:
                            value_to_col[val] = col
                            break
                    except ValueError:
                        pass

        # Pokupi rezultate; pretpostavljamo isti redosljed feature-a kao originalni sloj
        for orig_fid, hist_feat in zip(fid_list, hist.getFeatures()):
            for val, soil_name in SOIL_TYPES.items():
                col   = value_to_col.get(val)
                count = hist_feat[col] if col else 0
                if count is None:
                    count = 0
                fname = field_name(radius, soil_name)
                all_results[orig_fid][fname] = round(float(count) * pixel_area_km2, 6)

        print(f"  → gotovo")

    # Dodaj polja i upiši vrijednosti u originalni sloj
    print()
    print("Zapisujem atribute u sloj...")
    settlements_layer.startEditing()

    for radius in RADII:
        for soil_name in SOIL_TYPES.values():
            fname = field_name(radius, soil_name)
            if settlements_layer.fields().indexFromName(fname) == -1:
                settlements_layer.addAttribute(QgsField(fname, QVariant.Double))

    settlements_layer.updateFields()

    for feat in settlements_layer.getFeatures():
        fid = feat.id()
        for fname, val in all_results.get(fid, {}).items():
            idx = settlements_layer.fields().indexFromName(fname)
            if idx != -1:
                settlements_layer.changeAttributeValue(fid, idx, val)

    settlements_layer.commitChanges()

    n_fields = len(RADII) * len(SOIL_TYPES)
    print()
    print("=" * 60)
    print(f"GOTOVO! Dodano {n_fields} novih atributa.")
    print(f"Primjeri: {field_name(RADII[0], list(SOIL_TYPES.values())[0])}, ...")
    print("=" * 60)


# ============================================================
#  POKRENI - odkomentiraj što trebaš
# ============================================================
#discover_soil_types()
run_analysis()
