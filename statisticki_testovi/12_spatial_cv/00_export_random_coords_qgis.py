"""
00_export_random_coords_qgis.py
================================
Pokrene se JEDNOM u QGIS Python konzoli.
Izvlaci koordinate (x, y) iz random_ceste_biased sloja i sprema ih u CSV.

Output: statisticki_testovi/12_spatial_cv/random_ceste_coords.csv
        sa stupcima: fid, x, y

Identicna logika kao 01_prostorna_autokorelacija/export_coords_qgis.py,
samo za drugi sloj.

Pretpostavka: QGIS projekt je otvoren, sloj "random_ceste_biased" je ucitan.
"""

import csv
import os
from qgis.core import QgsProject


LAYER_NAME = "random_ceste_biased"
OUT_PATH   = (r"c:\Users\Martin\Desktop\skripte_za_diplomski"
              r"\statisticki_testovi\12_spatial_cv\random_ceste_coords.csv")


def main():
    layers = QgsProject.instance().mapLayersByName(LAYER_NAME)
    if not layers:
        print(f"GRESKA: sloj '{LAYER_NAME}' nije pronaden u projektu.")
        return
    layer = layers[0]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    n = 0
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fid", "x", "y"])
        for feat in layer.getFeatures():
            geom = feat.geometry()
            if geom is None or geom.isEmpty():
                continue
            pt = geom.asPoint()
            writer.writerow([feat.id(), pt.x(), pt.y()])
            n += 1

    print(f"GOTOVO. {n} tocaka  ->  {OUT_PATH}")
    print(f"CRS sloja: {layer.crs().authid()}  ({layer.crs().description()})")


if __name__ == "__console__":
    main()
else:
    main()
