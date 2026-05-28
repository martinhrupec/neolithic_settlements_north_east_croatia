# -*- coding: utf-8 -*-
"""
03_QGIS_LOAD_HEATMAP — ucitaj i stiliziraj probability raster u QGIS
======================================================================

Cilj: ucitaj probability_neolitik.tif u QGIS i postavi pseudo-color
renderer prema kategorijama vjerojatnosti:

  0.00 - 0.30   prozirno / siva     (model "nije siguran")
  0.30 - 0.50   svjetlo zuta        (eksplorativna zona)
  0.50 - 0.70   narandzasta         (srednja konfidencija)
  0.70 - 1.00   crvena              (visoka konfidencija)

Pokretanje: u QGIS Python konzoli, otvori projekt s loadanim slojevima.
"""

import os
from qgis.core import (
    QgsProject, QgsRasterLayer,
    QgsColorRampShader, QgsRasterShader,
    QgsSingleBandPseudoColorRenderer,
)
from PyQt5.QtGui import QColor


RASTER_PATH = r"c:\Users\Martin\Desktop\skripte_za_diplomski\12_heatmap_qgis\probability_neolitik.tif"
LAYER_NAME  = "Probability_Neolitik"


# Color stops:  (value, QColor, label)
COLOR_STOPS = [
    (0.00, QColor(255, 255, 255,   0), "0.00 - 0.30  (model nesiguran)"),
    (0.30, QColor(255, 255, 178, 200), "0.30 - 0.50  (eksplorativna)"),
    (0.50, QColor(254, 204,  92, 220), "0.50 - 0.70  (srednja konfidencija)"),
    (0.70, QColor(253, 141,  60, 235), "0.70 - 0.85  (visoka konfidencija)"),
    (0.85, QColor(189,   0,  38, 245), "0.85 - 1.00  (vrlo visoka konfidencija)"),
    (1.00, QColor(128,   0,  38, 255), "1.00"),
]


def main():
    if not os.path.exists(RASTER_PATH):
        raise RuntimeError(f"Raster nije pronaden: {RASTER_PATH}\n"
                           f"Pokreni prvo 02_predict_raster.py.")

    print(f"Ucitavam raster: {RASTER_PATH}")
    layer = QgsRasterLayer(RASTER_PATH, LAYER_NAME)
    if not layer.isValid():
        raise RuntimeError(f"Raster nije validan: {RASTER_PATH}")

    # ----- Pseudo-color shader -----
    shader = QgsRasterShader()
    ramp = QgsColorRampShader()
    ramp.setColorRampType(QgsColorRampShader.Interpolated)

    items = [
        QgsColorRampShader.ColorRampItem(val, color, label)
        for val, color, label in COLOR_STOPS
    ]
    ramp.setColorRampItemList(items)
    shader.setRasterShaderFunction(ramp)

    renderer = QgsSingleBandPseudoColorRenderer(
        layer.dataProvider(), 1, shader,
    )
    # Postavi raspon eksplicitno na 0..1 (probability)
    renderer.setClassificationMin(0.0)
    renderer.setClassificationMax(1.0)

    layer.setRenderer(renderer)
    layer.triggerRepaint()

    # ----- Dodaj u projekt -----
    QgsProject.instance().addMapLayer(layer)

    print(f"Sloj dodan u projekt: {LAYER_NAME}")
    print()
    print("INTERPRETACIJA HEATMAPE:")
    print("  bijelo/prozirno  : p < 0.30  - model ne predvida neolitik")
    print("  zuto             : 0.30-0.50 - eksplorativna zona, mozda vrijedi obici")
    print("  narandzasto      : 0.50-0.70 - srednja konfidencija")
    print("  crveno           : > 0.70    - visoka konfidencija, prioritet za field survey")
    print()
    print("Provjeri u QGIS Layer panelu da je sloj '%s' aktivan." % LAYER_NAME)


main()
