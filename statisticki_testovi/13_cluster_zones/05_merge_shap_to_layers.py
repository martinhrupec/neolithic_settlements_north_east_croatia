"""
Merge SHAP attribution atributa u clusters_thr_85 slojeve (QGIS Python console)
==============================================================================

Pokrenuti u QGIS-u (Plugins -> Python Console -> Show Editor -> Open Script).
Slojevi se dohvacaju po imenu iz aktivnog projekta, treba samo CSV path.

Slojevi koji se ocekuju u projektu:
  - clusters_thr_85
  - clusters_thr_85_with_confidence
  - clusters_thr_85_top10pct

Stupci iz CSV-a koji vec postoje u sloju (osim cluster_id) preskacu se,
da se ne pregaze postojece vrijednosti.
"""

import csv

from qgis.core import (
    QgsProject,
    QgsField,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QVariant


# === KONFIG =================================================================

CSV_PATH = r"c:\Users\Martin\Desktop\skripte_za_diplomski\statisticki_testovi\13_cluster_zones\shap_attribution_types.csv"

LAYER_NAMES = [
    "clusters_thr_85",
    "clusters_thr_85_with_confidence",
    "clusters_thr_85_top10pct",
]

JOIN_KEY = "cluster_id"          # stupac u CSV-u i u sloju
SKIP_ALWAYS = {"source", JOIN_KEY}   # ne kopirati iz CSV-a

# ============================================================================


def _parse_value(v):
    """'' -> None, brojcani string -> int/float, inace string."""
    if v is None or v == "":
        return None
    try:
        if "." in v or "e" in v.lower():
            return float(v)
        return int(v)
    except ValueError:
        return v


def load_csv_by_cluster(csv_path):
    """Vrati (rows_by_cid, column_order)."""
    by_id = {}
    column_order = []
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        column_order = [c for c in reader.fieldnames if c not in SKIP_ALWAYS]
        for row in reader:
            try:
                cid = int(row[JOIN_KEY])
            except (KeyError, ValueError):
                continue
            by_id[cid] = {k: _parse_value(row.get(k)) for k in column_order}
    return by_id, column_order


def _qvariant_type_for(values):
    """Odluci QVariant tip za stupac na osnovu skupa vrijednosti."""
    has_float = False
    has_int = False
    has_str = False
    for v in values:
        if v is None:
            continue
        if isinstance(v, float):
            has_float = True
        elif isinstance(v, int) and not isinstance(v, bool):
            has_int = True
        else:
            has_str = True
    if has_str:
        return QVariant.String
    if has_float:
        return QVariant.Double
    if has_int:
        return QVariant.Int
    return QVariant.String


def merge_into_layer(layer, csv_by_id, csv_columns):
    name = layer.name()
    print(f"\n--- {name} ---")

    existing = {f.name() for f in layer.fields()}
    new_cols = [c for c in csv_columns if c not in existing]
    skipped  = [c for c in csv_columns if c in existing]

    print(f"    postojecih stupaca : {len(existing)}")
    print(f"    novih stupaca      : {len(new_cols)} -> {new_cols}")
    if skipped:
        print(f"    preskacem          : {skipped}")

    if not new_cols:
        print("    nista novo za dodati.")
        return

    # tip svakog novog stupca
    col_types = {}
    for c in new_cols:
        vals = [row.get(c) for row in csv_by_id.values()]
        col_types[c] = _qvariant_type_for(vals)

    # ----- dodaj polja -----
    if not layer.startEditing():
        print("    GRESKA: ne mogu otvoriti layer za editing.")
        return

    provider = layer.dataProvider()
    new_fields = [QgsField(c, col_types[c]) for c in new_cols]
    provider.addAttributes(new_fields)
    layer.updateFields()

    # mapping cluster_id -> feature id
    cid_field_idx = layer.fields().indexOf(JOIN_KEY)
    if cid_field_idx < 0:
        print(f"    GRESKA: sloj nema polje '{JOIN_KEY}'.")
        layer.rollBack()
        return

    new_field_idx = {c: layer.fields().indexOf(c) for c in new_cols}

    matched, unmatched = 0, 0
    for feat in layer.getFeatures():
        cid_raw = feat[JOIN_KEY]
        try:
            cid = int(cid_raw)
        except (TypeError, ValueError):
            unmatched += 1
            continue
        csv_row = csv_by_id.get(cid)
        if csv_row is None:
            unmatched += 1
            continue
        for c in new_cols:
            layer.changeAttributeValue(feat.id(), new_field_idx[c], csv_row.get(c))
        matched += 1

    if not layer.commitChanges():
        errs = layer.commitErrors()
        print(f"    GRESKA pri commit-u: {errs}")
        layer.rollBack()
        return

    print(f"    matched   : {matched}")
    print(f"    unmatched : {unmatched}")
    print(f"    OK, polja dodana i popunjena.")


def main():
    print("=" * 70)
    print("MERGE SHAP attribution -> cluster slojevi (QGIS)")
    print("=" * 70)

    print(f"\n[1] CSV: {CSV_PATH}")
    csv_by_id, csv_columns = load_csv_by_cluster(CSV_PATH)
    print(f"    cluster-a u CSV-u : {len(csv_by_id)}")
    print(f"    stupaca za join   : {len(csv_columns)}")

    project = QgsProject.instance()
    print("\n[2] Mergeam slojeve")
    for name in LAYER_NAMES:
        matches = project.mapLayersByName(name)
        if not matches:
            print(f"\n--- {name}: NEMA U PROJEKTU, preskacem.")
            continue
        layer = matches[0]
        if not isinstance(layer, QgsVectorLayer):
            print(f"\n--- {name}: nije vector layer, preskacem.")
            continue
        merge_into_layer(layer, csv_by_id, csv_columns)

    print("\nGotovo.")


main()
