"""
BLIZINA RIJEKAMA - usporedba pravih naselja i nasumičnih točaka
===============================================================
Testira hipotezu: nalaze li se neolitička naselja statistički bliže
modeliranoj mreži rijeka nego nasumično raspoređene točke?

Metoda:
  - za svaku točku (naselje / nasumična) računa udaljenost do
    najbliže rijeke u mreži
  - Mann-Whitney U test: jesu li udaljenosti naselja statistički
    manje nego udaljenosti nasumičnih točaka
  - Cohen's d: veličina efekta (koliko su bliže)
  - postotak točaka unutar pragova (100m, 250m, 500m, 1000m, 2000m)

Interpretacija rezultata:
  p < 0.05  → statistički značajno (može se tvrditi razlika postoji)
  p < 0.01  → jaki dokazi
  Cohen's d:  0.2=mali, 0.5=srednji, 0.8=veliki efekt
  omjer medijana (nasumično / naselja): koliko su puta nasumicne točke
    dalje od rijeka nego prava naselja (npr. 2.5x)
"""

from qgis.core import (
    QgsProject,
    QgsField,
    QgsSpatialIndex,
)
from PyQt5.QtCore import QVariant
import os, math, statistics, csv

# ============================================================
#  POSTAVKE - OVDJE MIJENJAJ PARAMETRE
# ============================================================

SETTLEMENTS_LAYER  = "neolitik_svi_odredeni"
RANDOM_LAYER       = "nasumicni_lokaliteti_umjetno_generirani"
RIVERS_LAYER       = "tekucice"        # originalna mreža
RIVERS_COPY_LAYER  = "tekucice_copy"   # korigirana mreža (+10 rucno dodanih rijeka)

OUTPUT_DIR      = r"C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna\vode_output"
OUTPUT_CSV      = "blizina_rijekama.csv"
OUTPUT_CSV_COPY = "blizina_rijekama_copy.csv"

# koliko susjednih rijeka provjeriti u spatial indexu (više = točnije, sporije)
N_CANDIDATES = 10

# pragovi za % analizu (u metrima)
THRESHOLDS = [100, 250, 500, 1000, 2000, 5000]

# ============================================================
#  POMOĆNE FUNKCIJE
# ============================================================

def get_layer(name):
    layers = QgsProject.instance().mapLayersByName(name)
    if not layers:
        raise ValueError(f"Sloj '{name}' nije pronađen! Provjeri naziv u POSTAVKAMA.")
    return layers[0]


def build_river_index(river_layer):
    """Izgradi prostorni indeks za brzo traženje najbližih rijeka."""
    idx = QgsSpatialIndex()
    geoms = {}
    for feat in river_layer.getFeatures():
        idx.addFeature(feat)
        geoms[feat.id()] = feat.geometry()
    print(f"  Indeksirano {len(geoms)} segmenata rijeka.")
    return idx, geoms


def distance_to_nearest_river(point_geom, river_index, river_geoms):
    """Udaljenost od točke do najbliže rijeke (u jedinicama CRS-a)."""
    candidates = river_index.nearestNeighbor(point_geom.asPoint(), N_CANDIDATES)
    if not candidates:
        return None
    return min(point_geom.distance(river_geoms[fid]) for fid in candidates)


def compute_distances(layer, river_index, river_geoms, label=""):
    """Izračunaj udaljenosti svih točaka u sloju do najbliže rijeke."""
    distances = []
    total = layer.featureCount()
    for i, feat in enumerate(layer.getFeatures()):
        geom = feat.geometry()
        if geom.isEmpty():
            continue
        d = distance_to_nearest_river(geom, river_index, river_geoms)
        if d is not None:
            distances.append((feat.id(), d))
        if (i + 1) % 50 == 0:
            print(f"  {label}: {i+1}/{total} obrađeno...")
    return distances  # lista (fid, udaljenost)


def mann_whitney_u_test(group1, group2):
    """
    Mann-Whitney U test (normalna aproksimacija, radi za n > 20).
    H0: distribucije su iste
    H1: group1 ima manje vrijednosti od group2
    Vraća (U, z, p_dvostrani)
    """
    n1, n2 = len(group1), len(group2)
    combined = sorted([(v, 0) for v in group1] + [(v, 1) for v in group2])
    n = n1 + n2

    # Dodjeli rangove (prosječni rang za izjednačene vrijednosti)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    r1 = sum(ranks[k] for k in range(n) if combined[k][1] == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    mu_u    = n1 * n2 / 2.0
    sigma_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = (u - mu_u) / sigma_u
    p = math.erfc(abs(z) / math.sqrt(2))   # dvostrani p
    return u, z, p


def vda(group1, group2):
    """
    Vargha-Delaney A: P(group1 > group2) + 0.5 * P(group1 == group2).
    Poziva se kao vda(r_dists, s_dists) da VD-a > 0.5 znači naselja bliža rijekama.
    0.5 = nema razlike; 0.56=mali, 0.64=srednji, 0.71=veliki efekt.
    """
    n1, n2 = len(group1), len(group2)
    combined = sorted([(v, 0) for v in group1] + [(v, 1) for v in group2])
    n = n1 + n2
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and combined[j][0] == combined[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg
        i = j
    r1 = sum(ranks[k] for k in range(n) if combined[k][1] == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    return u1 / (n1 * n2)


def pct_within(distances, threshold):
    if not distances:
        return 0.0
    return 100.0 * sum(1 for d in distances if d <= threshold) / len(distances)


def percentile(data, p):
    s = sorted(data)
    idx = (len(s) - 1) * p / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])


def add_distance_attribute(layer, dist_dict, field_name="dist_rijeka"):
    """Dodaj udaljenost kao atribut na sloj."""
    if layer.fields().indexFromName(field_name) == -1:
        layer.startEditing()
        layer.addAttribute(QgsField(field_name, QVariant.Double))
        layer.updateFields()
    else:
        layer.startEditing()

    idx = layer.fields().indexFromName(field_name)
    for fid, dist in dist_dict.items():
        layer.changeAttributeValue(fid, idx, round(dist, 2))
    layer.commitChanges()
    print(f"  Atribut '{field_name}' dodan na sloj '{layer.name()}'.")


def interpret_result(p, a, ratio):
    """Tekstualna interpretacija rezultata."""
    lines = []

    if p < 0.001:
        lines.append("STATISTIČKA ZNAČAJNOST: p < 0.001 → iznimno jaki dokazi")
    elif p < 0.01:
        lines.append("STATISTIČKA ZNAČAJNOST: p < 0.01  → jaki dokazi")
    elif p < 0.05:
        lines.append("STATISTIČKA ZNAČAJNOST: p < 0.05  → značajno (standardni prag)")
    else:
        lines.append(f"STATISTIČKA ZNAČAJNOST: p = {p:.3f} → NIJE značajno (p ≥ 0.05)")

    if a >= 0.71:
        lines.append(f"VELIČINA EFEKTA: VD-a = {a:.3f} → veliki efekt")
    elif a >= 0.64:
        lines.append(f"VELIČINA EFEKTA: VD-a = {a:.3f} → srednji efekt")
    elif a >= 0.56:
        lines.append(f"VELIČINA EFEKTA: VD-a = {a:.3f} → mali efekt")
    else:
        lines.append(f"VELIČINA EFEKTA: VD-a = {a:.3f} → zanemariv efekt")

    lines.append(f"OMJER MEDIJANA: nasumično / naselja = {ratio:.2f}x")

    lines.append("")
    if p < 0.05 and a >= 0.64 and ratio >= 1.5:
        lines.append("ZAKLJUČAK: Prava naselja su statistički značajno i supstancijalno")
        lines.append("  bliža modeliranoj mreži rijeka od nasumičnih točaka.")
        lines.append("  → Model riječne mreže je KORISTAN prediktor neolitičke naseljenosti.")
        lines.append("  → Možeš ga koristiti kao aproksimaciju neolitičke hidrologije,")
        lines.append("    uz napomenu da je riječ o korelaciji, ne kauzalnosti.")
    elif p < 0.05 and a < 0.64:
        lines.append("ZAKLJUČAK: Razlika postoji, ali je mala. Model pokazuje neki")
        lines.append("  signal, no efekt je slab. Koristiti s oprezom.")
    elif p >= 0.05:
        lines.append("ZAKLJUČAK: Nema statistički značajne razlike. Model riječne mreže")
        lines.append("  ne predviđa lokacije naselja bolje od slučajnog rasporeda.")
        lines.append("  → Razmotri alternative (drugi DEM, drugačija rezolucija mreže).")

    return "\n".join(lines)


# ============================================================
#  GLAVNA LOGIKA
# ============================================================

def run_analysis(rivers_layer_name=None, output_csv=None, label="", attr_name="dist_rijeka"):
    """Pokreni analizu blizine rijekama.
    rivers_layer_name: naziv sloja u QGIS-u (default: RIVERS_LAYER)
    output_csv: naziv izlaznog CSV-a (default: OUTPUT_CSV)
    label: oznaka za ispis (npr. 'ORIGINALNA' ili 'KORIGIRANA')
    attr_name: naziv atributa koji se upisuje na tockovne slojeve
               ('dist_rijeka' za orig, 'dist_rijeka_korig' za korigirani)
    """
    if rivers_layer_name is None:
        rivers_layer_name = RIVERS_LAYER
    if output_csv is None:
        output_csv = OUTPUT_CSV

    print("=" * 65)
    print(f"BLIZINA RIJEKAMA - {label if label else rivers_layer_name}")
    print("=" * 65)

    settlements_layer = get_layer(SETTLEMENTS_LAYER)
    random_layer      = get_layer(RANDOM_LAYER)
    river_layer       = get_layer(rivers_layer_name)

    print(f"Naselja:       {settlements_layer.name()}  "
          f"({settlements_layer.featureCount()} točaka)")
    print(f"Nasumične:     {random_layer.name()}  "
          f"({random_layer.featureCount()} točaka)")
    print(f"Mreža rijeka:  {river_layer.name()}  "
          f"({river_layer.featureCount()} segmenata)")
    print(f"CRS naselja:   {settlements_layer.crs().authid()}")
    print(f"CRS rijeka:    {river_layer.crs().authid()}")

    if settlements_layer.crs() != river_layer.crs():
        print("UPOZORENJE: CRS se ne podudaraju! Udaljenosti mogu biti netočne.")
    print()

    # Izgradi prostorni indeks rijeka
    print("Gradim prostorni indeks rijeka...")
    river_index, river_geoms = build_river_index(river_layer)
    print()

    # Izračunaj udaljenosti
    print("Računam udaljenosti pravih naselja...")
    s_pairs = compute_distances(settlements_layer, river_index, river_geoms, "Naselja")
    print(f"  → {len(s_pairs)} naselja obrađena\n")

    print("Računam udaljenosti nasumičnih točaka...")
    r_pairs = compute_distances(random_layer, river_index, river_geoms, "Nasumične")
    print(f"  → {len(r_pairs)} točaka obrađeno\n")

    s_dists = [d for _, d in s_pairs]
    r_dists = [d for _, d in r_pairs]

    if not s_dists or not r_dists:
        print("GREŠKA: Nema dovoljno podataka za analizu.")
        return

    # Statistike
    s_mean   = statistics.mean(s_dists)
    s_median = statistics.median(s_dists)
    s_std    = statistics.stdev(s_dists)
    r_mean   = statistics.mean(r_dists)
    r_median = statistics.median(r_dists)
    r_std    = statistics.stdev(r_dists)

    # Testovi
    u, z, p = mann_whitney_u_test(s_dists, r_dists)
    a = vda(r_dists, s_dists)   # P(random > settlement) → >0.5 znači naselja bliža rijekama
    ratio = r_median / s_median if s_median > 0 else float("inf")

    # Ispis rezultata
    print("=" * 65)
    print("REZULTATI")
    print("=" * 65)
    print(f"{'':25s}  {'NASELJA':>12s}  {'NASUMIČNO':>12s}")
    print(f"{'N točaka':25s}  {len(s_dists):>12d}  {len(r_dists):>12d}")
    print(f"{'Srednja udalj. (m)':25s}  {s_mean:>12.1f}  {r_mean:>12.1f}")
    print(f"{'Medijan udalj. (m)':25s}  {s_median:>12.1f}  {r_median:>12.1f}")
    print(f"{'St. devijacija (m)':25s}  {s_std:>12.1f}  {r_std:>12.1f}")
    print(f"{'P10 (m)':25s}  {percentile(s_dists,10):>12.1f}  {percentile(r_dists,10):>12.1f}")
    print(f"{'P25 (m)':25s}  {percentile(s_dists,25):>12.1f}  {percentile(r_dists,25):>12.1f}")
    print(f"{'P75 (m)':25s}  {percentile(s_dists,75):>12.1f}  {percentile(r_dists,75):>12.1f}")
    print(f"{'P90 (m)':25s}  {percentile(s_dists,90):>12.1f}  {percentile(r_dists,90):>12.1f}")
    print()
    print(f"{'Prag':25s}  {'% naselja':>12s}  {'% nasumično':>12s}")
    for thr in THRESHOLDS:
        ps = pct_within(s_dists, thr)
        pr = pct_within(r_dists, thr)
        print(f"  unutar {thr:>5d} m       {ps:>11.1f}%  {pr:>11.1f}%")
    print()
    p_str = f"{p:.4f}" if p >= 0.0001 else f"{p:.2e}"
    print(f"Mann-Whitney U = {u:.0f},  z = {z:.3f},  p = {p_str}")
    print(f"VD-a = {a:.3f}")
    print()
    print(interpret_result(p, a, ratio))
    print()

    # Spremi CSV s pojedinačnim udaljenostima
    csv_path = os.path.join(OUTPUT_DIR, output_csv)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "fid", "dist_rijeka_m"])
        for fid, d_val in s_pairs:
            writer.writerow(["naselje", fid, round(d_val, 2)])
        for fid, d_val in r_pairs:
            writer.writerow(["nasumicno", fid, round(d_val, 2)])
    print(f"CSV s udaljenostima spremljen: {csv_path}")
    print()

    # Dodaj udaljenosti kao atribute na slojeve
    print("Dodajem atribute na slojeve...")
    add_distance_attribute(settlements_layer, dict(s_pairs), field_name=attr_name)
    add_distance_attribute(random_layer,      dict(r_pairs), field_name=attr_name)

    print()
    print("=" * 65)
    print("GOTOVO.")
    print("=" * 65)


# ============================================================
#  POKRENI
# ============================================================

# --- Originalna mreža ---
run_analysis(
    rivers_layer_name = RIVERS_LAYER,
    output_csv        = OUTPUT_CSV,
    label             = "ORIGINALNA MREZA",
    attr_name         = "dist_rijeka",
)

# --- Korigirana mreža (+10 rucno dodanih rijeka) ---
run_analysis(
    rivers_layer_name = RIVERS_COPY_LAYER,
    output_csv        = OUTPUT_CSV_COPY,
    label             = "KORIGIRANA MREZA (+10 rijeka)",
    attr_name         = "dist_rijeka_korig",
)


'''
=================================================================
BLIZINA RIJEKAMA - ORIGINALNA MREZA
=================================================================
Naselja:       neolitik_svi_odredeni  (274 točaka)
Nasumične:     nasumicni_lokaliteti_umjetno_generirani  (274 točaka)
Mreža rijeka:  tekucice  (865 segmenata)
CRS naselja:   EPSG:3765
CRS rijeka:    EPSG:3765

Gradim prostorni indeks rijeka...
  Indeksirano 865 segmenata rijeka.

Računam udaljenosti pravih naselja...
  Naselja: 50/274 obrađeno...
  Naselja: 100/274 obrađeno...
  Naselja: 150/274 obrađeno...
  Naselja: 200/274 obrađeno...
  Naselja: 250/274 obrađeno...
  → 274 naselja obrađena

Računam udaljenosti nasumičnih točaka...
  Nasumične: 50/274 obrađeno...
  Nasumične: 100/274 obrađeno...
  Nasumične: 150/274 obrađeno...
  Nasumične: 200/274 obrađeno...
  Nasumične: 250/274 obrađeno...
  → 274 točaka obrađeno

=================================================================
REZULTATI
=================================================================
                                NASELJA     NASUMIČNO
N točaka                            274           274
Srednja udalj. (m)                848.1        1188.3
Medijan udalj. (m)                524.6         899.2
St. devijacija (m)                841.0         986.4
P10 (m)                           110.3         146.8
P25 (m)                           203.5         443.5
P75 (m)                          1222.3        1718.2
P90 (m)                          2079.0        2609.2

Prag                          % naselja   % nasumično
  unutar   100 m               9.5%          7.7%
  unutar   250 m              29.9%         15.0%
  unutar   500 m              49.3%         28.5%
  unutar  1000 m              68.2%         53.3%
  unutar  2000 m              88.3%         81.4%
  unutar  5000 m             100.0%         99.6%

Mann-Whitney U = 28919,  z = -4.651,  p = 3.31e-06
VD-a = 0.615

STATISTIČKA ZNAČAJNOST: p < 0.001 → iznimno jaki dokazi
VELIČINA EFEKTA: VD-a = 0.615 → mali efekt
OMJER MEDIJANA: nasumično / naselja = 1.71x

ZAKLJUČAK: Razlika postoji, ali je mala. Model pokazuje neki
  signal, no efekt je slab. Koristiti s oprezom.

CSV s udaljenostima spremljen: C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna\vode_output\blizina_rijekama.csv

Dodajem atribute na slojeve...
  Atribut 'dist_rijeka' dodan na sloj 'neolitik_svi_odredeni'.
  Atribut 'dist_rijeka' dodan na sloj 'nasumicni_lokaliteti_umjetno_generirani'.

=================================================================
GOTOVO.
=================================================================
=================================================================
BLIZINA RIJEKAMA - KORIGIRANA MREZA (+10 rijeka)
=================================================================
Naselja:       neolitik_svi_odredeni  (274 točaka)
Nasumične:     nasumicni_lokaliteti_umjetno_generirani  (274 točaka)
Mreža rijeka:  tekucice_copy  (875 segmenata)
CRS naselja:   EPSG:3765
CRS rijeka:    EPSG:3765

Gradim prostorni indeks rijeka...
  Indeksirano 875 segmenata rijeka.

Računam udaljenosti pravih naselja...
  Naselja: 50/274 obrađeno...
  Naselja: 100/274 obrađeno...
  Naselja: 150/274 obrađeno...
  Naselja: 200/274 obrađeno...
  Naselja: 250/274 obrađeno...
  → 274 naselja obrađena

Računam udaljenosti nasumičnih točaka...
  Nasumične: 50/274 obrađeno...
  Nasumične: 100/274 obrađeno...
  Nasumične: 150/274 obrađeno...
  Nasumične: 200/274 obrađeno...
  Nasumične: 250/274 obrađeno...
  → 274 točaka obrađeno

=================================================================
REZULTATI
=================================================================
                                NASELJA     NASUMIČNO
N točaka                            274           274
Srednja udalj. (m)                709.8        1146.6
Medijan udalj. (m)                436.8         881.6
St. devijacija (m)                699.3         936.9
P10 (m)                           110.3         146.8
P25 (m)                           193.9         438.7
P75 (m)                          1011.5        1690.6
P90 (m)                          1702.3        2471.0

Prag                          % naselja   % nasumično
  unutar   100 m               9.5%          7.7%
  unutar   250 m              32.5%         15.0%
  unutar   500 m              54.4%         28.8%
  unutar  1000 m              74.5%         54.0%
  unutar  2000 m              93.4%         82.5%
  unutar  5000 m             100.0%         99.6%

Mann-Whitney U = 26217,  z = -6.109,  p = 1.01e-09
VD-a = 0.651

STATISTIČKA ZNAČAJNOST: p < 0.001 → iznimno jaki dokazi
VELIČINA EFEKTA: VD-a = 0.651 → srednji efekt
OMJER MEDIJANA: nasumično / naselja = 2.02x

ZAKLJUČAK: Prava naselja su statistički značajno i supstancijalno
  bliža modeliranoj mreži rijeka od nasumičnih točaka.
  → Model riječne mreže je KORISTAN prediktor neolitičke naseljenosti.
  → Možeš ga koristiti kao aproksimaciju neolitičke hidrologije,
    uz napomenu da je riječ o korelaciji, ne kauzalnosti.

CSV s udaljenostima spremljen: C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna\vode_output\blizina_rijekama_copy.csv

Dodajem atribute na slojeve...
  Atribut 'dist_rijeka_korig' dodan na sloj 'neolitik_svi_odredeni'.
  Atribut 'dist_rijeka_korig' dodan na sloj 'nasumicni_lokaliteti_umjetno_generirani'.

=================================================================
GOTOVO.
=================================================================
'''