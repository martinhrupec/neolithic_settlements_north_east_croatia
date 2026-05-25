"""
CESTE - analiza biasa istraženosti
=====================================
Dvije analize:

  A) TOČKE: KS test + Vargha-Delaney A između pravih i nasumičnih
     nalazišta za atribute duljine cesta u radijusu 1 km i 2 km
     → testira jesu li prava nalazišta bliže cestama (sampling bias)

  B) GRID: Pearson i Spearman korelacija između gustoće cesta i
     gustoće nalazišta po ćelijama grida
     → testira postoji li prostorna veza na razini krajobraza

Pokretanje: Otvori QGIS → Plugins → Python Console → Run Script
"""

from qgis.core import QgsProject
import math, statistics, os

# ============================================================
#  POSTAVKE
# ============================================================

# --- A) Točkasta analiza ---
SETTLEMENTS_LAYER = "neolitik_svi_odredeni"
RANDOM_LAYER      = "nasumicni_lokaliteti_umjetno_generirani"

# Nazivi atributa s duljinom cesta (metri cesta u radijusu)
ROAD_FIELD_1KM  = "ceste_1km_buffer_sum_LENGTH"    # <- provjeri naziv atributa
ROAD_FIELD_2KM  = "ceste_2km_buffer_sum_LENGTH"    # <- provjeri naziv atributa

# --- B) Grid analiza ---
GRID_LAYER              = "grid_s_brojem_nalazista_i_duljinom_cesta"              # <- naziv grid sloja
GRID_ROAD_DENSITY       = "gustoca_cesta_km_ceste_po_km2_grida"     # <- km cesta / km²
GRID_SETTLEMENT_DENSITY = "gustoca_nalazista_po_km2" # <- nalazišta / km²

# --- C) Parcijalna korelacija ---
# Skripta će izračunati % ćelije koji je Gleysol ili Fluvisol i dodati ga na grid.
# Mora biti isti WRB raster kao u tlo.py.
SOIL_RASTER_LAYER  = "tipovi_tla"   # <- naziv rasterskog sloja tla
GLEYSOLS_VALUE     = 12             # <- rasterska vrijednost Gleysola
FLUVISOLS_VALUE    = 11             # <- rasterska vrijednost Fluvisola
GRID_WETSOIL_FIELD = "pct_mocvara"  # <- naziv novog atributa koji se dodaje na grid

# ============================================================
#  POMOĆNE FUNKCIJE - STATISTIKA
# ============================================================

def get_layer(name):
    layers = QgsProject.instance().mapLayersByName(name)
    if not layers:
        raise ValueError(f"Sloj '{name}' nije pronađen!")
    return layers[0]


def read_field(layer, field_name):
    """Pokupi ne-null vrijednosti iz jednog polja kao listu floatova."""
    vals = []
    for feat in layer.getFeatures():
        v = feat[field_name]
        if v is not None and v == v:   # None i NaN check
            vals.append(float(v))
    return vals


def ks_test_2sample(group1, group2):
    """
    Dvostrani Kolmogorov-Smirnov test.
    H0: obje grupe imaju istu distribuciju.
    Vraća (D, p).
    """
    n1, n2 = len(group1), len(group2)
    all_vals = sorted(set(group1 + group2))

    g1_sorted = sorted(group1)
    g2_sorted = sorted(group2)

    def ecdf(sorted_data, x):
        lo, hi = 0, len(sorted_data)
        while lo < hi:
            mid = (lo + hi) // 2
            if sorted_data[mid] <= x:
                lo = mid + 1
            else:
                hi = mid
        return lo / len(sorted_data)

    D = max(abs(ecdf(g1_sorted, x) - ecdf(g2_sorted, x)) for x in all_vals)

    # Asimptotska p-vrijednost (Kolmogorov distribucija)
    lam = D * math.sqrt(n1 * n2 / (n1 + n2))
    p = 0.0
    for k in range(1, 101):
        p += ((-1) ** (k - 1)) * math.exp(-2 * k * k * lam * lam)
    p = max(0.0, min(1.0, 2 * p))
    return D, p


def vda(group1, group2):
    """
    Vargha-Delaney A: P(group1 > group2) + 0.5 * P(group1 == group2).
    0.5 = nema razlike; > 0.5 = group1 ima veće vrijednosti.
    Koristi Mann-Whitney U.
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


def pearson(x, y):
    """Pearsonov r i p-vrijednost (t-aproksimacija)."""
    n = len(x)
    mx, my = statistics.mean(x), statistics.mean(y)
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if sx == 0 or sy == 0:
        return 0.0, 1.0
    r = sum((x[i] - mx) * (y[i] - my) for i in range(n)) / (sx * sy)
    r = max(-1.0, min(1.0, r))
    if abs(r) == 1.0:
        return r, 0.0
    t = r * math.sqrt((n - 2) / (1 - r * r))
    p = math.erfc(abs(t) / math.sqrt(2))   # normalna aproksimacija (dobra za n>30)
    return r, p


def spearman(x, y):
    """Spearmanov rho i p-vrijednost."""
    def rank(data):
        order = sorted(range(len(data)), key=lambda i: data[i])
        ranks = [0.0] * len(data)
        i = 0
        while i < len(data):
            j = i
            while j < len(data) and data[order[j]] == data[order[i]]:
                j += 1
            avg = (i + j + 1) / 2.0
            for k in range(i, j):
                ranks[order[k]] = avg
            i = j
        return ranks
    return pearson(rank(x), rank(y))


def gaussian_elim(A, b):
    """Rješava sustav Ax=b Gaussovom eliminacijom s parcijalnim pivotiranjem."""
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[pivot] = M[pivot], M[col]
        if abs(M[col][col]) < 1e-12:
            continue
        for row in range(col + 1, n):
            f = M[row][col] / M[col][col]
            for j in range(col, n + 1):
                M[row][j] -= f * M[col][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))
        if abs(M[i][i]) > 1e-12:
            x[i] /= M[i][i]
    return x


def ols_residuals(y, *controls):
    """OLS regresija y na kontrolnim varijablama; vraća rezidualne."""
    n = len(y)
    k = 1 + len(controls)
    X = [[1.0] + [controls[j][i] for j in range(len(controls))] for i in range(n)]
    XtX = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    Xty = [sum(X[i][a] * y[i] for i in range(n)) for a in range(k)]
    beta = gaussian_elim(XtX, Xty)
    y_hat = [sum(beta[j] * X[i][j] for j in range(k)) for i in range(n)]
    return [y[i] - y_hat[i] for i in range(n)]


def partial_correlation(x, y, *controls):
    """Pearsonova parcijalna korelacija r(x,y|controls) i p-vrijednost."""
    ex = ols_residuals(x, *controls)
    ey = ols_residuals(y, *controls)
    return pearson(ex, ey)


def interpret_r(r, p, label=""):
    strength = (
        "zanemariva" if abs(r) < 0.1 else
        "slaba"      if abs(r) < 0.3 else
        "umjerena"   if abs(r) < 0.5 else
        "jaka"       if abs(r) < 0.7 else
        "vrlo jaka"
    )
    direction = "pozitivna" if r > 0 else "negativna"
    sig = "značajna" if p < 0.05 else "NIJE značajna"
    p_str = f"{p:.4f}" if p >= 0.0001 else f"{p:.2e}"
    return f"  r = {r:+.3f}  p = {p_str}  → {strength} {direction} korelacija, {sig}"


def interpret_vda(a):
    if a > 0.71:   return "veliki efekt"
    if a > 0.64:   return "srednji efekt"
    if a > 0.56:   return "mali efekt"
    return "zanemariv efekt"


# ============================================================
#  A) TOČKASTA ANALIZA - KS test + VD-a
# ============================================================

def run_point_analysis():
    print("=" * 65)
    print("A) TOČKASTA ANALIZA: ceste oko naselja vs. nasumičnih točaka")
    print("=" * 65)

    s_layer = get_layer(SETTLEMENTS_LAYER)
    r_layer = get_layer(RANDOM_LAYER)

    for field, label in [(ROAD_FIELD_1KM, "1 km"), (ROAD_FIELD_2KM, "2 km")]:
        s_vals = read_field(s_layer, field)
        r_vals = read_field(r_layer, field)

        if not s_vals or not r_vals:
            print(f"  GREŠKA: Nema podataka za polje '{field}'")
            continue

        D, p_ks = ks_test_2sample(s_vals, r_vals)
        a = vda(s_vals, r_vals)

        p_str = f"{p_ks:.4f}" if p_ks >= 0.0001 else f"{p_ks:.2e}"

        print(f"\nRadius {label}  (polje: {field})")
        print(f"  N:                  {len(s_vals)} naselja  /  {len(r_vals)} nasumičnih")
        print(f"  Medijan naselja:    {statistics.median(s_vals):.1f} m")
        print(f"  Medijan nasumičnih: {statistics.median(r_vals):.1f} m")
        print(f"  KS D = {D:.4f},  p = {p_str}")
        print(f"  VD-a = {a:.3f}  → {interpret_vda(a)}")

        if p_ks < 0.05:
            if a > 0.5:
                print(f"  → Prava naselja imaju VIŠE cesta u radijusu {label} (sampling bias moguć)")
            else:
                print(f"  → Prava naselja imaju MANJE cesta u radijusu {label}")
        else:
            print(f"  → Nema statistički značajne razlike")

    print()


# ============================================================
#  B) GRID ANALIZA - Pearson + Spearman korelacija
# ============================================================

def run_grid_analysis():
    print("=" * 65)
    print("B) GRID ANALIZA: korelacija gustoće cesta i nalazišta")
    print("=" * 65)

    grid = get_layer(GRID_LAYER)

    road_vals, sett_vals = [], []
    n_skipped = 0
    for feat in grid.getFeatures():
        r = feat[GRID_ROAD_DENSITY]
        s = feat[GRID_SETTLEMENT_DENSITY]
        if r is None or s is None or r != r or s != s:
            n_skipped += 1
            continue
        road_vals.append(float(r))
        sett_vals.append(float(s))

    n = len(road_vals)
    print(f"\n  Ćelije s podacima: {n}  (preskočeno null: {n_skipped})")

    if n < 10:
        print("  GREŠKA: Premalo ćelija za korelaciju.")
        return

    pr, pp = pearson(road_vals, sett_vals)
    sr, sp = spearman(road_vals, sett_vals)

    print(f"\n  Pearson:  {interpret_r(pr, pp)}")
    print(f"  Spearman: {interpret_r(sr, sp)}")

    print()
    if abs(pr - sr) > 0.15:
        print("  NAPOMENA: Pearson i Spearman se značajno razlikuju →")
        print("    distribucija nije normalna ili postoje outlieri.")
        print("    Osloni se na Spearman kao robusniji rezultat.")
    else:
        print("  Pearson i Spearman su konzistentni → rezultat je stabilan.")

    print()
    if (pp < 0.05 and sr > 0.3) or (sp < 0.05 and sr > 0.3):
        print("  INTERPRETACIJA: Postoji pozitivna korelacija između gustoće")
        print("  cesta i gustoće nalazišta. Ovo MOŽE značiti:")
        print("    (1) Sampling bias: bolje istražena područja uz ceste")
        print("    (2) Stvarna veza: ceste prate stare komunikacijske pravce")
        print("        koji su bili atraktivni i u neolitiku")
        print("    → Za razlikovanje ovih scenarija potrebna je parcijalna")
        print("      korelacija s kontrolnom varijablom (npr. nagib terena).")
    elif pp >= 0.05 and sp >= 0.05:
        print("  INTERPRETACIJA: Nema značajne korelacije između gustoće cesta")
        print("  i gustoće nalazišta → sampling bias nije dominantan faktor.")
    print()


# ============================================================
#  C1) PREPROCESSING - dodaj % mocvarnog tla na grid
# ============================================================

def add_wetsoil_to_grid():
    """
    Za svaku ćeliju grida izračuna koliki % površine pokrivaju
    Gleysoli + Fluvisoli i doda to kao atribut GRID_WETSOIL_FIELD.
    Pokreni jednom prije run_partial_correlation().
    """
    import processing
    from qgis.core import QgsField
    from PyQt5.QtCore import QVariant

    grid  = get_layer(GRID_LAYER)
    raster = get_layer(SOIL_RASTER_LAYER)

    if grid.crs() != raster.crs():
        print("UPOZORENJE: CRS grida i rastera se ne podudaraju!")

    print("Računam % mocvarnog tla po ćelijama grida...")

    prefix = "_wsoil_"
    hist = processing.run("native:zonalhistogram", {
        'INPUT_RASTER':  raster,
        'RASTER_BAND':   1,
        'INPUT_VECTOR':  grid,
        'COLUMN_PREFIX': prefix,
        'OUTPUT':        'memory:wsoil_hist',
    })['OUTPUT']

    # Pronađi stupce za Gleysol i Fluvisol
    hist_fields = [f.name() for f in hist.fields()]
    def find_col(target_val):
        for col in hist_fields:
            if col.startswith(prefix):
                try:
                    if int(float(col[len(prefix):])) == target_val:
                        return col
                except ValueError:
                    pass
        return None

    gl_col  = find_col(GLEYSOLS_VALUE)
    fl_col  = find_col(FLUVISOLS_VALUE)
    all_soil_cols = [c for c in hist_fields if c.startswith(prefix)]

    print(f"  Gleysol stupac:  {gl_col}")
    print(f"  Fluvisol stupac: {fl_col}")

    if not gl_col and not fl_col:
        print("  GREŠKA: Nisu pronađeni ni Gleysol ni Fluvisol u gridu.")
        return

    # Dodaj atribut na grid
    grid.startEditing()
    if grid.fields().indexFromName(GRID_WETSOIL_FIELD) == -1:
        grid.addAttribute(QgsField(GRID_WETSOIL_FIELD, QVariant.Double))
    grid.updateFields()
    idx = grid.fields().indexFromName(GRID_WETSOIL_FIELD)

    fid_list = [f.id() for f in grid.getFeatures()]
    for grid_fid, hist_feat in zip(fid_list, hist.getFeatures()):
        gl_cnt  = float(hist_feat[gl_col])  if gl_col  and hist_feat[gl_col]  is not None else 0.0
        fl_cnt  = float(hist_feat[fl_col])  if fl_col  and hist_feat[fl_col]  is not None else 0.0
        total   = sum(float(hist_feat[c]) for c in all_soil_cols if hist_feat[c] is not None)
        pct = 100.0 * (gl_cnt + fl_cnt) / total if total > 0 else 0.0
        grid.changeAttributeValue(grid_fid, idx, round(pct, 4))

    grid.commitChanges()
    print(f"  Atribut '{GRID_WETSOIL_FIELD}' dodan na grid.")
    print()


# ============================================================
#  C2) PARCIJALNA KORELACIJA (ceste ↔ naselja | % mocvara)
# ============================================================

def run_partial_correlation():
    print("=" * 65)
    print("C) PARCIJALNA KORELACIJA: ceste ↔ naselja | % mocvarnog tla")
    print("=" * 65)

    grid = get_layer(GRID_LAYER)

    road_vals, sett_vals, wet_vals = [], [], []
    n_skip = 0
    for feat in grid.getFeatures():
        r  = feat[GRID_ROAD_DENSITY]
        s  = feat[GRID_SETTLEMENT_DENSITY]
        w  = feat[GRID_WETSOIL_FIELD]
        if any(v is None or v != v for v in [r, s, w]):
            n_skip += 1
            continue
        road_vals.append(float(r))
        sett_vals.append(float(s))
        wet_vals.append(float(w))

    n = len(road_vals)
    print(f"\n  Ćelije s kompletnim podacima: {n}  (preskočeno: {n_skip})")

    if n < 15:
        print("  GREŠKA: Premalo ćelija. Provjeri je li add_wetsoil_to_grid() pokrenut.")
        return

    # Obična korelacija (referenca)
    r0, p0 = spearman(road_vals, sett_vals)
    # Parcijalna korelacija kontrolirajući za % mocvare
    rp, pp = partial_correlation(road_vals, sett_vals, wet_vals)

    print(f"\n  Spearman r (bez kontrole):       {interpret_r(r0, p0)}")
    print(f"  Parcijalna r (| % mocvara):      {interpret_r(rp, pp)}")
    print()

    delta = abs(r0) - abs(rp)
    print(f"  Pad apsolutne korelacije: |r| {abs(r0):.3f} → {abs(rp):.3f}  (Δ = {delta:.3f})")
    print()

    if delta > 0.1 and pp >= 0.05:
        print("  INTERPRETACIJA: Korelacija ceste↔naselja nestaje kad kontroliramo")
        print("  za % mocvarnog tla → KRAJOBRAZ (mocvare) objašnjava praznine,")
        print("  a ne sampling bias. Rupe u distribuciji su stvarne.")
    elif delta > 0.1 and pp < 0.05:
        print("  INTERPRETACIJA: Mocvara djelomično objašnjava vezu, ali korelacija")
        print("  ceste↔naselja ostaje značajna → OBOJE: i krajobraz i sampling bias")
        print("  doprinose prazninama u distribuciji.")
    elif delta <= 0.1 and pp < 0.05:
        print("  INTERPRETACIJA: Mocvara ne mijenja vezu ceste↔naselja →")
        print("  sampling bias je dominantan, krajobraz ga ne objašnjava.")
    else:
        print("  INTERPRETACIJA: Nije pronađena značajna veza ni bez ni s kontrolom.")
    print()


# ============================================================
#  D) GENERIRANJE CESTE-BIASED RANDOM TOČAKA
# ============================================================
#
# Generira N točaka čija je vjerojatnost smještaja u ćeliji grida
# proporcionalna gustoći cesta u toj ćeliji.
# Interpretacija: "gdje bi nasumični istraživač pronašao nalazišta
# kad bi mu pristupačnost bila jedini faktor".
#
# Workflow:
#   1. Pokreni generate_road_biased_random()  → dodaje novi sloj u QGIS
#   2. Pokreni vode.py i ceste.py s tim novim slojem umjesto originalnog
#   3. Usporedi rezultate:
#        prava vs. skroz random  → što smo već imali
#        prava vs. ceste-biased  → nova usporedba
#
# Ako prava ≈ ceste-biased  → road bias potpuno objašnjava distribuciju
# Ako prava ≠ ceste-biased  → postoji dodatna prostorna preferenca

BIASED_N_POINTS    = 274           # koliko točaka generirati (= broj pravih naselja)
BIASED_LAYER_NAME  = "random_ceste_biased"
BIASED_OUTPUT_PATH = r"D:\arheologija\transformirani_slojevi\random_ceste_biased.gpkg"


def generate_road_biased_random():
    import random, bisect
    from qgis.core import (
        QgsVectorLayer, QgsFeature, QgsGeometry, QgsPointXY,
        QgsField, QgsFields, QgsVectorFileWriter, QgsWkbTypes,
    )
    from PyQt5.QtCore import QVariant

    grid = get_layer(GRID_LAYER)

    print("Čitam gustoće cesta po ćelijama...")
    cells = []
    for feat in grid.getFeatures():
        density = feat[GRID_ROAD_DENSITY]
        if density is None or density != density or float(density) <= 0:
            continue
        cells.append((feat.geometry(), float(density)))

    if not cells:
        print("GREŠKA: Nema ćelija s podacima o gustoći cesta.")
        return

    # Kumulativne vjerojatnosti proporcionalne gustoći cesta
    total   = sum(d for _, d in cells)
    cumprobs = []
    cum = 0.0
    for _, d in cells:
        cum += d / total
        cumprobs.append(cum)

    print(f"  {len(cells)} ćelija s cestama, ukupna gustoća = {total:.1f}")
    print(f"  Generiram {BIASED_N_POINTS} točaka...")

    # Ukloni stari fajl ako postoji
    if os.path.exists(BIASED_OUTPUT_PATH):
        os.remove(BIASED_OUTPUT_PATH)

    fields = QgsFields()
    fields.append(QgsField("id", QVariant.Int))

    writer = QgsVectorFileWriter(
        BIASED_OUTPUT_PATH, "UTF-8", fields,
        QgsWkbTypes.Point, grid.crs(), "GPKG",
    )

    generated = 0
    attempts  = 0
    max_attempts = BIASED_N_POINTS * 50

    while generated < BIASED_N_POINTS and attempts < max_attempts:
        attempts += 1

        # Odaberi ćeliju weighted po gustoći cesta
        r   = random.random()
        idx = bisect.bisect_left(cumprobs, r)
        if idx >= len(cells):
            idx = len(cells) - 1
        cell_geom = cells[idx][0]

        # Slučajna točka unutar bounding boxa ćelije
        bbox = cell_geom.boundingBox()
        for _ in range(20):
            x  = random.uniform(bbox.xMinimum(), bbox.xMaximum())
            y  = random.uniform(bbox.yMinimum(), bbox.yMaximum())
            pt = QgsGeometry.fromPointXY(QgsPointXY(x, y))
            if cell_geom.contains(pt):
                feat = QgsFeature()
                feat.setGeometry(pt)
                feat.setAttributes([generated + 1])
                writer.addFeature(feat)
                generated += 1
                break

    del writer

    print(f"  Generirano {generated} točaka u {attempts} pokušaja.")
    print(f"  Spremljeno: {BIASED_OUTPUT_PATH}")

    # Učitaj u QGIS projekt
    layer = QgsVectorLayer(BIASED_OUTPUT_PATH, BIASED_LAYER_NAME, "ogr")
    if layer.isValid():
        from qgis.core import QgsProject
        QgsProject.instance().addMapLayer(layer)
        print(f"  Sloj '{BIASED_LAYER_NAME}' dodan u projekt.")
    else:
        print("  UPOZORENJE: Ne mogu učitati generirani sloj.")
    print()


# ============================================================
#  POKRENI
# ============================================================
# run_point_analysis()
# run_grid_analysis()

'''
=================================================================
A) TOČKASTA ANALIZA: ceste oko naselja vs. nasumičnih točaka
=================================================================

Radius 1 km  (polje: ceste_1km_buffer_sum_LENGTH)
  N:                  274 naselja  /  274 nasumičnih
  Medijan naselja:    2889.6 m
  Medijan nasumičnih: 632.4 m
  KS D = 0.3577,  p = 1.20e-15
  VD-a = 0.723  → veliki efekt
  → Prava naselja imaju VIŠE cesta u radijusu 1 km (sampling bias moguć)

Radius 2 km  (polje: ceste_2km_buffer_sum_LENGTH)
  N:                  274 naselja  /  274 nasumičnih
  Medijan naselja:    9997.4 m
  Medijan nasumičnih: 6331.8 m
  KS D = 0.2883,  p = 2.56e-10
  VD-a = 0.705  → srednji efekt
  → Prava naselja imaju VIŠE cesta u radijusu 2 km (sampling bias moguć)

=================================================================
B) GRID ANALIZA: korelacija gustoće cesta i nalazišta
=================================================================

  Ćelije s podacima: 164  (preskočeno null: 0)

  Pearson:    r = +0.372  p = 3.35e-07  → umjerena pozitivna korelacija, značajna
  Spearman:   r = +0.454  p = 9.08e-11  → umjerena pozitivna korelacija, značajna

  Pearson i Spearman su konzistentni → rezultat je stabilan.

  INTERPRETACIJA: Postoji pozitivna korelacija između gustoće
  cesta i gustoće nalazišta. Ovo MOŽE značiti:
    (1) Sampling bias: bolje istražena područja uz ceste
    (2) Stvarna veza: ceste prate stare komunikacijske pravce
        koji su bili atraktivni i u neolitiku
    → Za razlikovanje ovih scenarija potrebna je parcijalna
      korelacija s kontrolnom varijablom (npr. nagib terena).
'''
# add_wetsoil_to_grid()   # <- pokreni jednom da dodaš pct_mocvara na grid

# run_partial_correlation()  # <- pokreni nakon add_wetsoil_to_grid()

'''
=================================================================
C) PARCIJALNA KORELACIJA: ceste ↔ naselja | % mocvarnog tla
=================================================================

  Ćelije s kompletnim podacima: 164  (preskočeno: 0)

  Spearman r (bez kontrole):         r = +0.454  p = 9.08e-11  → umjerena pozitivna korelacija, značajna
  Parcijalna r (| % mocvara):        r = +0.391  p = 6.14e-08  → umjerena pozitivna korelacija, značajna

  Pad apsolutne korelacije: |r| 0.454 → 0.391  (Δ = 0.062)

  INTERPRETACIJA: Mocvara ne mijenja vezu ceste↔naselja →
  sampling bias je dominantan, krajobraz ga ne objašnjava.
'''
generate_road_biased_random()