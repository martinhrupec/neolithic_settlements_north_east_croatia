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

import math, statistics, os

# QGIS je dostupan samo unutar QGIS Python okruženja. Izvan njega (npr. običan
# Python za crtanje grafova iz CSV-a) import se preskače, pa rade plots_from_csv()
# i ostale čiste funkcije, dok QGIS-funkcije jave jasnu grešku ako se pozovu.
try:
    from qgis.core import QgsProject
except ModuleNotFoundError:
    QgsProject = None

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
# Skripta ce izracunati % celije koji je Gleysol ili Fluvisol i dodati ga na grid.
# Mora biti isti WRB raster kao u tlo.py.
SOIL_RASTER_LAYER  = "tipovi_tla"   # <- naziv rasterskog sloja tla
GLEYSOLS_VALUE     = 12             # <- rasterska vrijednost Gleysola
FLUVISOLS_VALUE    = 11             # <- rasterska vrijednost Fluvisola
GRID_WETSOIL_FIELD = "pct_mocvara"  # <- naziv novog atributa koji se dodaje na grid

# --- D) Parcijalna korelacija s terenom (TRI i/ili nadmorska visina) ---
# Pokretanje:
#   1. add_terrain_mean_to_grid("tri_raster",  "mean_tri",  1)
#   2. add_terrain_mean_to_grid("dem_raster",  "mean_elev", 1)
#   3. run_partial_correlation_terrain()
#
# Nazive slojeva prilagodi svom QGIS projektu:
TRI_RASTER_LAYER   = "TRI"      # <- naziv TRI rastera u projektu
DEM_RASTER_LAYER   = "nadmorska_visina"      # <- naziv DEM rastera u projektu
GRID_TRI_FIELD     = "mean_tri"     # <- atribut koji ce biti dodan na grid
GRID_ELEV_FIELD    = "mean_elev"    # <- atribut koji ce biti dodan na grid

# ============================================================
#  POMOĆNE FUNKCIJE - STATISTIKA
# ============================================================

def get_layer(name):
    if QgsProject is None:
        raise RuntimeError(
            "QGIS nije dostupan — ova funkcija radi samo u QGIS Python konzoli. "
            "Izvan QGIS-a koristi plots_from_csv() na izvezenom CSV-u.")
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


def rankdata(data):
    """Rangovi vrijednosti s prosječnim rangom za vezane vrijednosti (ties)."""
    order = sorted(range(len(data)), key=lambda i: data[i])
    ranks = [0.0] * len(data)
    i = 0
    while i < len(data):
        j = i
        while j < len(data) and data[order[j]] == data[order[i]]:
            j += 1
        avg = (i + j + 1) / 2.0           # prosjek rangova vezane skupine
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def spearman(x, y):
    """Spearmanov rho i p-vrijednost (= Pearson na rangovima)."""
    return pearson(rankdata(x), rankdata(y))


def _linfit(x, y):
    """Najbolji pravac (OLS): vraća (a, b) za y = b*x + a."""
    n = len(x)
    mx, my = statistics.mean(x), statistics.mean(y)
    sxx = sum((xi - mx) ** 2 for xi in x)
    sxy = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    b = sxy / sxx if sxx else 0.0
    a = my - b * mx
    return a, b


# --- zajednička paleta i pomoćne funkcije za grafove ---
_C_PT  = "#2c7fb8"   # točke (sirove vrijednosti)
_C_PTR = "#31a354"   # točke (rangovi)
_C_RES = "#d7301f"   # reziduali


def _save_or_show(fig, save_path):
    import matplotlib.pyplot as plt
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Graf spremljen: {save_path}")
        plt.close(fig)
    else:
        plt.show()
    return fig


def plot_total_variation(x, y, xlabel="X", ylabel="Y",
                         title="Ukupna varijabilnost (nazivnik u r²)",
                         save_path=None, ylim=None):
    """
    Graf 1 — vodoravni pravac y = ȳ + odstupanja svake točke od prosjeka.
    Σ(y−ȳ)² je NAZIVNIK u r² ("najlošija moguća prilagodba"). Prati stvarne podatke.
    """
    import matplotlib.pyplot as plt
    n  = len(x)
    my = statistics.mean(y)
    ss_tot = sum((yi - my) ** 2 for yi in y)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i in range(n):
        ax.plot([x[i], x[i]], [y[i], my], ls=":", color=_C_RES, lw=0.9, zorder=2)
    ax.scatter(x, y, s=28, color=_C_PT, edgecolor="white", lw=0.4, zorder=3)
    ax.axhline(my, color="black", lw=2, zorder=1, label=f"y = ȳ = {my:.2f}")
    ax.set_title(f"{title}\n$\\Sigma(y-\\bar{{y}})^2$ = {ss_tot:.1f}")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=9)
    if ylim:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_regression(x, y, xlabel="X", ylabel="Y",
                    title="Pearsonova korelacija",
                    save_path=None, ylim=None):
    """
    Graf 2 — najbolji pravac + reziduali do pravca. Σ(y−ŷ)² je BROJNIK u r².
    Ispisuje Pearsonov r i r². Prati stvarne podatke.
    """
    import matplotlib.pyplot as plt
    n  = len(x)
    my = statistics.mean(y)
    a, b = _linfit(x, y)
    y_hat = [a + b * xi for xi in x]
    ss_tot = sum((yi - my) ** 2 for yi in y)
    ss_res = sum((y[i] - y_hat[i]) ** 2 for i in range(n))
    r, _ = pearson(x, y)
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    xs = [min(x), max(x)]
    for i in range(n):
        ax.plot([x[i], x[i]], [y[i], y_hat[i]], ls=":", color=_C_RES, lw=0.9, zorder=2)
    ax.scatter(x, y, s=28, color=_C_PT, edgecolor="white", lw=0.4, zorder=3)
    ax.axhline(my, color="gray", lw=1, ls="--", alpha=0.6, zorder=1)
    ax.plot(xs, [a + b * xs[0], a + b * xs[1]], color="black", lw=2, zorder=4,
            label=f"y = {b:.3f}·x + {a:.2f}")
    ax.set_title(f"{title}\n$\\Sigma(y-\\hat{{y}})^2$ = {ss_res:.1f}   "
                 f"r = {r:+.3f}   r² = {r2:.3f}")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=9)
    if ylim:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_spearman_ranks(x, y, xlabel="X", ylabel="Y",
                        title="Spearman (rangovi)",
                        save_path=None):
    """
    Graf 3 — scatter rang(x) vs rang(y) + pravac + reziduali na rangovima.
    Spearman ne traži linearan odnos sirovih vrijednosti: monotoni odnos
    postaje (približno) pravac tek u prostoru rangova. Prati stvarne podatke.
    """
    import matplotlib.pyplot as plt
    n = len(x)
    rx, ry = rankdata(x), rankdata(y)
    ra, rb = _linfit(rx, ry)
    rs, _ = spearman(x, y)
    r_hat = [ra + rb * xi for xi in rx]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i in range(n):
        ax.plot([rx[i], rx[i]], [ry[i], r_hat[i]], ls=":", color=_C_RES, lw=0.7, zorder=2)
    ax.scatter(rx, ry, s=28, color=_C_PTR, edgecolor="white", lw=0.4, zorder=3)
    rxs = [min(rx), max(rx)]
    ax.plot(rxs, [ra + rb * rxs[0], ra + rb * rxs[1]], color="black", lw=2, zorder=4)
    ax.set_title(f"{title}\nr_s = {rs:+.3f}")
    ax.set_xlabel(f"rang — {xlabel}")
    ax.set_ylabel(f"rang — {ylabel}")
    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_distribution(values, label="gustoća nalazišta (/km²)",
                      title="Distribucija", save_path=None):
    """
    Dijagnostika raspodjele jedne varijable:
      - GORE: histogram + normalna krivulja (ista μ i σ) za vizualnu usporedbu
              koliko podaci odstupaju od normalne raspodjele.
      - DOLJE: boxplot (pravokutnik = IQR, crta = medijan, brkovi = 1.5·IQR),
               a točke izvan brkova su OUTLIERI.
    Ispisuje μ, medijan i koeficijent asimetrije (skewness).
    """
    import matplotlib.pyplot as plt

    n  = len(values)
    mu = statistics.mean(values)
    med = statistics.median(values)
    sd = statistics.pstdev(values)
    # koeficijent asimetrije (Fisher): >0 desno zakošeno, ~0 simetrično
    skew = (sum((v - mu) ** 3 for v in values) / n) / (sd ** 3) if sd else 0.0

    fig, (ax_h, ax_b) = plt.subplots(
        2, 1, figsize=(7, 5.5), sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05})

    # --- histogram + normalna krivulja ---
    nbins = max(10, min(30, int(math.sqrt(n))))
    ax_h.hist(values, bins=nbins, density=True, color=_C_PT,
              edgecolor="white", alpha=0.85, zorder=2)
    if sd > 0:
        lo, hi = min(values), max(values)
        step = (hi - lo) / 200 or 1
        xs = [lo + i * step for i in range(201)]
        norm = [math.exp(-((xi - mu) ** 2) / (2 * sd * sd)) /
                (sd * math.sqrt(2 * math.pi)) for xi in xs]
        ax_h.plot(xs, norm, color="black", lw=2, zorder=3,
                  label="normalna (ista μ, σ)")
    ax_h.axvline(mu,  color=_C_RES, lw=1.5, ls="-",  label=f"μ = {mu:.3f}")
    ax_h.axvline(med, color="green", lw=1.5, ls="--", label=f"medijan = {med:.3f}")
    ax_h.set_ylabel("gustoća (density)")
    ax_h.legend(fontsize=8, loc="upper right")
    ax_h.set_title(f"{title}\nasimetrija (skew) = {skew:+.2f}   "
                   f"({'desno zakošeno' if skew > 0.5 else 'lijevo zakošeno' if skew < -0.5 else 'približno simetrično'})")

    # --- boxplot s outlierima ---
    ax_b.boxplot(values, vert=False, widths=0.6,
                 flierprops=dict(marker="o", markerfacecolor=_C_RES,
                                 markeredgecolor=_C_RES, markersize=4, alpha=0.6),
                 medianprops=dict(color="green", lw=2),
                 boxprops=dict(color="black"),
                 whiskerprops=dict(color="black"),
                 capprops=dict(color="black"))
    ax_b.set_yticks([])
    ax_b.set_xlabel(label)
    ax_b.annotate("crvene točke = outlieri", xy=(0.99, 0.1),
                  xycoords="axes fraction", ha="right", fontsize=8, color=_C_RES)

    # tight_layout ne radi dobro sa sharex+gridspec; bbox='tight' u savefig to rješava
    return _save_or_show(fig, save_path)


def plot_correlation(x, y,
                     xlabel="gustoća cesta (km/km²)",
                     ylabel="gustoća nalazišta (/km²)",
                     out_dir=None, prefix="ceste_korelacija"):
    """
    Generira TRI ZASEBNA grafa iz stvarnih podataka (x, y):
       <prefix>_1_ukupna_varijabilnost.png   (nazivnik u r²)
       <prefix>_2_regresija_reziduali.png    (brojnik u r², Pearson r)
       <prefix>_3_spearman_rangovi.png       (Spearman na rangovima)

    Ako je out_dir zadan, svaki se graf sprema u svoj PNG; inače se prikazuju.
    Y-osi grafova 1 i 2 su izjednačene radi poštene vizualne usporedbe.
    """
    ymin, ymax = min(y), max(y)
    pad = 0.06 * (ymax - ymin or 1)
    ylim = (ymin - pad, ymax + pad)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)   # stvori izlaznu mapu ako ne postoji

    def _path(name):
        if not out_dir:
            return None
        return os.path.join(out_dir, f"{prefix}_{name}.png")

    plot_total_variation(x, y, xlabel, ylabel,
                         save_path=_path("1_ukupna_varijabilnost"), ylim=ylim)
    plot_regression(x, y, xlabel, ylabel,
                    save_path=_path("2_regresija_reziduali"), ylim=ylim)
    plot_spearman_ranks(x, y, xlabel, ylabel,
                        save_path=_path("3_spearman_rangovi"))


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


def partial_correlation_spearman(x, y, *controls):
    """
    Spearmanova parcijalna korelacija r_s(x,y|controls) i p-vrijednost.

    Definira se kao Pearsonova parcijalna korelacija izračunata na RANGOVIMA
    svih varijabli (x, y i svih kontrolnih). Time je dosljedna s referentnom
    Spearman korelacijom i prikladna za nenormalne/zakošene podatke.
    """
    rx, ry = rankdata(x), rankdata(y)
    rcontrols = [rankdata(c) for c in controls]
    return partial_correlation(rx, ry, *rcontrols)


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

def run_grid_analysis(plot=False, out_dir=None):
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

    if plot:
        plot_correlation(road_vals, sett_vals, out_dir=out_dir)


# ============================================================
#  B2) IZVOZ GRID PODATAKA U CSV  (pokreni jednom u QGIS-u)
# ============================================================

# Stupci koji se izvoze ako postoje na gridu (ostali se preskaču bez greške):
GRID_EXPORT_FIELDS = [
    (GRID_ROAD_DENSITY,       "gustoca_cesta_km_ceste_po_km2_grida"),
    (GRID_SETTLEMENT_DENSITY, "gustoca_nalazista_po_km2"),
    (GRID_WETSOIL_FIELD,      "pct_mocvara"),
    (GRID_TRI_FIELD,          "mean_tri"),
    (GRID_ELEV_FIELD,         "mean_elev"),
]

# Zadana putanja CSV-a (pokraj skripte) — odatle ga čita plots_from_csv().
# __file__ ne postoji ako se skripta zalijepi u QGIS konzolu → fallback na cwd.
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _SCRIPT_DIR = os.getcwd()
GRID_CSV_PATH = os.path.join(_SCRIPT_DIR, "ceste_grid_podaci.csv")


def export_grid_to_csv(path=GRID_CSV_PATH):
    """
    Izvuče vrijednosti svih ćelija grida (gustoća cesta, gustoća nalazišta i
    kontrolne varijable ako su dodane) i spremi ih u CSV pokraj skripte.

    Pokreni JEDNOM u QGIS Python konzoli:
        export_grid_to_csv()

    Nakon toga grafove možeš generirati i izvan QGIS-a:
        plots_from_csv(out_dir=r"D:\\...\\grafovi")
    """
    import csv

    grid = get_layer(GRID_LAYER)

    # Mapiraj očišćen naziv (bez razmaka, mala slova) → STVARNI naziv polja.
    # Tako matchiranje radi i kad QGIS nazive vrati s nevidljivim razmakom i sl.
    name_map = {f.name().strip().lower(): f.name() for f in grid.fields()}

    def resolve(name):
        return name_map.get(name.strip().lower())

    # zadrži samo stupce koji stvarno postoje na gridu (s razriješenim nazivima)
    cols, missing = [], []
    for src, alias in GRID_EXPORT_FIELDS:
        actual = resolve(src)
        if actual is not None:
            cols.append((actual, alias))
        else:
            missing.append(src)
    if missing:
        print(f"  Napomena: preskačem nepostojeća polja: {missing}")

    road_f = resolve(GRID_ROAD_DENSITY)
    sett_f = resolve(GRID_SETTLEMENT_DENSITY)
    if road_f is None or sett_f is None:
        print("  GREŠKA: grid nema osnovna polja gustoće cesta/nalazišta.")
        print(f"  Tražim:  GRID_ROAD_DENSITY = '{GRID_ROAD_DENSITY}'")
        print(f"           GRID_SETTLEMENT_DENSITY = '{GRID_SETTLEMENT_DENSITY}'")
        print(f"  Sloj '{GRID_LAYER}' stvarno ima ova polja:")
        for f in grid.fields():
            print(f"     - '{f.name()}'")
        print("  → Uskladi konstante GRID_ROAD_DENSITY / GRID_SETTLEMENT_DENSITY")
        print("    (na vrhu skripte) s gornjim nazivima i pokreni ponovno.")
        return

    headers = [alias for _, alias in cols]
    rows, n_skip = [], 0
    for feat in grid.getFeatures():
        vals = [feat[src] for src, _ in cols]
        # preskoči ćeliju ako su gustoća cesta ili nalazišta null/NaN
        rd = feat[road_f]
        st = feat[sett_f]
        if rd is None or st is None or rd != rd or st != st:
            n_skip += 1
            continue
        rows.append([("" if v is None or v != v else float(v)) for v in vals])

    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        w.writerows(rows)

    print(f"  Izvezeno {len(rows)} ćelija (preskočeno {n_skip}) → {path}")
    print(f"  Stupci: {headers}")


def load_grid_csv(path=GRID_CSV_PATH):
    """Učita CSV koji je napravio export_grid_to_csv() → dict {stupac: [floatovi]}."""
    import csv
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = {h: [] for h in reader.fieldnames}
        for row in reader:
            for h in reader.fieldnames:
                v = row[h]
                cols[h].append(float(v) if v not in ("", None) else float("nan"))
    return cols


def plots_from_csv(path=GRID_CSV_PATH, out_dir=None):
    """
    Generira tri zasebna grafa korelacije iz izvezenog CSV-a — radi i izvan QGIS-a.
        plots_from_csv(out_dir=r"D:\\...\\grafovi")
    """
    data = load_grid_csv(path)
    x = data["gustoca_cesta_km_ceste_po_km2_grida"]
    y = data["gustoca_nalazista_po_km2"]
    # ukloni eventualne NaN parove
    pairs = [(a, b) for a, b in zip(x, y) if a == a and b == b]
    x = [a for a, _ in pairs]
    y = [b for _, b in pairs]
    print(f"  Učitano {len(x)} ćelija iz {path}")
    plot_correlation(x, y, out_dir=out_dir)

    def _dpath(name):
        return os.path.join(out_dir, name) if out_dir else None

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # distribucija odgovora (bitna za odabir Pearson/Spearman) + prediktora (info)
    plot_distribution(y, label="gustoća nalazišta (/km²)",
                      title="Distribucija gustoće nalazišta",
                      save_path=_dpath("ceste_distribucija_nalazista.png"))
    plot_distribution(x, label="gustoća cesta (km/km²)",
                      title="Distribucija gustoće cesta",
                      save_path=_dpath("ceste_distribucija_cesta.png"))


def dist_from_csv(column=GRID_SETTLEMENT_DENSITY, path=GRID_CSV_PATH,
                  label=None, title=None, save_path=None):
    """
    Učita JEDAN stupac iz izvezenog CSV-a i nacrta njegovu distribuciju
    (histogram + boxplot s outlierima). Radi i izvan QGIS-a.
        dist_from_csv(GRID_SETTLEMENT_DENSITY)              # prikaži na ekran
        dist_from_csv(GRID_ROAD_DENSITY, save_path="d.png") # spremi u file
    """
    data = load_grid_csv(path)
    if column not in data:
        print(f"  GREŠKA: stupac '{column}' ne postoji u CSV-u.")
        print(f"  Dostupni stupci: {list(data.keys())}")
        return
    vals = [v for v in data[column] if v == v]   # makni NaN
    plot_distribution(vals,
                      label=label or column,
                      title=title or f"Distribucija — {column}",
                      save_path=save_path)


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

    # Obična Spearman korelacija (referenca)
    r0, p0 = spearman(road_vals, sett_vals)
    # Spearmanova parcijalna korelacija kontrolirajući za % mocvare
    rp, pp = partial_correlation_spearman(road_vals, sett_vals, wet_vals)

    print(f"\n  Spearman r (bez kontrole):          {interpret_r(r0, p0)}")
    print(f"  Spearman parcijalna (| % mocvara):  {interpret_r(rp, pp)}")
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
#  D) DODAJ SREDNJU VRIJEDNOST RASTERA NA GRID (zonalna statistika)
# ============================================================

def add_terrain_mean_to_grid(raster_layer_name, grid_field_name, band=1):
    """
    Za svaku celiju grida izracuna srednju vrijednost piksel rastera
    (npr. TRI ili nadmorska visina) i doda je kao novi atribut grida.

    Pokretanje:
        add_terrain_mean_to_grid("tri_25m",  "mean_tri",  1)
        add_terrain_mean_to_grid("dem_25m",  "mean_elev", 1)

    Nakon toga pokreni run_partial_correlation_terrain().
    """
    import processing
    from qgis.core import QgsField
    from PyQt5.QtCore import QVariant

    grid   = get_layer(GRID_LAYER)
    raster = get_layer(raster_layer_name)

    if grid.crs() != raster.crs():
        print(f"UPOZORENJE: CRS grida i rastera '{raster_layer_name}' se ne podudaraju!")

    print(f"Racunam zonalnu srednju vrijednost rastera '{raster_layer_name}'...")

    prefix = "_zt_"
    result = processing.run("native:zonalstatisticsfb", {
        "INPUT":         grid,
        "INPUT_RASTER":  raster,
        "RASTER_BAND":   band,
        "COLUMN_PREFIX": prefix,
        "STATISTICS":    [2],     # 2 = mean
        "OUTPUT":        "memory:terrain_zonal",
    })["OUTPUT"]

    mean_field = prefix + "mean"

    # Kopiraj vrijednosti natrag na originalni grid sloj
    grid.startEditing()
    if grid.fields().indexFromName(grid_field_name) == -1:
        grid.addAttribute(QgsField(grid_field_name, QVariant.Double))
    grid.updateFields()
    idx = grid.fields().indexFromName(grid_field_name)

    fid_list   = [f.id() for f in grid.getFeatures()]
    res_feats   = list(result.getFeatures())

    if len(fid_list) != len(res_feats):
        print(f"  GRESKA: broj znacajki grida ({len(fid_list)}) != "
              f"broj rezultata ({len(res_feats)})")
        grid.rollBack()
        return

    for gfid, rfeat in zip(fid_list, res_feats):
        val = rfeat[mean_field]
        if val is not None and val == val:
            grid.changeAttributeValue(gfid, idx, float(val))

    grid.commitChanges()
    print(f"  Atribut '{grid_field_name}' dodan na grid.")
    print()


# ============================================================
#  E) PARCIJALNA KORELACIJA S TERENOM  (TRI i visina)
# ============================================================

def run_partial_correlation_terrain():
    """
    Parcijalna Spearman korelacija: ceste <-> naselja | kontrolna varijabla terena.

    Testira se:
      (1) ceste <-> naselja | mean_tri   (kontrola: hrapavost terena)
      (2) ceste <-> naselja | mean_elev  (kontrola: nadmorska visina)
      (3) ceste <-> naselja | mean_tri + mean_elev  (obje zajedno)

    Interpretacija:
      Ako r pada >0.10 i postaje neznacajan: teren objasnjava ceste-nalazista vezu
        -> road bias je posredovan terenom, a ne cista istrazivacka pristranost
      Ako r pada malo (<0.05): teren ne objasnjava bias
        -> sampling bias je realan i neovisan o terenu

    Preduvjet: pokreni jednom:
        add_wetsoil_to_grid()                                      # pct_mocvara
        add_terrain_mean_to_grid(TRI_RASTER_LAYER,  GRID_TRI_FIELD,  1)
        add_terrain_mean_to_grid(DEM_RASTER_LAYER,  GRID_ELEV_FIELD, 1)
    """
    print("=" * 65)
    print("E) PARCIJALNA KORELACIJA: ceste <-> naselja | teren + mocvara")
    print("=" * 65)

    grid = get_layer(GRID_LAYER)

    road_vals, sett_vals, tri_vals, elev_vals, wet_vals = [], [], [], [], []
    n_skip = 0

    for feat in grid.getFeatures():
        r  = feat[GRID_ROAD_DENSITY]
        s  = feat[GRID_SETTLEMENT_DENSITY]
        t  = feat[GRID_TRI_FIELD]
        e  = feat[GRID_ELEV_FIELD]
        w  = feat[GRID_WETSOIL_FIELD]
        if any(v is None or v != v for v in [r, s, t, e, w]):
            n_skip += 1
            continue
        road_vals.append(float(r))
        sett_vals.append(float(s))
        tri_vals.append(float(t))
        elev_vals.append(float(e))
        wet_vals.append(float(w))

    _report_partial_terrain(road_vals, sett_vals, tri_vals, elev_vals, wet_vals, n_skip)


def _report_partial_terrain(road_vals, sett_vals, tri_vals, elev_vals, wet_vals,
                            n_skip=0):
    """Zajednicka analiza+ispis za parcijalnu korelaciju (koriste QGIS i CSV verzija)."""
    n = len(road_vals)
    print(f"\n  Celije s kompletnim podacima: {n}  (preskoceno: {n_skip})")

    if n < 15:
        print("  GRESKA: Premalo celija. Provjeri jesu li atributi dodani na grid/CSV.")
        return

    # Referentna korelacija (bez kontrole)
    r0, p0 = spearman(road_vals, sett_vals)

    print(f"\n  Referentna Spearman (bez kontrole): {interpret_r(r0, p0)}")
    print()

    scenarios = [
        ("| mocvara",                 [wet_vals]),
        ("| mean_tri",                [tri_vals]),
        ("| mean_elev",               [elev_vals]),
        ("| mean_tri + elev",         [tri_vals, elev_vals]),
        ("| mocvara + tri + elev",    [wet_vals, tri_vals, elev_vals]),
    ]

    for label, controls in scenarios:
        rp, pp = partial_correlation_spearman(road_vals, sett_vals, *controls)
        delta  = abs(r0) - abs(rp)
        print(f"  Spearman parcijalna {label:<24s}: {interpret_r(rp, pp)}")
        print(f"    Pad |r|: {abs(r0):.3f} -> {abs(rp):.3f}  (delta = {delta:.3f})")

        if delta > 0.10 and pp >= 0.05:
            print("    INTERPRETACIJA: Krajobraz objasnjava ceste<->nalazista vezu.")
            print("      Road bias je posredovan tipom krajolika, a ne istrazivackom")
            print("      pristranoscu. Korelacija je artefakt krajobraza.")
        elif delta > 0.10 and pp < 0.05:
            print("    INTERPRETACIJA: Krajobraz DJELOMICNO objasnjava vezu, no korelacija")
            print("      ostaje znacajna -> i krajobraz I sampling bias doprinose signalu.")
        elif delta <= 0.05 and pp < 0.05:
            print("    INTERPRETACIJA: Krajobraz ne objasnjava vezu ceste<->nalazista.")
            print("      Sampling bias je realan i NEZAVISAN od krajobraza.")
        else:
            print("    INTERPRETACIJA: Blagi pad, ali veza ostaje znacajna.")
        print()

    print("  SAVJET: Ako svi scenariji pokazuju mali pad (delta<0.05),")
    print("  to ucvrsuje zakljucak da je road bias istrazivacki, a ne krajobrazni.")
    print()


def partial_terrain_from_csv(path=GRID_CSV_PATH):
    """
    Parcijalna Spearman korelacija iz izvezenog CSV-a — radi IZVAN QGIS-a.
    CSV mora imati stupce: gustoca_cesta..., gustoca_nalazista..., pct_mocvara,
    mean_tri, mean_elev (tj. izvezen nakon add_wetsoil_to_grid + add_terrain...).
        partial_terrain_from_csv()
    """
    print("=" * 65)
    print("E) PARCIJALNA KORELACIJA (iz CSV-a): ceste <-> naselja | teren + mocvara")
    print("=" * 65)

    data = load_grid_csv(path)
    need = [GRID_ROAD_DENSITY, GRID_SETTLEMENT_DENSITY,
            GRID_WETSOIL_FIELD, GRID_TRI_FIELD, GRID_ELEV_FIELD]
    missing = [c for c in need if c not in data]
    if missing:
        print(f"  GRESKA: CSV nema potrebne stupce: {missing}")
        print(f"  Dostupni: {list(data.keys())}")
        print("  Ponovno izvezi CSV u QGIS-u nakon add_wetsoil_to_grid() i add_terrain_mean_to_grid().")
        return

    cols = [data[c] for c in need]
    road, sett, wet, tri, elev = [], [], [], [], []
    n_skip = 0
    for vals in zip(*cols):
        if any(v != v for v in vals):   # NaN check
            n_skip += 1
            continue
        r, s, w, t, e = vals
        road.append(r); sett.append(s); wet.append(w); tri.append(t); elev.append(e)

    _report_partial_terrain(road, sett, tri, elev, wet, n_skip)


# ============================================================
#  F) GENERIRANJE CESTE-BIASED RANDOM TOCAKA
# ============================================================
#
# Generira N tocaka cija je vjerojatnost smjestaja u celiji grida
# proporcionalna gustoci cesta u toj celiji.
# Interpretacija: "gdje bi nasumicni istrazivac pronasao nalazista
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

if QgsProject is not None:
    # Unutar QGIS-a: čitaj grid uživo, izvezi CSV (za kasnije) i nacrtaj grafove
    export_grid_to_csv()
    run_grid_analysis(plot=True, out_dir=r"C:\\Users\\Martin\\Desktop\\slike_za_diplomski")
else:
    # Izvan QGIS-a: nacrtaj iz prethodno izvezenog CSV-a (mora postojati)
    #plots_from_csv(out_dir=r"C:\\Users\\Martin\\Desktop\\slike_za_diplomski")
    _SLIKE_DIR = r"C:\Users\Martin\Desktop\slike_za_diplomski"
    os.makedirs(_SLIKE_DIR, exist_ok=True)
    #dist_from_csv(GRID_SETTLEMENT_DENSITY, label="gustoća nalazišta (/km²)",
    #              title="Distribucija gustoće nalazišta",
    #              save_path=os.path.join(_SLIKE_DIR, "ceste_distribucija_nalazista.png"))
    partial_terrain_from_csv()

# --- Novi workflow: parcijalna korelacija s terenom ---
# Korak 1: dodaj terenske atribute na grid (jedanput)
#   add_terrain_mean_to_grid(TRI_RASTER_LAYER,  GRID_TRI_FIELD,  1)
#   add_terrain_mean_to_grid(DEM_RASTER_LAYER,  GRID_ELEV_FIELD, 1)
# Korak 2: pokreni analizu
#   run_partial_correlation_terrain()

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
# generate_road_biased_random()
# add_terrain_mean_to_grid("TRI",  "mean_tri",  1)   # TRI raster
# add_terrain_mean_to_grid("nadmorska_visina",  "mean_elev",  1)   # DEM raster

#run_partial_correlation_terrain()

"""
=================================================================

E) PARCIJALNA KORELACIJA: ceste <-> naselja | teren
=================================================================

  Celije s kompletnim podacima: 164  (preskoceno: 0)

  Referentna Spearman (bez kontrole):   r = +0.454  p = 9.08e-11  → umjerena pozitivna korelacija, značajna

  Parcijalna r | mean_tri               :   r = +0.353  p = 1.58e-06  → umjerena pozitivna korelacija, značajna
    Pad |r|: 0.454 -> 0.353  (delta = 0.101)
    INTERPRETACIJA: Teren DJELOMICNO objasnjava vezu, no korelacija
      ostaje znacajna -> i teren I sampling bias doprinose signalu.

  Parcijalna r | mean_elev              :   r = +0.354  p = 1.40e-06  → umjerena pozitivna korelacija, značajna
    Pad |r|: 0.454 -> 0.354  (delta = 0.099)
    INTERPRETACIJA: Blagi pad, ali veza ostaje znacajna.

  Parcijalna r | mean_tri + elev        :   r = +0.353  p = 1.55e-06  → umjerena pozitivna korelacija, značajna
    Pad |r|: 0.454 -> 0.353  (delta = 0.101)
    INTERPRETACIJA: Teren DJELOMICNO objasnjava vezu, no korelacija
      ostaje znacajna -> i teren I sampling bias doprinose signalu.

  SAVJET: Ako svi scenariji pokazuju mali pad (delta<0.05),
  to ucvrsuje zakljucak da je road bias istrazivacki, a ne krajobrazni.


  ........................................................

  =================================================================
E) PARCIJALNA KORELACIJA (iz CSV-a): ceste <-> naselja | teren + mocvara
=================================================================

  Celije s kompletnim podacima: 164  (preskoceno: 0)

  Referentna Spearman (bez kontrole):   r = +0.454  p = 9.08e-11  → umjerena pozitivna korelacija, značajna

  Spearman parcijalna | mocvara               :   r = +0.472  p = 9.85e-12  → umjerena pozitivna korelacija, značajna
    Pad |r|: 0.454 -> 0.472  (delta = -0.018)
    INTERPRETACIJA: Krajobraz ne objasnjava vezu ceste<->nalazista.
      Sampling bias je realan i NEZAVISAN od krajobraza.

  Spearman parcijalna | mean_tri              :   r = +0.443  p = 3.19e-10  → umjerena pozitivna korelacija, značajna
    Pad |r|: 0.454 -> 0.443  (delta = 0.011)
    INTERPRETACIJA: Krajobraz ne objasnjava vezu ceste<->nalazista.
      Sampling bias je realan i NEZAVISAN od krajobraza.

  Spearman parcijalna | mean_elev             :   r = +0.459  p = 4.67e-11  → umjerena pozitivna korelacija, značajna
    Pad |r|: 0.454 -> 0.459  (delta = -0.006)
    INTERPRETACIJA: Krajobraz ne objasnjava vezu ceste<->nalazista.
      Sampling bias je realan i NEZAVISAN od krajobraza.

  Spearman parcijalna | mean_tri + elev       :   r = +0.434  p = 8.96e-10  → umjerena pozitivna korelacija, značajna
    Pad |r|: 0.454 -> 0.434  (delta = 0.020)
    INTERPRETACIJA: Krajobraz ne objasnjava vezu ceste<->nalazista.
      Sampling bias je realan i NEZAVISAN od krajobraza.

  Spearman parcijalna | mocvara + tri + elev  :   r = +0.446  p = 2.38e-10  → umjerena pozitivna korelacija, značajna
    Pad |r|: 0.454 -> 0.446  (delta = 0.008)
    INTERPRETACIJA: Krajobraz ne objasnjava vezu ceste<->nalazista.
      Sampling bias je realan i NEZAVISAN od krajobraza.

  SAVJET: Ako svi scenariji pokazuju mali pad (delta<0.05),
  to ucvrsuje zakljucak da je road bias istrazivacki, a ne krajobrazni.
"""


#export_grid_to_csv()