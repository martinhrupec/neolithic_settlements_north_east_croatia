# -*- coding: utf-8 -*-
"""
POZADINSKA DISTRIBUCIJA
========================
Računa distribuciju varijabli kroz CIJELI RASTERSKI SLOJ (pikselna razina)
kao referentnu pozadinsku raspodjelu za usporedbu s 8 tockastih slojeva.

STRUKTURA SKRIPTE:
  Sekcija 1 — QGIS izvoz (MODE = "export")
    export_background_vtt()   → pozadinska distribucija tipova tla
    export_background_sm()    → ista, klasificirana kao mocvarno/suho

  Sekcija 2 — Vizualizacija (MODE = "analyse", odkomentirati)
    plot_background_vtt()     → bar chart pozadine (isti stil kao res_vecinski_tip_tla)
    plot_background_sm()      → bar chart pozadine (isti stil kao res_suho_mocvarno)
    plot_comparison_vtt()     → složeni bar chart: pozadina + svih 8 slojeva (po polumjeru)
    plot_comparison_sm()      → isti za mocvarno/suho

Izlazni CSV-ovi idu u isti folder kao i 8-slojni CSVovi (vecinski_tip_tla_/csv_output/)
tako da ih analiza moze citati zajedno.

Izlazni PNG-ovi idu u res_output/pozadina/.

Promijeni MODE ispod, pa pokreni.
"""

import os
import csv

# ============================================================
#  KONSTANTE  (kopirane iz postojecih skripti)
# ============================================================

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

MOCVARNA_TLA = {"Gleysols", "Fluvisols", "Vertisols"}

SOIL_RASTER_LAYER = "tipovi_tla"

VTT_CSV_DIR   = r"C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna\vecinski_tip_tla_\csv_output"
RES_VTT_DIR   = r"C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna\res_output\vecinski_tip_tla"
RES_SM_DIR    = r"C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna\res_output\suho_mocvarno"
CFRAG_CSV_DIR = r"C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna\coarse_fragments_\csv_output"
PNG_OUT_DIR   = r"C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna\res_output\pozadina"

CFRAG_RASTER_LAYER = "coarse_fragments"

LAYERS = [
    "random_ceste_biased",
    "nasumicni_lokaliteti_umjetno_generirani",
    "neolitik_svi_odredeni",
    "neolitik_c_starcevacka",
    "neolitik_c_sop_kor_len",
    "kontinuirana_naselja",
    "samo_rani",
    "samo_srednji_kasni",
]
RADII = [100, 250, 500, 1000]

LAYER_LABELS = {
    "random_ceste_biased":                     "Random\n(pristranost cestama)",
    "nasumicni_lokaliteti_umjetno_generirani": "Random\n(potpuno nasumični)",
    "neolitik_svi_odredeni":                   "Neolitik\n(svi)",
    "neolitik_c_starcevacka":                  "Rani\nneolitik",
    "neolitik_c_sop_kor_len":                  "Srednji i\nkasni neolitik",
    "kontinuirana_naselja":                    "Kontinuirana\nnaselja",
    "samo_rani":                               "Isključivo\nrana faza",
    "samo_srednji_kasni":                      "Isključivo srednja i\nkasna faza",
    "__background__":                          "Pozadina\n(cijeli raster)",
}

# ============================================================
#  SEKCIJA 1 — QGIS IZVOZ
#  Pokretati u QGIS Python konzoli (MODE = "export")
# ============================================================

def export_background_vtt():
    """
    Cita distribuciju piksela rasterskog sloja tipovi_tla i
    exporta background_vtt.csv u VTT_CSV_DIR.
    Format: tip_tla, n_piksela, postotak
    """
    from qgis.core import QgsProject
    import processing

    layers = QgsProject.instance().mapLayersByName(SOIL_RASTER_LAYER)
    if not layers:
        raise ValueError(f"Sloj '{SOIL_RASTER_LAYER}' nije pronaden. Provjeri naziv.")
    raster = layers[0]

    result = processing.run("native:rasterlayeruniquevaluesreport", {
        'INPUT':        raster,
        'BAND':         1,
        'OUTPUT_TABLE': 'memory:vals',
    })
    table = result['OUTPUT_TABLE']

    rows = []
    total_pixels = 0
    for feat in table.getFeatures():
        val   = int(float(feat['value']))
        count = int(feat['count'])
        name  = SOIL_TYPES.get(val)
        if name is None:
            continue
        rows.append((name, count))
        total_pixels += count

    rows.sort(key=lambda r: r[1], reverse=True)

    os.makedirs(VTT_CSV_DIR, exist_ok=True)
    out_path = os.path.join(VTT_CSV_DIR, "background_vtt.csv")
    with open(out_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["tip_tla", "n_piksela", "postotak"])
        for name, count in rows:
            w.writerow([name, count, round(count / total_pixels * 100, 4)])

    print(f"Exportano: background_vtt.csv  ({len(rows)} tipova, {total_pixels:,} piksela ukupno)")
    for name, count in rows:
        print(f"  {name:<16} {count:>10,}  {count/total_pixels*100:>6.2f}%")


def export_background_sm():
    """
    Cita background_vtt.csv (mora vec postojati), klasificira na
    Mocvarno/Suho i exporta background_sm.csv u VTT_CSV_DIR.
    Format: kategorija, n_piksela, postotak
    """
    vtt_path = os.path.join(VTT_CSV_DIR, "background_vtt.csv")
    if not os.path.exists(vtt_path):
        raise FileNotFoundError(f"Nije pronaden background_vtt.csv — prvo pokreni export_background_vtt()")

    mocvarno = 0
    suho     = 0
    with open(vtt_path, encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            n = int(row["n_piksela"])
            if row["tip_tla"] in MOCVARNA_TLA:
                mocvarno += n
            else:
                suho += n

    total = mocvarno + suho
    out_path = os.path.join(VTT_CSV_DIR, "background_sm.csv")
    with open(out_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["kategorija", "n_piksela", "postotak"])
        w.writerow(["Mocvarno", mocvarno, round(mocvarno / total * 100, 4)])
        w.writerow(["Suho",     suho,     round(suho     / total * 100, 4)])

    print(f"Exportano: background_sm.csv")
    print(f"  Mocvarno: {mocvarno:,}  ({mocvarno/total*100:.2f}%)")
    print(f"  Suho:     {suho:,}  ({suho/total*100:.2f}%)")


def export_background_cfrag():
    """
    Cita distribuciju piksela rasterskog sloja coarse_fragments i
    exporta background_cfrag.csv u CFRAG_CSV_DIR.
    Format: value_volpct, n_piksela, postotak
    SoilGrids: raw vrijednosti su cm3/dm3 * 10  →  dijeli s 10 za vol%
    """
    from qgis.core import QgsProject
    import processing

    layers = QgsProject.instance().mapLayersByName(CFRAG_RASTER_LAYER)
    if not layers:
        raise ValueError(f"Sloj '{CFRAG_RASTER_LAYER}' nije pronaden. Provjeri naziv.")
    raster = layers[0]

    result = processing.run("native:rasterlayeruniquevaluesreport", {
        'INPUT':        raster,
        'BAND':         1,
        'OUTPUT_TABLE': 'memory:cfrag_vals',
    })
    table = result['OUTPUT_TABLE']

    rows = []
    total_pixels = 0
    for feat in table.getFeatures():
        raw   = float(feat['value'])
        count = int(feat['count'])
        if raw < 0:
            continue
        rows.append((round(raw / 10, 1), count))
        total_pixels += count

    rows.sort(key=lambda r: r[0])

    os.makedirs(CFRAG_CSV_DIR, exist_ok=True)
    out_path = os.path.join(CFRAG_CSV_DIR, "background_cfrag.csv")
    with open(out_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["value_volpct", "n_piksela", "postotak"])
        for val, count in rows:
            w.writerow([val, count, round(count / total_pixels * 100, 6)])

    print(f"Exportano: background_cfrag.csv  ({len(rows)} jedinstvenih vrijednosti, {total_pixels:,} piksela)")
    # ispis prvih i zadnjih 5 vrijednosti kao provjera
    for val, count in rows[:5]:
        print(f"  {val:>6.1f} vol%  →  {count:>10,}  ({count/total_pixels*100:.3f}%)")
    if len(rows) > 10:
        print(f"  ...")
    for val, count in rows[-5:]:
        print(f"  {val:>6.1f} vol%  →  {count:>10,}  ({count/total_pixels*100:.3f}%)")


# ============================================================
#  SEKCIJA 2 — VIZUALIZACIJA
#  Funkcije su aktivne; samo odkomentiraj pozive ispod
# ============================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOIL_COLORS = {
    "Alisols":    "#7B5C3E",
    "Arenosols":  "#D4A96A",
    "Calcisols":  "#E8D5A3",
    "Cambisols":  "#B8860B",
    "Chernozems": "#3B2B1A",
    "Fluvisols":  "#4472C4",
    "Gleysols":   "#70A8D8",
    "Leptosols":  "#9B6A4A",
    "Luvisols":   "#C17D3C",
    "Pheozems":   "#6B8E3C",
    "Regosols":   "#C4A882",
    "Vertisols":  "#7A6020",
    "Nepoznato":  "#CCCCCC",
}
SM_COLORS = {"Mocvarno": "#4472C4", "Suho": "#B8860B"}

LAYER_COLORS = {
    "random_ceste_biased":                     "#999999",
    "nasumicni_lokaliteti_umjetno_generirani": "#BBBBBB",
    "neolitik_svi_odredeni":                   "#2E75B6",
    "neolitik_c_starcevacka":                  "#4BACC6",
    "neolitik_c_sop_kor_len":                  "#70AD47",
    "kontinuirana_naselja":                    "#ED7D31",
    "samo_rani":                               "#FFC000",
    "samo_srednji_kasni":                      "#C00000",
}


def _load_freq_csv(path, col_tip, col_pct):
    """Ucitaj freq CSV, vrati dict {tip: postotak}."""
    result = {}
    with open(path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            result[row[col_tip]] = float(row[col_pct])
    return result


def plot_background_vtt():
    """Bar chart pozadinske distribucije tipova tla."""
    os.makedirs(PNG_OUT_DIR, exist_ok=True)
    vtt_path = os.path.join(VTT_CSV_DIR, "background_vtt.csv")
    freq = _load_freq_csv(vtt_path, "tip_tla", "postotak")
    soil_types = sorted(freq, key=lambda k: freq[k], reverse=True)
    values  = [freq[t] for t in soil_types]
    colors  = [SOIL_COLORS.get(t, "#999") for t in soil_types]

    fig, ax = plt.subplots(figsize=(max(7, len(soil_types) * 0.9), 5))
    bars = ax.bar(soil_types, values, color=colors, edgecolor="white", linewidth=0.6)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_title("Pozadinska distribucija — tipovi tla (cijeli raster)", fontsize=11)
    ax.set_ylabel("Udio površine (%)")
    ax.set_xlabel("Tip tla (WRB)")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(PNG_OUT_DIR, "background_vtt_bar.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  background_vtt_bar.png")


def plot_background_sm():
    """Bar chart pozadinske distribucije mocvarno/suho."""
    os.makedirs(PNG_OUT_DIR, exist_ok=True)
    sm_path = os.path.join(VTT_CSV_DIR, "background_sm.csv")
    freq = _load_freq_csv(sm_path, "kategorija", "postotak")

    labels  = ["Mocvarno", "Suho"]
    values  = [freq.get(l, 0) for l in labels]
    colors  = [SM_COLORS[l] for l in labels]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8, width=0.45)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_title("Pozadinska distribucija — mocvarno / suho (cijeli raster)", fontsize=10)
    ax.set_ylabel("Udio površine (%)")
    ax.set_xlabel("Tip tla")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(PNG_OUT_DIR, "background_sm_bar.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  background_sm_bar.png")


def plot_comparison_vtt(radius):
    """
    Slozeni (stacked) horizontalni bar chart: pozadina + svih 8 slojeva.
    X-os = udio svake kategorije tla, redci = grupe (pozadina + 8 slojeva).
    Jedan graf po polumjeru.
    """
    os.makedirs(PNG_OUT_DIR, exist_ok=True)

    bg_freq = _load_freq_csv(os.path.join(VTT_CSV_DIR, "background_vtt.csv"),
                             "tip_tla", "postotak")

    layer_freqs = {"__background__": bg_freq}
    for layer in LAYERS:
        freq_path = os.path.join(RES_VTT_DIR, f"{layer}_vtt_r{radius}_freq.csv")
        if os.path.exists(freq_path):
            layer_freqs[layer] = _load_freq_csv(freq_path, "tip_tla", "postotak")
        else:
            print(f"  NEDOSTAJE: {layer}_vtt_r{radius}_freq.csv")

    all_soils = sorted(
        {t for fd in layer_freqs.values() for t in fd},
        key=lambda t: bg_freq.get(t, 0), reverse=True
    )

    groups = list(layer_freqs.keys())
    ylabels = [LAYER_LABELS.get(g, g) for g in groups]
    n_groups = len(groups)

    fig, ax = plt.subplots(figsize=(11, max(5, n_groups * 0.6)))
    lefts = [0.0] * n_groups

    for soil in all_soils:
        vals = [layer_freqs[g].get(soil, 0) for g in groups]
        color = SOIL_COLORS.get(soil, "#999")
        bars = ax.barh(range(n_groups), vals, left=lefts, color=color,
                       edgecolor="white", linewidth=0.4, label=soil)
        for i, (v, l) in enumerate(zip(vals, lefts)):
            if v >= 5:
                ax.text(l + v/2, i, f"{v:.0f}%",
                        ha="center", va="center", fontsize=6.5, color="white", fontweight="bold")
        lefts = [l + v for l, v in zip(lefts, vals)]

    ax.set_yticks(range(n_groups))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xlabel("Udio (%)")
    ax.set_title(f"Tipovi tla — pozadina vs. slojevi  |  polumjer {radius} m", fontsize=11)
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(PNG_OUT_DIR, f"comparison_vtt_r{radius}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  comparison_vtt_r{radius}.png")


def plot_comparison_sm(radius):
    """Slozeni horizontalni bar chart mocvarno/suho: pozadina + 8 slojeva."""
    os.makedirs(PNG_OUT_DIR, exist_ok=True)

    bg_freq = _load_freq_csv(os.path.join(VTT_CSV_DIR, "background_sm.csv"),
                             "kategorija", "postotak")

    layer_freqs = {"__background__": bg_freq}
    for layer in LAYERS:
        freq_path = os.path.join(RES_SM_DIR, f"{layer}_sm_r{radius}_freq.csv")
        if os.path.exists(freq_path):
            layer_freqs[layer] = _load_freq_csv(freq_path, "kategorija", "postotak")
        else:
            print(f"  NEDOSTAJE: {layer}_sm_r{radius}_freq.csv")

    groups  = list(layer_freqs.keys())
    ylabels = [LAYER_LABELS.get(g, g) for g in groups]
    n_groups = len(groups)
    cats    = ["Mocvarno", "Suho"]

    fig, ax = plt.subplots(figsize=(9, max(5, n_groups * 0.6)))
    lefts = [0.0] * n_groups

    for cat in cats:
        vals  = [layer_freqs[g].get(cat, 0) for g in groups]
        color = SM_COLORS[cat]
        ax.barh(range(n_groups), vals, left=lefts, color=color,
                edgecolor="white", linewidth=0.4, label=cat)
        for i, (v, l) in enumerate(zip(vals, lefts)):
            if v >= 5:
                ax.text(l + v/2, i, f"{v:.0f}%",
                        ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        lefts = [l + v for l, v in zip(lefts, vals)]

    ax.set_yticks(range(n_groups))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xlabel("Udio (%)")
    ax.set_title(f"Mocvarno / suho — pozadina vs. slojevi  |  polumjer {radius} m", fontsize=11)
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(PNG_OUT_DIR, f"comparison_sm_r{radius}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  comparison_sm_r{radius}.png")


def _weighted_stats_cfrag(bg_df):
    """Izracunaj opisnu statistiku iz histograma pozadinskog rastera."""
    vals  = bg_df["value_volpct"].values.astype(float)
    cnts  = bg_df["n_piksela"].values.astype(float)
    total = cnts.sum()

    mean = float(np.average(vals, weights=cnts))

    def wp(p):
        cumul = np.cumsum(cnts)
        idx   = np.searchsorted(cumul, total * p / 100.0)
        return float(vals[min(idx, len(vals) - 1)])

    q1  = wp(25)
    med = wp(50)
    q3  = wp(75)
    iqr = q3 - q1

    return {
        "n":      int(total),
        "mean":   round(mean, 4),
        "median": round(med, 4),
        "q1":     round(q1, 4),
        "q3":     round(q3, 4),
        "iqr":    round(iqr, 4),
        "min":    round(float(vals.min()), 4),
        "max":    round(float(vals.max()), 4),
        "whislo": round(max(float(vals.min()), q1 - 1.5 * iqr), 4),
        "whishi": round(min(float(vals.max()), q3 + 1.5 * iqr), 4),
    }


def plot_background_cfrag():
    """Histogram pozadinske distribucije grubih fragmenata tla."""
    os.makedirs(PNG_OUT_DIR, exist_ok=True)
    bg_path = os.path.join(CFRAG_CSV_DIR, "background_cfrag.csv")
    if not os.path.exists(bg_path):
        print("  NEDOSTAJE: background_cfrag.csv — pokreni export_background_cfrag() u QGIS-u")
        return

    bg_df = pd.read_csv(bg_path, encoding="utf-8-sig")
    s     = _weighted_stats_cfrag(bg_df)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(bg_df["value_volpct"], bg_df["postotak"],
                    alpha=0.65, color="#4472C4", linewidth=0)
    ax.plot(bg_df["value_volpct"], bg_df["postotak"],
            color="#2255A0", linewidth=0.6)
    ax.axvline(s["mean"],   color="#333333", linestyle="--", linewidth=1.3,
               label=f"Srednja vrijednost = {s['mean']:.1f}%")
    ax.axvline(s["median"], color="#CC0000", linestyle=":",  linewidth=1.3,
               label=f"Medijan = {s['median']:.1f}%")

    ax.set_title("Pozadinska distribucija — grubi fragmenti tla (cijeli raster)", fontsize=11)
    ax.set_xlabel("Grubi fragmenti tla (vol%)")
    ax.set_ylabel("Udio površine (%)")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(PNG_OUT_DIR, "background_cfrag_hist.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  background_cfrag_hist.png")


def plot_comparison_cfrag():
    """Box plot: pozadina (iz rastera) + svih 8 slojeva za coarse fragments."""
    os.makedirs(PNG_OUT_DIR, exist_ok=True)

    bg_path = os.path.join(CFRAG_CSV_DIR, "background_cfrag.csv")
    if not os.path.exists(bg_path):
        print("  NEDOSTAJE: background_cfrag.csv — pokreni export_background_cfrag() u QGIS-u")
        return

    bg_df    = pd.read_csv(bg_path, encoding="utf-8-sig")
    bg_stats = _weighted_stats_cfrag(bg_df)

    layer_series = {}
    for layer in LAYERS:
        csv_path = os.path.join(CFRAG_CSV_DIR, f"{layer}.csv")
        if not os.path.exists(csv_path):
            print(f"  NEDOSTAJE: {layer}.csv")
            continue
        df     = pd.read_csv(csv_path, encoding="utf-8-sig")
        series = pd.to_numeric(df["c_frag"], errors="coerce").dropna() / 10
        layer_series[layer] = series

    all_groups = ["__background__"] + [l for l in LAYERS if l in layer_series]
    xlabels    = [LAYER_LABELS.get(g, g) for g in all_groups]
    n_groups   = len(all_groups)

    # Sastavi bxp_stats za svaku grupu
    bxp_stats = []
    colors    = []

    bxp_stats.append({
        "med":    bg_stats["median"],
        "q1":     bg_stats["q1"],
        "q3":     bg_stats["q3"],
        "whislo": bg_stats["whislo"],
        "whishi": bg_stats["whishi"],
        "fliers": [],
    })
    colors.append("#888888")

    for layer in LAYERS:
        if layer not in layer_series:
            continue
        s   = layer_series[layer]
        q1, q3 = np.percentile(s, [25, 75])
        iqr = q3 - q1
        fliers = s[((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr))].values.tolist()
        bxp_stats.append({
            "med":    float(s.median()),
            "q1":     float(q1),
            "q3":     float(q3),
            "whislo": float(max(s.min(), q1 - 1.5 * iqr)),
            "whishi": float(min(s.max(), q3 + 1.5 * iqr)),
            "fliers": fliers,
        })
        colors.append(LAYER_COLORS.get(layer, "#999999"))

    fig, ax = plt.subplots(figsize=(max(9, n_groups * 1.4), 5))

    bp = ax.bxp(
        bxp_stats,
        positions=range(n_groups),
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
        showfliers=True,
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    bp["boxes"][0].set_hatch("///")  # pozadina dobiva šrafuru

    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Grubi fragmenti tla (vol%)")
    ax.set_title("Grubi fragmenti tla — pozadina vs. slojevi", fontsize=11, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(PNG_OUT_DIR, "comparison_cfrag.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  comparison_cfrag.png")


# ============================================================
#  POKRENI
#  Promijeni MODE:
#    "export"  → pokreni u QGIS-u  (izvoz CSVova)
#    "analyse" → odkomentiraj Sekciju 2 i pokreni bilo gdje
# ============================================================

MODE = "analyse"  # "export" ili "analyse"

if MODE == "export":
    #export_background_vtt()
    #export_background_sm()
    export_background_cfrag()

elif MODE == "analyse":
    #plot_background_vtt()
    #plot_background_sm()
    for r in RADII:
        #plot_comparison_vtt(r)
        #plot_comparison_sm(r)
        pass
    plot_background_cfrag()
    plot_comparison_cfrag()
