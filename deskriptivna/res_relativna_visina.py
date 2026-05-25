# -*- coding: utf-8 -*-
"""
RES_RELATIVNA_VISINA - Deskriptivna statistika relativne prominencije
======================================================================
Ulaz:  48 CSV-a iz relativna_visina.py   (fid, rel_{x}_{y}m)
       mapa: deskriptivna/relativna_visina_/csv_output/

Izlaz u deskriptivna/res_output/relativna_visina/:
  summary_stats.csv            - sve statistike za svih 48 kombinacija
  {naziv}_rv_{x}_{y}_hist.png  - histogram po sloju × kombinacija (48 kom.)
  boxplot_rv_{x}_{y}.png       - box plot svih 8 slojeva po kombinaciji (6 kom.)

Pokretanje: python res_relativna_visina.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
#  POSTAVKE
# ============================================================

INPUT_DIR  = r"C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna\relativna_visina_\csv_output"
OUTPUT_DIR = r"C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna\res_output\relativna_visina"

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

RADIUS_COMBINATIONS = [
    (100,  250),
    (100,  500),
    (100, 1000),
    (200,  500),
    (200, 1000),
    (500, 1000),
]

LAYER_LABELS = {
    "random_ceste_biased":                     "Random (ceste)",
    "nasumicni_lokaliteti_umjetno_generirani": "Random (nasumični)",
    "neolitik_svi_odredeni":                   "Neolitik (svi određeni)",
    "neolitik_c_starcevacka":                  "Rani neolitik",
    "neolitik_c_sop_kor_len":                  "Srednji i kasni neolitik",
    "kontinuirana_naselja":                    "Kontinuirana naselja",
    "samo_rani":                               "Isključivo rana faza",
    "samo_srednji_kasni":                      "Isključivo srednja i kasna faza",
}

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

# ============================================================
#  UCITAVANJE
# ============================================================

def col_name(x, y):
    return f"rel_{x}_{y}m"


def load_all():
    """Vrati dict  data[(layer, x, y)] = pd.Series."""
    data = {}
    missing = 0
    for layer in LAYERS:
        for (x, y) in RADIUS_COMBINATIONS:
            csv_path = os.path.join(INPUT_DIR, f"{layer}_rv_{x}_{y}.csv")
            if not os.path.exists(csv_path):
                print(f"  NEDOSTAJE: {layer}_rv_{x}_{y}.csv")
                missing += 1
                continue
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            col = col_name(x, y)
            if col not in df.columns:
                print(f"  GREŠKA: stupac '{col}' nije u {layer}_rv_{x}_{y}.csv")
                missing += 1
                continue
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            data[(layer, x, y)] = series
    print(f"  Učitano: {len(data)}   Nedostaje: {missing}")
    return data

# ============================================================
#  DESKRIPTIVNA STATISTIKA
# ============================================================

def descriptive_stats(series):
    q1, q3 = np.percentile(series, [25, 75])
    return {
        "n":        len(series),
        "mean":     round(series.mean(), 4),
        "median":   round(series.median(), 4),
        "sd":       round(series.std(ddof=1), 4),
        "iqr":      round(q3 - q1, 4),
        "min":      round(series.min(), 4),
        "max":      round(series.max(), 4),
        "skewness": round(stats.skew(series), 4),
        "kurtosis": round(stats.kurtosis(series), 4),
        "q1":       round(q1, 4),
        "q3":       round(q3, 4),
    }

# ============================================================
#  VIZUALIZACIJA
# ============================================================

def plot_histogram(series, layer_name, x, y, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    color = LAYER_COLORS.get(layer_name, "#555555")
    s = descriptive_stats(series)

    ax.hist(series, bins=30, color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.axvline(s["mean"],   color="#333333", linestyle="--", linewidth=1.2,
               label=f"Srednja vrijednost = {s['mean']:.2f} m")
    ax.axvline(s["median"], color="#CC0000", linestyle=":",  linewidth=1.2,
               label=f"Medijan = {s['median']:.2f} m")
    ax.axvline(0, color="#000000", linestyle="-", linewidth=0.8, alpha=0.3)

    label = LAYER_LABELS.get(layer_name, layer_name)
    ax.set_title(f"{label}  |  unutarnji {x} m / vanjski {y} m  (n={s['n']})", fontsize=10)
    ax.set_xlabel("Relativna prominencija (m)")
    ax.set_ylabel("Frekvencija")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_boxplot_combination(data, x, y, out_path):
    """Box plot svih 8 slojeva za jednu kombinaciju polumjera."""
    plot_data   = [data[(l, x, y)].values for l in LAYERS if (l, x, y) in data]
    plot_labels = [LAYER_LABELS.get(l, l)   for l in LAYERS if (l, x, y) in data]
    plot_colors = [LAYER_COLORS.get(l, "#999") for l in LAYERS if (l, x, y) in data]

    fig, ax = plt.subplots(figsize=(max(9, len(plot_data) * 1.4), 5))

    bp = ax.boxplot(
        plot_data,
        labels=plot_labels,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    for patch, color in zip(bp["boxes"], plot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.axhline(0, color="#333333", linestyle="--", linewidth=0.9, alpha=0.6,
               label="Nulta razina (ravan teren)")
    ax.set_title(f"Relativna prominencija — unutarnji {x} m / vanjski {y} m", fontsize=11, pad=10)
    ax.set_ylabel("Relativna prominencija (m)")
    ax.tick_params(axis="x", labelsize=8, rotation=15)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ============================================================
#  GLAVNA ANALIZA
# ============================================================

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("RELATIVNA VISINA - učitavanje CSV-a")
    print("=" * 60)
    data = load_all()

    if not data:
        print("Nema podataka — provjeri INPUT_DIR i pokreni relativna_visina.py u QGIS-u.")
        return

    # ── Deskriptivna statistika ─────────────────────────────
    print(f"\n{'='*60}")
    print("DESKRIPTIVNA STATISTIKA")
    print(f"{'='*60}")

    stats_rows = []
    for layer in LAYERS:
        for (x, y) in RADIUS_COMBINATIONS:
            key = (layer, x, y)
            if key not in data:
                continue
            s = descriptive_stats(data[key])
            s["sloj"]       = layer
            s["inner_r_m"]  = x
            s["outer_r_m"]  = y
            s["kombinacija"] = f"{x}/{y} m"
            stats_rows.append(s)

    cols = ["sloj", "kombinacija", "inner_r_m", "outer_r_m",
            "n", "mean", "median", "sd", "iqr", "q1", "q3", "min", "max", "skewness", "kurtosis"]
    summary_df = pd.DataFrame(stats_rows)[cols]
    stats_path = os.path.join(OUTPUT_DIR, "summary_stats.csv")
    summary_df.to_csv(stats_path, index=False, encoding="utf-8-sig")
    print(f"  → summary_stats.csv  ({len(stats_rows)} redaka)")

    # Konzolni ispis po kombinaciji
    for (x, y) in RADIUS_COMBINATIONS:
        print(f"\n  Kombinacija {x}/{y} m")
        print(f"  {'Sloj':<44} {'Mean':>7}  {'Median':>7}  {'SD':>6}  {'Skew':>6}")
        print(f"  {'-'*76}")
        for layer in LAYERS:
            key = (layer, x, y)
            if key not in data:
                continue
            s = descriptive_stats(data[key])
            print(f"  {LAYER_LABELS.get(layer, layer):<44} {s['mean']:>7.2f}  {s['median']:>7.2f}  {s['sd']:>6.2f}  {s['skewness']:>6.2f}")

    # ── Histogrami ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("HISTOGRAMI (48 kom.)")
    print(f"{'='*60}")
    for layer in LAYERS:
        for (x, y) in RADIUS_COMBINATIONS:
            key = (layer, x, y)
            if key not in data:
                continue
            out_path = os.path.join(OUTPUT_DIR, f"{layer}_rv_{x}_{y}_hist.png")
            plot_histogram(data[key], layer, x, y, out_path)
    print(f"  → {len(data)} histograma")

    # ── Box plotovi po kombinaciji ──────────────────────────
    print(f"\n{'='*60}")
    print("BOX PLOTOVI (6 kom.)")
    print(f"{'='*60}")
    for (x, y) in RADIUS_COMBINATIONS:
        out_path = os.path.join(OUTPUT_DIR, f"boxplot_rv_{x}_{y}.png")
        plot_boxplot_combination(data, x, y, out_path)
        print(f"  boxplot_rv_{x}_{y}.png")

    print(f"\n{'='*60}")
    print(f"GOTOVO!  Izlaz: {OUTPUT_DIR}")
    print(f"{'='*60}")


run()
