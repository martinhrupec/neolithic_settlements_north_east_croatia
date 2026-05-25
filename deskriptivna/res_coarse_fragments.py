# -*- coding: utf-8 -*-
"""
RES_COARSE_FRAGMENTS - Deskriptivna statistika kontinuirane varijable
======================================================================
Ulaz:  8 CSV-a iz coarse_fragments.py   (fid, c_frag)
       mapa: deskriptivna/coarse_fragments_/csv_output/

Izlaz u deskriptivna/res_output/coarse_fragments/:
  summary_stats.csv         - sve statistike za svih 8 slojeva u jednoj tablici
  {naziv}_hist.png          - histogram po sloju
  boxplot_all.png           - kombinirani box plot svih 8 slojeva
  boxplot_groups.png        - kljucne grupe usporedo (random vs arheo vs faze)

Pokretanje: python res_coarse_fragments.py
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

INPUT_DIR  = r"C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna\coarse_fragments_\csv_output"
OUTPUT_DIR = r"C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna\res_output\coarse_fragments"

# Naziv atributa u CSV-u
COL = "c_frag"

# Citat&jive oznake za dijagrame (isti redosljed kao LAYERS)
LAYER_LABELS = {
    "random_ceste_biased":                     "Random\n(ceste)",
    "nasumicni_lokaliteti_umjetno_generirani": "Random\n(nasumični)",
    "neolitik_svi_odredeni":                   "Neolitik\n(svi)",
    "neolitik_c_starcevacka":                  "Rani\nneolitik",
    "neolitik_c_sop_kor_len":                  "Srednji i\nkasni neolitik",
    "kontinuirana_naselja":                    "Kontinuirana\nnaselja",
    "samo_rani":                               "Isključivo\nrana faza",
    "samo_srednji_kasni":                      "Isključivo srednja i\nkasna faza",
}

LAYERS = list(LAYER_LABELS.keys())

# Kljucne grupe za usporedni box plot
KEY_GROUPS = [
    "random_ceste_biased",
    "nasumicni_lokaliteti_umjetno_generirani",
    "neolitik_svi_odredeni",
    "samo_rani",
    "samo_srednji_kasni",
]

# Boje po sloju
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
#  UCITAVANJE PODATAKA
# ============================================================

def load_all():
    """Ucitaj sve CSV-e, vrati dict layer_name -> pd.Series (bez NaN)."""
    data = {}
    for layer in LAYERS:
        csv_path = os.path.join(INPUT_DIR, f"{layer}.csv")
        if not os.path.exists(csv_path):
            print(f"  NEDOSTAJE: {layer}.csv")
            continue
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if COL not in df.columns:
            print(f"  GRESKA: stupac '{COL}' nije u {layer}.csv")
            continue
        series = pd.to_numeric(df[COL], errors="coerce").dropna() / 10  # SoilGrids: cm³/dm³ → vol%
        data[layer] = series
        print(f"  {layer:<44}  n={len(series):<5}  nodata={df[COL].isna().sum() + (df[COL]=='').sum()}")
    return data

# ============================================================
#  DESKRIPTIVNA STATISTIKA
# ============================================================

def descriptive_stats(series):
    """Vrati dict sa svim statistikama za jednu seriju."""
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
        "kurtosis": round(stats.kurtosis(series), 4),  # excess kurtosis (normal=0)
        "q1":       round(q1, 4),
        "q3":       round(q3, 4),
    }

# ============================================================
#  VIZUALIZACIJA
# ============================================================

def plot_histogram(series, layer_name, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    color = LAYER_COLORS.get(layer_name, "#555555")
    s = descriptive_stats(series)

    ax.hist(series, bins=30, color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.axvline(s["mean"],   color="#333333", linestyle="--", linewidth=1.2, label=f"Srednja vrijednost = {s['mean']:.1f}%")
    ax.axvline(s["median"], color="#CC0000", linestyle=":",  linewidth=1.2, label=f"Medijan = {s['median']:.1f}%")

    ax.set_title(f"{LAYER_LABELS.get(layer_name, layer_name)}  (n={s['n']})", fontsize=11)
    ax.set_xlabel("Grubi fragmenti tla (vol%)")
    ax.set_ylabel("Frekvencija")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_boxplot(data, layers, out_path, title=""):
    """Kombinirani box plot za odabrane slojeve."""
    plot_data   = [data[l].values for l in layers if l in data]
    plot_labels = [LAYER_LABELS.get(l, l) for l in layers if l in data]
    plot_colors = [LAYER_COLORS.get(l, "#999999") for l in layers if l in data]

    fig, ax = plt.subplots(figsize=(max(7, len(plot_data) * 1.3), 5))

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

    if title:
        ax.set_title(title, fontsize=11, pad=10)
    ax.set_ylabel("Grubi fragmenti tla (vol%)")
    ax.tick_params(axis="x", labelsize=8)
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
    print("COARSE FRAGMENTS - ucitavanje CSV-a")
    print("=" * 60)
    data = load_all()

    if not data:
        print("Nema podataka - provjeri INPUT_DIR i pokreni coarse_fragments.py u QGIS-u.")
        return

    # ── Deskriptivna statistika ─────────────────────────────
    print(f"\n{'='*60}")
    print("DESKRIPTIVNA STATISTIKA")
    print(f"{'='*60}")

    stats_rows = []
    for layer in LAYERS:
        if layer not in data:
            continue
        s = descriptive_stats(data[layer])
        s["sloj"] = layer
        stats_rows.append(s)

        print(f"\n  {layer}")
        print(f"  {'n':<12}: {s['n']}")
        print(f"  {'Mean':<12}: {s['mean']}")
        print(f"  {'Median':<12}: {s['median']}")
        print(f"  {'SD':<12}: {s['sd']}")
        print(f"  {'IQR':<12}: {s['iqr']}  (Q1={s['q1']}, Q3={s['q3']})")
        print(f"  {'Min':<12}: {s['min']}")
        print(f"  {'Max':<12}: {s['max']}")
        print(f"  {'Skewness':<12}: {s['skewness']}")
        print(f"  {'Kurtosis':<12}: {s['kurtosis']}")

    # Spremi summary tablicu
    cols_order = ["sloj", "n", "mean", "median", "sd", "iqr", "q1", "q3", "min", "max", "skewness", "kurtosis"]
    summary_df = pd.DataFrame(stats_rows)[cols_order]
    stats_path = os.path.join(OUTPUT_DIR, "summary_stats.csv")
    summary_df.to_csv(stats_path, index=False, encoding="utf-8-sig")
    print(f"\n  → Tablica: summary_stats.csv")

    # ── Histogrami po sloju ─────────────────────────────────
    print(f"\n{'='*60}")
    print("HISTOGRAMI")
    print(f"{'='*60}")
    for layer, series in data.items():
        out_path = os.path.join(OUTPUT_DIR, f"{layer}_hist.png")
        plot_histogram(series, layer, out_path)
        print(f"  {layer}_hist.png")

    # ── Box plot - svih 8 slojeva ───────────────────────────
    print(f"\n{'='*60}")
    print("BOX PLOTOVI")
    print(f"{'='*60}")

    bp_all = os.path.join(OUTPUT_DIR, "boxplot_all.png")
    plot_boxplot(data, LAYERS, bp_all, title="Grubi fragmenti tla - svi slojevi")
    print(f"  boxplot_all.png")

    # Box plot - kljucne grupe (svih 8)
    bp_key = os.path.join(OUTPUT_DIR, "boxplot_groups.png")
    plot_boxplot(data, LAYERS, bp_key, title="Grubi fragmenti tla - ključne grupe")
    print(f"  boxplot_groups.png")

    print(f"\n{'='*60}")
    print(f"GOTOVO!  Izlaz: {OUTPUT_DIR}")
    print(f"{'='*60}")


run()
