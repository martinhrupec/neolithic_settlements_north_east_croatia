# -*- coding: utf-8 -*-
"""
RES_GUSTOCA_RIJEKA - Deskriptivna statistika gustoce rijeka (km/km²)
=====================================================================
Ulaz:  16 CSV-a iz tekucice.py  (fid, gustoca_km_km2)
       + background_gustoca.csv  (total_km, area_km2, gustoca_km_km2)
       mapa: deskriptivna/gustoca_rijeka_/csv_output/

Izlaz u deskriptivna/res_output/gustoca_rijeka/:
  summary_stats.csv                      - opisna statistika (16 redaka)
  {naziv}_gr_{radius}_hist.png           - histogrami po sloju × radijus (16 kom.)
  boxplot_gr_{radius}.png                - box plot svih 8 slojeva po radijusu (2 kom.)
  comparison_gr_{radius}.png             - box plot + pozadinska referentna linija (2 kom.)

Pokretanje: python res_gustoca_rijeka.py
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

INPUT_DIR  = r"C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna\gustoca_rijeka_\csv_output"
OUTPUT_DIR = r"C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna\res_output\gustoca_rijeka"

COL    = "gustoca_km_km2"
RADII  = [1000, 2000]

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

def load_all():
    """Vrati dict  data[(layer, radius)] = pd.Series."""
    data    = {}
    missing = 0
    for layer in LAYERS:
        for r in RADII:
            csv_path = os.path.join(INPUT_DIR, f"{layer}_gr_{r}.csv")
            if not os.path.exists(csv_path):
                print(f"  NEDOSTAJE: {layer}_gr_{r}.csv")
                missing += 1
                continue
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            if COL not in df.columns:
                print(f"  GREŠKA: stupac '{COL}' nije u {layer}_gr_{r}.csv")
                missing += 1
                continue
            series = pd.to_numeric(df[COL], errors="coerce").dropna()
            data[(layer, r)] = series
    print(f"  Učitano: {len(data)}   Nedostaje: {missing}")
    return data


def load_background():
    bg_path = os.path.join(INPUT_DIR, "background_gustoca.csv")
    if not os.path.exists(bg_path):
        print("  NEDOSTAJE: background_gustoca.csv")
        return None
    df = pd.read_csv(bg_path, encoding="utf-8-sig")
    return float(df["gustoca_km_km2"].iloc[0])

# ============================================================
#  DESKRIPTIVNA STATISTIKA
# ============================================================

def descriptive_stats(series):
    q1, q3 = np.percentile(series, [25, 75])
    return {
        "n":        len(series),
        "mean":     round(series.mean(), 6),
        "median":   round(series.median(), 6),
        "sd":       round(series.std(ddof=1), 6),
        "iqr":      round(q3 - q1, 6),
        "min":      round(series.min(), 6),
        "max":      round(series.max(), 6),
        "skewness": round(stats.skew(series), 4),
        "kurtosis": round(stats.kurtosis(series), 4),
        "q1":       round(q1, 6),
        "q3":       round(q3, 6),
    }

# ============================================================
#  VIZUALIZACIJA
# ============================================================

def plot_histogram(series, layer_name, radius, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    color = LAYER_COLORS.get(layer_name, "#555555")
    s = descriptive_stats(series)

    ax.hist(series, bins=25, color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.axvline(s["mean"],   color="#333333", linestyle="--", linewidth=1.2,
               label=f"Srednja vrijednost = {s['mean']:.4f}")
    ax.axvline(s["median"], color="#CC0000", linestyle=":",  linewidth=1.2,
               label=f"Medijan = {s['median']:.4f}")

    label = LAYER_LABELS.get(layer_name, layer_name).replace("\n", " ")
    ax.set_title(f"{label}  |  r = {radius} m  (n={s['n']})", fontsize=10)
    ax.set_xlabel("Gustoća rijeka (km/km²)")
    ax.set_ylabel("Frekvencija")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_boxplot_radius(data, radius, out_path, bg_value=None, title=""):
    """Box plot svih 8 slojeva za jedan radijus, s opcionalnom pozadinskom linijom."""
    plot_data   = [data[(l, radius)].values for l in LAYERS if (l, radius) in data]
    plot_labels = [LAYER_LABELS.get(l, l) for l in LAYERS if (l, radius) in data]
    plot_colors = [LAYER_COLORS.get(l, "#999999") for l in LAYERS if (l, radius) in data]

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

    if bg_value is not None:
        ax.axhline(bg_value, color="#CC3300", linestyle="--", linewidth=1.4,
                   label=f"Pozadina (cijelo područje) = {bg_value:.4f} km/km²")
        ax.legend(fontsize=8, loc="upper right")

    t = title or f"Gustoća rijeka — radijus {radius} m"
    ax.set_title(t, fontsize=11, pad=10)
    ax.set_ylabel("Gustoća rijeka (km/km²)")
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
    print("GUSTOCA RIJEKA - učitavanje CSV-a")
    print("=" * 60)
    data     = load_all()
    bg_value = load_background()

    if not data:
        print("Nema podataka — provjeri INPUT_DIR i pokreni tekucice.py u QGIS-u.")
        return

    if bg_value is not None:
        print(f"  Pozadinska gustoća: {bg_value:.6f} km/km²")

    # ── Deskriptivna statistika ─────────────────────────────
    print(f"\n{'='*60}")
    print("DESKRIPTIVNA STATISTIKA")
    print(f"{'='*60}")

    stats_rows = []
    for r in RADII:
        print(f"\n  Radijus {r} m")
        print(f"  {'Sloj':<44} {'Mean':>9}  {'Median':>9}  {'SD':>8}  {'Skew':>6}")
        print(f"  {'-'*82}")
        for layer in LAYERS:
            key = (layer, r)
            if key not in data:
                continue
            s = descriptive_stats(data[key])
            s["sloj"]    = layer
            s["radijus"] = r
            stats_rows.append(s)
            print(f"  {LAYER_LABELS.get(layer,layer).replace(chr(10),' '):<44} "
                  f"{s['mean']:>9.6f}  {s['median']:>9.6f}  {s['sd']:>8.6f}  {s['skewness']:>6.2f}")

    cols = ["sloj", "radijus", "n", "mean", "median", "sd", "iqr",
            "q1", "q3", "min", "max", "skewness", "kurtosis"]
    pd.DataFrame(stats_rows)[cols].to_csv(
        os.path.join(OUTPUT_DIR, "summary_stats.csv"), index=False, encoding="utf-8-sig"
    )
    print(f"\n  → summary_stats.csv  ({len(stats_rows)} redaka)")

    # ── Histogrami ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("HISTOGRAMI (16 kom.)")
    print(f"{'='*60}")
    for layer in LAYERS:
        for r in RADII:
            key = (layer, r)
            if key not in data:
                continue
            out_path = os.path.join(OUTPUT_DIR, f"{layer}_gr_{r}_hist.png")
            plot_histogram(data[key], layer, r, out_path)
    print(f"  → {len(data)} histograma")

    # ── Box plotovi ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("BOX PLOTOVI")
    print(f"{'='*60}")
    for r in RADII:
        bp_path = os.path.join(OUTPUT_DIR, f"boxplot_gr_{r}.png")
        plot_boxplot_radius(data, r, bp_path,
                            title=f"Gustoća rijeka — radijus {r} m")
        print(f"  boxplot_gr_{r}.png")

        cmp_path = os.path.join(OUTPUT_DIR, f"comparison_gr_{r}.png")
        plot_boxplot_radius(data, r, cmp_path, bg_value=bg_value,
                            title=f"Gustoća rijeka — pozadina vs. slojevi  |  radijus {r} m")
        print(f"  comparison_gr_{r}.png")

    print(f"\n{'='*60}")
    print(f"GOTOVO!  Izlaz: {OUTPUT_DIR}")
    print(f"{'='*60}")


run()
