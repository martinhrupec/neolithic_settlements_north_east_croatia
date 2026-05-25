# -*- coding: utf-8 -*-
"""
RES_DIST_RIJEKA - Deskriptivna statistika udaljenosti od rijeke
================================================================
Ulaz:  8 CSV-a iz tekucice.py  (fid, dist_rijeka)
       mapa: deskriptivna/dist_rijeka_/csv_output/

Izlaz u deskriptivna/res_output/dist_rijeka/:
  summary_stats.csv          - opisna statistika za svih 8 slojeva
  {naziv}_hist.png           - histogram po sloju  (8 kom.)
  boxplot_all.png            - kombinirani box plot svih 8 slojeva
  boxplot_groups.png         - kljucne grupe usporedo

Pokretanje: python res_dist_rijeka.py
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

INPUT_DIR      = r"C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna\dist_rijeka_\csv_output"
INPUT_COPY_DIR = r"C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna\dist_rijeka_korig_\csv_output"
OUTPUT_DIR     = r"C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna\res_output\dist_rijeka"

COL      = "dist_rijeka"
COL_COPY = "dist_rijeka_korig"

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

KEY_GROUPS = [
    "random_ceste_biased",
    "nasumicni_lokaliteti_umjetno_generirani",
    "neolitik_svi_odredeni",
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
    data = {}
    for layer in LAYERS:
        csv_path = os.path.join(INPUT_DIR, f"{layer}.csv")
        if not os.path.exists(csv_path):
            print(f"  NEDOSTAJE: {layer}.csv")
            continue
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if COL not in df.columns:
            print(f"  GREŠKA: stupac '{COL}' nije u {layer}.csv")
            continue
        series = pd.to_numeric(df[COL], errors="coerce").dropna()
        data[layer] = series
        print(f"  {layer:<44}  n={len(series):<5}  nodata={df[COL].isna().sum()}")
    return data

# ============================================================
#  DESKRIPTIVNA STATISTIKA
# ============================================================

def descriptive_stats(series):
    q1, q3 = np.percentile(series, [25, 75])
    return {
        "n":        len(series),
        "mean":     round(series.mean(), 2),
        "median":   round(series.median(), 2),
        "sd":       round(series.std(ddof=1), 2),
        "iqr":      round(q3 - q1, 2),
        "min":      round(series.min(), 2),
        "max":      round(series.max(), 2),
        "skewness": round(stats.skew(series), 4),
        "kurtosis": round(stats.kurtosis(series), 4),
        "q1":       round(q1, 2),
        "q3":       round(q3, 2),
    }

# ============================================================
#  VIZUALIZACIJA
# ============================================================

def plot_histogram(series, layer_name, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    color = LAYER_COLORS.get(layer_name, "#555555")
    s = descriptive_stats(series)

    ax.hist(series, bins=30, color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.axvline(s["mean"],   color="#333333", linestyle="--", linewidth=1.2,
               label=f"Srednja vrijednost = {s['mean']:.0f} m")
    ax.axvline(s["median"], color="#CC0000", linestyle=":",  linewidth=1.2,
               label=f"Medijan = {s['median']:.0f} m")

    ax.set_title(f"{LAYER_LABELS.get(layer_name, layer_name)}  (n={s['n']})", fontsize=11)
    ax.set_xlabel("Udaljenost od rijeke (m)")
    ax.set_ylabel("Frekvencija")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_boxplot(data, layers, out_path, title=""):
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
    ax.set_ylabel("Udaljenost od rijeke (m)")
    ax.tick_params(axis="x", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ============================================================
#  UCITAVANJE — KORIGIRANI SLOJ
# ============================================================

def load_copy():
    """Ucitaj CSV-e iz korigiranog sloja (tekucice_copy)."""
    data = {}
    for layer in LAYERS:
        csv_path = os.path.join(INPUT_COPY_DIR, f"{layer}.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if COL_COPY not in df.columns:
            continue
        series = pd.to_numeric(df[COL_COPY], errors="coerce").dropna()
        data[layer] = {"series": series, "fid_val": dict(zip(df["fid"], pd.to_numeric(df[COL_COPY], errors="coerce")))}
    return data


# ============================================================
#  USPOREDBA ORIGINAL vs KORIGIRANO
# ============================================================

def plot_comparison_boxplot(data_orig, data_copy, out_path):
    """Paired box plot: original (pune boje) vs korigirano (isprekidane boje) za svaki sloj."""
    layers_ok = [l for l in LAYERS if l in data_orig and l in data_copy]
    n = len(layers_ok)

    fig, ax = plt.subplots(figsize=(max(9, n * 1.8), 6))

    positions_orig = [i * 2.2        for i in range(n)]
    positions_copy = [i * 2.2 + 0.9  for i in range(n)]

    bp_orig = ax.boxplot(
        [data_orig[l].values for l in layers_ok],
        positions=positions_orig,
        widths=0.7,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(linewidth=1.1),
        capprops=dict(linewidth=1.1),
        flierprops=dict(marker="o", markersize=2.5, alpha=0.35),
    )
    bp_copy = ax.boxplot(
        [data_copy[l]["series"].values for l in layers_ok],
        positions=positions_copy,
        widths=0.7,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(linewidth=1.1, linestyle="--"),
        capprops=dict(linewidth=1.1),
        flierprops=dict(marker="^", markersize=2.5, alpha=0.35),
    )

    for patch, layer in zip(bp_orig["boxes"], layers_ok):
        patch.set_facecolor(LAYER_COLORS.get(layer, "#999999"))
        patch.set_alpha(0.85)
    for patch, layer in zip(bp_copy["boxes"], layers_ok):
        patch.set_facecolor(LAYER_COLORS.get(layer, "#999999"))
        patch.set_alpha(0.45)
        patch.set_hatch("///")

    tick_pos    = [(a + b) / 2 for a, b in zip(positions_orig, positions_copy)]
    tick_labels = [LAYER_LABELS.get(l, l) for l in layers_ok]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=8)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#888888", alpha=0.85, label="Originalna mreža"),
        Patch(facecolor="#888888", alpha=0.45, hatch="///", label="Korigirana mreža (+10 rijeka)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

    ax.set_ylabel("Udaljenost od rijeke (m)")
    ax.set_title("Udaljenost od rijeke — originalna vs. korigirana mreža", fontsize=11, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_delta_histogram(data_orig, data_copy, layer_name, out_path):
    """Histogram promjene udaljenosti (delta = copy - orig) za jedan sloj."""
    orig_df = pd.read_csv(os.path.join(INPUT_DIR, f"{layer_name}.csv"), encoding="utf-8-sig")
    copy_df = pd.read_csv(os.path.join(INPUT_COPY_DIR, f"{layer_name}.csv"), encoding="utf-8-sig")

    orig_df["fid"] = orig_df["fid"].astype(int)
    copy_df["fid"] = copy_df["fid"].astype(int)
    orig_df[COL]      = pd.to_numeric(orig_df[COL],      errors="coerce")
    copy_df[COL_COPY] = pd.to_numeric(copy_df[COL_COPY], errors="coerce")

    merged = orig_df[["fid", COL]].merge(copy_df[["fid", COL_COPY]], on="fid").dropna()
    merged["delta"] = merged[COL_COPY] - merged[COL]

    n_changed   = int((merged["delta"].abs() > 1).sum())
    mean_delta  = merged["delta"].mean()
    max_improve = merged["delta"].min()   # negativan = smanjenje udaljenosti

    fig, ax = plt.subplots(figsize=(6, 4))
    color = LAYER_COLORS.get(layer_name, "#555555")
    ax.hist(merged["delta"], bins=20, color=color, edgecolor="white",
            linewidth=0.5, alpha=0.85)
    ax.axvline(0, color="#333333", linewidth=1.2, linestyle="--", label="Nema promjene")
    ax.axvline(mean_delta, color="#CC0000", linewidth=1.2, linestyle=":",
               label=f"Srednja delta = {mean_delta:.0f} m")

    ax.set_title(
        f"{LAYER_LABELS.get(layer_name, layer_name)}\n"
        f"Promjena udaljenosti (korigirano − originalno)\n"
        f"n={len(merged)}  promijenjenih={n_changed}  max poboljšanje={abs(max_improve):.0f} m",
        fontsize=9,
    )
    ax.set_xlabel("Delta udaljenosti (m)  [negativno = bliže rijeci]")
    ax.set_ylabel("Frekvencija")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def export_comparison_stats(data_orig, data_copy, out_path):
    """CSV s deskriptivnom statistikom za obje verzije i delta medijana."""
    rows = []
    for layer in LAYERS:
        if layer not in data_orig or layer not in data_copy:
            continue
        s_o = descriptive_stats(data_orig[layer])
        s_c = descriptive_stats(data_copy[layer]["series"])
        rows.append({
            "sloj":           layer,
            "n_orig":         s_o["n"],
            "median_orig":    s_o["median"],
            "mean_orig":      s_o["mean"],
            "n_copy":         s_c["n"],
            "median_copy":    s_c["median"],
            "mean_copy":      s_c["mean"],
            "delta_median":   round(s_c["median"] - s_o["median"], 2),
            "delta_mean":     round(s_c["mean"]   - s_o["mean"],   2),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


# ============================================================
#  GLAVNA ANALIZA
# ============================================================

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("UDALJENOST OD RIJEKE - učitavanje CSV-a")
    print("=" * 60)
    data = load_all()

    if not data:
        print("Nema podataka — provjeri INPUT_DIR i pokreni tekucice.py u QGIS-u.")
        return

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
        print(f"  {'Mean':<12}: {s['mean']} m")
        print(f"  {'Median':<12}: {s['median']} m")
        print(f"  {'SD':<12}: {s['sd']}")
        print(f"  {'IQR':<12}: {s['iqr']}  (Q1={s['q1']}, Q3={s['q3']})")
        print(f"  {'Min':<12}: {s['min']} m")
        print(f"  {'Max':<12}: {s['max']} m")
        print(f"  {'Skewness':<12}: {s['skewness']}")
        print(f"  {'Kurtosis':<12}: {s['kurtosis']}")

    cols_order = ["sloj", "n", "mean", "median", "sd", "iqr", "q1", "q3",
                  "min", "max", "skewness", "kurtosis"]
    pd.DataFrame(stats_rows)[cols_order].to_csv(
        os.path.join(OUTPUT_DIR, "summary_stats.csv"), index=False, encoding="utf-8-sig"
    )
    print(f"\n  → summary_stats.csv")

    print(f"\n{'='*60}")
    print("HISTOGRAMI")
    print(f"{'='*60}")
    for layer, series in data.items():
        out_path = os.path.join(OUTPUT_DIR, f"{layer}_hist.png")
        plot_histogram(series, layer, out_path)
        print(f"  {layer}_hist.png")

    print(f"\n{'='*60}")
    print("BOX PLOTOVI")
    print(f"{'='*60}")
    plot_boxplot(data, LAYERS, os.path.join(OUTPUT_DIR, "boxplot_all.png"),
                 title="Udaljenost od rijeke — svi slojevi")
    print("  boxplot_all.png")

    plot_boxplot(data, KEY_GROUPS, os.path.join(OUTPUT_DIR, "boxplot_groups.png"),
                 title="Udaljenost od rijeke — ključne grupe")
    print("  boxplot_groups.png")

    print(f"\n{'='*60}")
    print(f"GOTOVO!  Izlaz: {OUTPUT_DIR}")
    print(f"{'='*60}")


def run_comparison():
    """Usporedna analiza: originalna vs. korigirana rijecna mreza."""
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("USPOREDBA — originalna vs. korigirana mreza")
    print("=" * 60)

    print("\nUcitavanje originalne mreze...")
    data_orig = load_all()
    print("\nUcitavanje korigirane mreze...")
    data_copy = load_copy()

    if not data_orig or not data_copy:
        print("Nedostaju podaci — provjeri je li tekucice.py (run_copy) pokrenut u QGIS-u.")
        return

    # Comparison stats CSV
    stats_path = os.path.join(OUTPUT_DIR, "comparison_stats_orig_vs_copy.csv")
    export_comparison_stats(data_orig, data_copy, stats_path)
    print(f"\n  comparison_stats_orig_vs_copy.csv")

    # Paired box plot
    plot_comparison_boxplot(
        data_orig, data_copy,
        os.path.join(OUTPUT_DIR, "comparison_boxplot_orig_vs_copy.png"),
    )
    print("  comparison_boxplot_orig_vs_copy.png")

    # Delta histogrami po sloju
    print("\nDelta histogrami:")
    for layer in LAYERS:
        orig_csv = os.path.join(INPUT_DIR, f"{layer}.csv")
        copy_csv = os.path.join(INPUT_COPY_DIR, f"{layer}.csv")
        if not os.path.exists(orig_csv) or not os.path.exists(copy_csv):
            continue
        out_path = os.path.join(OUTPUT_DIR, f"{layer}_delta_hist.png")
        plot_delta_histogram(data_orig, data_copy, layer, out_path)
        print(f"  {layer}_delta_hist.png")

    print(f"\n{'='*60}")
    print(f"GOTOVO!  Izlaz: {OUTPUT_DIR}")
    print(f"{'='*60}")


def run_hist_korig():
    """Histogrami korigirane mreze — {layer}_hist_korig.png po sloju."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("HISTOGRAMI — korigirana mreza (dist_rijeka_korig)")
    print("=" * 60)

    data_copy = load_copy()
    if not data_copy:
        print("Nema podataka — provjeri je li tekucice.py pokrenut u QGIS-u.")
        return

    for layer, d in data_copy.items():
        series   = d["series"]
        out_path = os.path.join(OUTPUT_DIR, f"{layer}_hist_korig.png")
        plot_histogram(series, layer, out_path)
        print(f"  {layer}_hist_korig.png  (n={len(series)})")

    print(f"\nGOTOVO!  Izlaz: {OUTPUT_DIR}")


# run()             # standardna deskriptivna analiza
# run_comparison()    # usporedba original vs. korigirano
run_hist_korig()  # histogrami korigirane mreze (_hist_korig.png)
