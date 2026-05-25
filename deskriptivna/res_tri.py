# -*- coding: utf-8 -*-
"""
RES_TRI - Terrain Ruggedness Index — opisna statistika i usporedba
==================================================================
Ulaz:  8 CSV-a iz tri.py  (fid, tri)
       + background_tri.csv  (tri_value, n_piksela, postotak)
       mapa: deskriptivna/tri_/csv_output/

Izlaz u deskriptivna/res_output/tri/:
  summary_stats.csv          - opisna statistika za svih 8 slojeva
  {naziv}_hist.png           - histogram po sloju  (8 kom.)
  boxplot_all.png            - kombinirani box plot svih 8 slojeva
  boxplot_groups.png         - kljucne grupe usporedo
  background_tri_hist.png    - histogram pozadinskog rastera
  comparison_tri.png         - box plot: pozadina + svih 8 slojeva

Pokretanje: python res_tri.py
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

INPUT_DIR  = r"C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna\tri_\csv_output"
OUTPUT_DIR = r"C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna\res_output\tri"

COL = "tri"

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
    "__background__":                          "Pozadina\n(cijeli raster)",
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


def load_background():
    bg_path = os.path.join(INPUT_DIR, "background_tri.csv")
    if not os.path.exists(bg_path):
        print("  NEDOSTAJE: background_tri.csv")
        return None
    return pd.read_csv(bg_path, encoding="utf-8-sig")

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


def weighted_stats_bg(bg_df):
    """Opisna statistika iz histogramskog CSV-a (ponderirana po broju piksela)."""
    vals  = bg_df["tri_value"].values.astype(float)
    cnts  = bg_df["n_piksela"].values.astype(float)
    total = cnts.sum()
    mean  = float(np.average(vals, weights=cnts))

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

# ============================================================
#  VIZUALIZACIJA — STANDARDNA
# ============================================================

def plot_histogram(series, layer_name, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    color = LAYER_COLORS.get(layer_name, "#555555")
    s = descriptive_stats(series)

    ax.hist(series, bins=30, color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.axvline(s["mean"],   color="#333333", linestyle="--", linewidth=1.2,
               label=f"Srednja vrijednost = {s['mean']:.2f}")
    ax.axvline(s["median"], color="#CC0000", linestyle=":",  linewidth=1.2,
               label=f"Medijan = {s['median']:.2f}")

    ax.set_title(f"{LAYER_LABELS.get(layer_name, layer_name)}  (n={s['n']})", fontsize=11)
    ax.set_xlabel("Terrain Ruggedness Index (TRI)")
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
    ax.set_ylabel("Terrain Ruggedness Index (TRI)")
    ax.tick_params(axis="x", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ============================================================
#  VIZUALIZACIJA — POZADINSKA USPOREDBA
# ============================================================

def plot_background_hist(bg_df, out_path):
    """Histogram (fill_between) pozadinskog rastera TRI."""
    s = weighted_stats_bg(bg_df)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(bg_df["tri_value"], bg_df["postotak"],
                    alpha=0.65, color="#2E75B6", linewidth=0)
    ax.plot(bg_df["tri_value"], bg_df["postotak"],
            color="#1A5090", linewidth=0.7)
    ax.axvline(s["mean"],   color="#333333", linestyle="--", linewidth=1.3,
               label=f"Srednja vrijednost = {s['mean']:.2f}")
    ax.axvline(s["median"], color="#CC0000", linestyle=":",  linewidth=1.3,
               label=f"Medijan = {s['median']:.2f}")

    ax.set_title("Pozadinska distribucija — Terrain Ruggedness Index (cijeli raster)", fontsize=11)
    ax.set_xlabel("Terrain Ruggedness Index (TRI)")
    ax.set_ylabel("Udio površine (%)")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(data, bg_df, out_path):
    """Box plot: pozadina (iz rastera) + svih 8 slojeva."""
    bg_stats = weighted_stats_bg(bg_df)
    all_groups = ["__background__"] + [l for l in LAYERS if l in data]
    xlabels    = [LAYER_LABELS.get(g, g) for g in all_groups]
    n_groups   = len(all_groups)

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
        if layer not in data:
            continue
        s = data[layer]
        q1, q3 = np.percentile(s, [25, 75])
        iqr    = q3 - q1
        bxp_stats.append({
            "med":    float(s.median()),
            "q1":     float(q1),
            "q3":     float(q3),
            "whislo": float(max(s.min(), q1 - 1.5 * iqr)),
            "whishi": float(min(s.max(), q3 + 1.5 * iqr)),
            "fliers": s[((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr))].values.tolist(),
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
    bp["boxes"][0].set_hatch("///")

    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Terrain Ruggedness Index (TRI)")
    ax.set_title("TRI — pozadina vs. slojevi", fontsize=11, pad=10)
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
    print("TRI - učitavanje CSV-a")
    print("=" * 60)
    data  = load_all()
    bg_df = load_background()

    if not data:
        print("Nema podataka — provjeri INPUT_DIR i pokreni tri.py u QGIS-u.")
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

    cols_order = ["sloj", "n", "mean", "median", "sd", "iqr", "q1", "q3", "min", "max", "skewness", "kurtosis"]
    pd.DataFrame(stats_rows)[cols_order].to_csv(
        os.path.join(OUTPUT_DIR, "summary_stats.csv"), index=False, encoding="utf-8-sig"
    )
    print(f"\n  → summary_stats.csv")

    # ── Histogrami ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("HISTOGRAMI")
    print(f"{'='*60}")
    for layer, series in data.items():
        out_path = os.path.join(OUTPUT_DIR, f"{layer}_hist.png")
        plot_histogram(series, layer, out_path)
        print(f"  {layer}_hist.png")

    # ── Box plotovi ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("BOX PLOTOVI")
    print(f"{'='*60}")

    plot_boxplot(data, LAYERS, os.path.join(OUTPUT_DIR, "boxplot_all.png"),
                 title="TRI — svi slojevi")
    print("  boxplot_all.png")

    plot_boxplot(data, KEY_GROUPS, os.path.join(OUTPUT_DIR, "boxplot_groups.png"),
                 title="TRI — ključne grupe")
    print("  boxplot_groups.png")

    # ── Pozadinska usporedba ────────────────────────────────
    if bg_df is not None:
        print(f"\n{'='*60}")
        print("POZADINSKA USPOREDBA")
        print(f"{'='*60}")

        s_bg = weighted_stats_bg(bg_df)
        print(f"  Pozadina — mean={s_bg['mean']:.4f}  median={s_bg['median']:.4f}  "
              f"Q1={s_bg['q1']:.4f}  Q3={s_bg['q3']:.4f}")

        plot_background_hist(bg_df, os.path.join(OUTPUT_DIR, "background_tri_hist.png"))
        print("  background_tri_hist.png")

        plot_comparison(data, bg_df, os.path.join(OUTPUT_DIR, "comparison_tri.png"))
        print("  comparison_tri.png")
    else:
        print("\n  Pozadinska usporedba preskočena (nema background_tri.csv)")

    print(f"\n{'='*60}")
    print(f"GOTOVO!  Izlaz: {OUTPUT_DIR}")
    print(f"{'='*60}")


run()
