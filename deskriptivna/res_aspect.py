# -*- coding: utf-8 -*-
"""
RES_ASPECT - Smjer padine — kategorijska analiza
==================================================
Ulaz:  8 CSV-a iz aspect.py  (fid, aspect, nagib)
       + background_aspect.csv  (aspect_value, n_piksela, postotak)

Izlaz u deskriptivna/res_output/aspect/:
  KATEGORIJE (4-dijelna kruznica):
    {sloj}_cat4.csv            - frekvencije NE, SE, SW, NW
    {sloj}_cat4_bar.png        - bar plot po sloju
    comparison_cat4.png        - stacked bar: pozadina + 8 slojeva

  SJEVER/JUG (2-dijelna):
    {sloj}_sn.csv              - frekvencije N, S
    {sloj}_sn_bar.png          - bar plot po sloju
    comparison_sn.png          - stacked bar: pozadina + 8 slojeva

  ISTOK/ZAPAD (2-dijelna):
    {sloj}_ew.csv              - frekvencije E, W
    {sloj}_ew_bar.png          - bar plot po sloju
    comparison_ew.png          - stacked bar: pozadina + 8 slojeva

QGIS aspect konvencija: 0°=N, 90°=E, 180°=S, 270°=W

Pokretanje: python res_aspect.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # registrira 3D projekciju kao side effect

# ============================================================
#  POSTAVKE
# ============================================================

INPUT_DIR  = r"C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna\aspect_\csv_output"
OUTPUT_DIR = r"C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna\res_output\aspect"

ASPECT_COL = "aspect"
NAGIB_COL  = "nagib"

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
    "__background__":                          "Pozadina",
}

CAT4_COLORS = {
    "NE": "#1f77b4",  # plava
    "SE": "#ff7f0e",  # narandžasta
    "SW": "#2ca02c",  # zelena
    "NW": "#d62728",  # crvena
}

SN_COLORS = {
    "N": "#1f77b4",  # plava (sjever)
    "S": "#d62728",  # crvena (jug)
}

EW_COLORS = {
    "E": "#1f77b4",  # plava (istok)
    "W": "#d62728",  # crvena (zapad)
}

# ============================================================
#  FUNKCIJE MAPIRANJA
# ============================================================

def aspect_to_cat4(aspect_deg):
    """Konvertuj aspect (0-360°) u 4 kategorije: NE, SE, SW, NW.

    0° = N (north)
    90° = E (east)
    180° = S (south)
    270° = W (west)

    NE (Sjeveroisток): 0-90°
    SE (Jugoistok): 90-180°
    SW (Jugozapad): 180-270°
    NW (Sjeverozapad): 270-360°
    """
    if pd.isna(aspect_deg):
        return None
    a = aspect_deg % 360
    if a < 90:
        return "NE"
    elif a < 180:
        return "SE"
    elif a < 270:
        return "SW"
    else:  # 270 <= a < 360
        return "NW"


def aspect_to_sn(aspect_deg):
    """Konvertuj aspect u SJEVER/JUG.

    N (sjever): 270-90° (kroz 0°, tj. W→N→E)
    S (jug):    90-270° (kroz 180°, tj. E→S→W)
    """
    if pd.isna(aspect_deg):
        return None
    a = aspect_deg % 360
    if 270 <= a < 360 or 0 <= a < 90:
        return "N"  # sjever
    else:  # 90 <= a < 270
        return "S"  # jug


def aspect_to_ew(aspect_deg):
    """Konvertuj aspect u ISTOK/ZAPAD.

    E (istok):  0-180°  (kroz 90°, tj. N→E→S)
    W (zapad): 180-360° (kroz 270°, tj. S→W→N)
    """
    if pd.isna(aspect_deg):
        return None
    a = aspect_deg % 360
    if 0 <= a < 180:
        return "E"  # istok
    else:  # 180 <= a < 360
        return "W"  # zapad


# ============================================================
#  UČITAVANJE
# ============================================================

def load_all():
    data = {}
    for layer in LAYERS:
        csv_path = os.path.join(INPUT_DIR, f"{layer}.csv")
        if not os.path.exists(csv_path):
            print(f"  NEDOSTAJE: {layer}.csv")
            continue
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if ASPECT_COL not in df.columns:
            print(f"  GREŠKA: stupac '{ASPECT_COL}' nije u {layer}.csv")
            continue

        aspect_series = pd.to_numeric(df[ASPECT_COL], errors="coerce")
        valid_aspects = aspect_series.dropna()

        data[layer] = {
            "aspect": aspect_series,
            "n_valid": len(valid_aspects),
            "n_null": aspect_series.isna().sum(),
            "total": len(df),
        }
        print(f"  {layer:<44}  n_valid={len(valid_aspects):<5}  null={aspect_series.isna().sum()}")
    return data


def load_background():
    bg_path = os.path.join(INPUT_DIR, "background_aspect.csv")
    if not os.path.exists(bg_path):
        print("  NEDOSTAJE: background_aspect.csv")
        return None
    return pd.read_csv(bg_path, encoding="utf-8-sig")

# ============================================================
#  KATEGORIJALIZACIJA I FREKVENCIJE
# ============================================================

def categorize_and_count(aspect_series, conversion_func):
    """Kategorijaliziraj aspect vrijednosti i broji frekvencije."""
    categories = aspect_series.apply(conversion_func)
    cat_valid = categories.dropna()
    if len(cat_valid) == 0:
        return {}
    freq = cat_valid.value_counts().to_dict()
    return freq


def freq_to_percentages(freq):
    """Konvertuj frekvencije u postotke."""
    total = sum(freq.values())
    if total == 0:
        return {}
    return {cat: round(count / total * 100, 2) for cat, count in freq.items()}


def background_to_cat(bg_df, conversion_func, category_order):
    """Konvertuj background histogram u kategorijske frekvencije."""
    if bg_df is None or len(bg_df) == 0:
        return {}

    aspect_vals = bg_df["aspect_value"].values.astype(float)
    n_piksela   = bg_df["n_piksela"].values.astype(float)

    freq = {cat: 0.0 for cat in category_order}
    for av, np_ in zip(aspect_vals, n_piksela):
        cat = conversion_func(av)
        if cat is not None:
            freq[cat] += np_

    return freq


# ============================================================
#   4-KATEGORIJE (NE, SE, SW, NW)
# ============================================================

def export_cat4(data):
    """Izvezi 4-kategorijsku analizu (NE, SE, SW, NW)."""
    print(f"\n{'='*60}")
    print("4-KATEGORIJE (NE, SE, SW, NW)")
    print(f"{'='*60}")

    category_order = ["NE", "SE", "SW", "NW"]

    for layer in LAYERS:
        if layer not in data:
            continue

        aspect_series = data[layer]["aspect"]
        freq = categorize_and_count(aspect_series, aspect_to_cat4)
        pct = freq_to_percentages(freq)

        # Ispuni nedostajuće kategorije s 0
        for cat in category_order:
            if cat not in freq:
                freq[cat] = 0
            if cat not in pct:
                pct[cat] = 0.0

        csv_path = os.path.join(OUTPUT_DIR, f"{layer}_cat4.csv")
        df_out = pd.DataFrame([
            {"kategor": cat, "n": int(freq[cat]), "postotak": pct[cat]}
            for cat in category_order
        ])
        df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  {layer}_cat4.csv")


def plot_cat4_bar(data, layer, out_path):
    """Bar plot 4-kategorija za jedan sloj."""
    category_order = ["NE", "SE", "SW", "NW"]
    aspect_series = data[layer]["aspect"]
    freq = categorize_and_count(aspect_series, aspect_to_cat4)

    for cat in category_order:
        if cat not in freq:
            freq[cat] = 0

    counts = [freq[cat] for cat in category_order]
    colors = [CAT4_COLORS[cat] for cat in category_order]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(category_order, counts, color=colors, edgecolor="white", linewidth=1, alpha=0.85)
    ax.set_title(f"{LAYER_LABELS.get(layer, layer)}", fontsize=11)
    ax.set_ylabel("Frekvencija")
    ax.set_xlabel("Smjer padine")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cat4_comparison(data, bg_df, out_path):
    """Stacked bar plot: pozadina + svi slojevi (4-kategorije)."""
    category_order = ["NE", "SE", "SW", "NW"]

    layers_with_data = [l for l in LAYERS if l in data and data[l]["n_valid"] > 0]
    all_groups = ["__background__"] + layers_with_data
    xlabels = [LAYER_LABELS.get(g, g) for g in all_groups]

    # Pripremi podatke za stacked bar
    data_for_plot = {cat: [] for cat in category_order}

    if bg_df is not None:
        bg_freq = background_to_cat(bg_df, aspect_to_cat4, category_order)
        bg_pct = freq_to_percentages(bg_freq)
        for cat in category_order:
            data_for_plot[cat].append(bg_pct[cat])
    else:
        for cat in category_order:
            data_for_plot[cat].append(0.0)

    for layer in layers_with_data:
        aspect_series = data[layer]["aspect"]
        freq = categorize_and_count(aspect_series, aspect_to_cat4)
        pct = freq_to_percentages(freq)
        for cat in category_order:
            data_for_plot[cat].append(pct.get(cat, 0.0))

    # Nacrtaj stacked bar
    fig, ax = plt.subplots(figsize=(max(10, len(all_groups) * 1.2), 6))

    bottom = np.zeros(len(all_groups))
    for cat in category_order:
        ax.bar(xlabels, data_for_plot[cat], bottom=bottom, label=cat,
               color=CAT4_COLORS[cat], edgecolor="white", linewidth=0.5, alpha=0.85)
        bottom += np.array(data_for_plot[cat])

    ax.set_ylabel("Udio (%)")
    ax.set_title("Smjer padine — 4 kategorije (NE, SE, SW, NW)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
#  SJEVER/JUG (N/S)
# ============================================================

def export_sn(data):
    """Izvezi S/N analizu."""
    print(f"\n{'='*60}")
    print("SJEVER/JUG (N/S)")
    print(f"{'='*60}")

    category_order = ["N", "S"]

    for layer in LAYERS:
        if layer not in data:
            continue

        aspect_series = data[layer]["aspect"]
        freq = categorize_and_count(aspect_series, aspect_to_sn)
        pct = freq_to_percentages(freq)

        for cat in category_order:
            if cat not in freq:
                freq[cat] = 0
            if cat not in pct:
                pct[cat] = 0.0

        csv_path = os.path.join(OUTPUT_DIR, f"{layer}_sn.csv")
        df_out = pd.DataFrame([
            {"kategor": ("Sjever" if cat == "N" else "Jug"), "n": int(freq[cat]), "postotak": pct[cat]}
            for cat in category_order
        ])
        df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  {layer}_sn.csv")


def plot_sn_bar(data, layer, out_path):
    """Bar plot N/S za jedan sloj."""
    category_order = ["N", "S"]
    labels_map = {"N": "Sjever", "S": "Jug"}
    aspect_series = data[layer]["aspect"]
    freq = categorize_and_count(aspect_series, aspect_to_sn)

    for cat in category_order:
        if cat not in freq:
            freq[cat] = 0

    counts = [freq[cat] for cat in category_order]
    labels = [labels_map[cat] for cat in category_order]
    colors = [SN_COLORS[cat] for cat in category_order]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=1, alpha=0.85)
    ax.set_title(f"{LAYER_LABELS.get(layer, layer)}", fontsize=11)
    ax.set_ylabel("Frekvencija")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sn_comparison(data, bg_df, out_path):
    """Stacked bar plot: pozadina + svi slojevi (S/N)."""
    category_order = ["N", "S"]
    labels_map = {"N": "Sjever", "S": "Jug"}

    layers_with_data = [l for l in LAYERS if l in data and data[l]["n_valid"] > 0]
    all_groups = ["__background__"] + layers_with_data
    xlabels = [LAYER_LABELS.get(g, g) for g in all_groups]

    data_for_plot = {cat: [] for cat in category_order}

    if bg_df is not None:
        bg_freq = background_to_cat(bg_df, aspect_to_sn, category_order)
        bg_pct = freq_to_percentages(bg_freq)
        for cat in category_order:
            data_for_plot[cat].append(bg_pct[cat])
    else:
        for cat in category_order:
            data_for_plot[cat].append(0.0)

    for layer in layers_with_data:
        aspect_series = data[layer]["aspect"]
        freq = categorize_and_count(aspect_series, aspect_to_sn)
        pct = freq_to_percentages(freq)
        for cat in category_order:
            data_for_plot[cat].append(pct.get(cat, 0.0))

    fig, ax = plt.subplots(figsize=(max(10, len(all_groups) * 1.2), 6))

    bottom = np.zeros(len(all_groups))
    for cat in category_order:
        ax.bar(xlabels, data_for_plot[cat], bottom=bottom, label=labels_map[cat],
               color=SN_COLORS[cat], edgecolor="white", linewidth=0.5, alpha=0.85)
        bottom += np.array(data_for_plot[cat])

    ax.set_ylabel("Udio (%)")
    ax.set_title("Smjer padine — Sjever / Jug", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
#  ISTOK/ZAPAD (E/W)
# ============================================================

def export_ew(data):
    """Izvezi E/W analizu."""
    print(f"\n{'='*60}")
    print("ISTOK/ZAPAD (E/W)")
    print(f"{'='*60}")

    category_order = ["E", "W"]

    for layer in LAYERS:
        if layer not in data:
            continue

        aspect_series = data[layer]["aspect"]
        freq = categorize_and_count(aspect_series, aspect_to_ew)
        pct = freq_to_percentages(freq)

        for cat in category_order:
            if cat not in freq:
                freq[cat] = 0
            if cat not in pct:
                pct[cat] = 0.0

        csv_path = os.path.join(OUTPUT_DIR, f"{layer}_ew.csv")
        df_out = pd.DataFrame([
            {"kategor": ("Istok" if cat == "E" else "Zapad"), "n": int(freq[cat]), "postotak": pct[cat]}
            for cat in category_order
        ])
        df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  {layer}_ew.csv")


def plot_ew_bar(data, layer, out_path):
    """Bar plot E/W za jedan sloj."""
    category_order = ["E", "W"]
    labels_map = {"E": "Istok", "W": "Zapad"}
    aspect_series = data[layer]["aspect"]
    freq = categorize_and_count(aspect_series, aspect_to_ew)

    for cat in category_order:
        if cat not in freq:
            freq[cat] = 0

    counts = [freq[cat] for cat in category_order]
    labels = [labels_map[cat] for cat in category_order]
    colors = [EW_COLORS[cat] for cat in category_order]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=1, alpha=0.85)
    ax.set_title(f"{LAYER_LABELS.get(layer, layer)}", fontsize=11)
    ax.set_ylabel("Frekvencija")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ew_comparison(data, bg_df, out_path):
    """Stacked bar plot: pozadina + svi slojevi (E/W)."""
    category_order = ["E", "W"]
    labels_map = {"E": "Istok", "W": "Zapad"}

    layers_with_data = [l for l in LAYERS if l in data and data[l]["n_valid"] > 0]
    all_groups = ["__background__"] + layers_with_data
    xlabels = [LAYER_LABELS.get(g, g) for g in all_groups]

    data_for_plot = {cat: [] for cat in category_order}

    if bg_df is not None:
        bg_freq = background_to_cat(bg_df, aspect_to_ew, category_order)
        bg_pct = freq_to_percentages(bg_freq)
        for cat in category_order:
            data_for_plot[cat].append(bg_pct[cat])
    else:
        for cat in category_order:
            data_for_plot[cat].append(0.0)

    for layer in layers_with_data:
        aspect_series = data[layer]["aspect"]
        freq = categorize_and_count(aspect_series, aspect_to_ew)
        pct = freq_to_percentages(freq)
        for cat in category_order:
            data_for_plot[cat].append(pct.get(cat, 0.0))

    fig, ax = plt.subplots(figsize=(max(10, len(all_groups) * 1.2), 6))

    bottom = np.zeros(len(all_groups))
    for cat in category_order:
        ax.bar(xlabels, data_for_plot[cat], bottom=bottom, label=labels_map[cat],
               color=EW_COLORS[cat], edgecolor="white", linewidth=0.5, alpha=0.85)
        bottom += np.array(data_for_plot[cat])

    ax.set_ylabel("Udio (%)")
    ax.set_title("Smjer padine — Istok / Zapad", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
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
    print("ASPECT - učitavanje CSV-a")
    print("=" * 60)
    data  = load_all()
    bg_df = load_background()

    if not data:
        print("Nema podataka — provjeri INPUT_DIR i pokreni aspect.py u QGIS-u.")
        return

    # ── 4-kategorije ────────────────────────────────────────
    export_cat4(data)
    print(f"\nBar plotovi — 4-kategorije:")
    for layer in LAYERS:
        if layer not in data or data[layer]["n_valid"] == 0:
            continue
        out_path = os.path.join(OUTPUT_DIR, f"{layer}_cat4_bar.png")
        plot_cat4_bar(data, layer, out_path)
        print(f"  {layer}_cat4_bar.png")

    plot_cat4_comparison(data, bg_df, os.path.join(OUTPUT_DIR, "comparison_cat4.png"))
    print("  comparison_cat4.png")

    # ── Sjever/Jug ──────────────────────────────────────────
    export_sn(data)
    print(f"\nBar plotovi — Sjever/Jug:")
    for layer in LAYERS:
        if layer not in data or data[layer]["n_valid"] == 0:
            continue
        out_path = os.path.join(OUTPUT_DIR, f"{layer}_sn_bar.png")
        plot_sn_bar(data, layer, out_path)
        print(f"  {layer}_sn_bar.png")

    plot_sn_comparison(data, bg_df, os.path.join(OUTPUT_DIR, "comparison_sn.png"))
    print("  comparison_sn.png")

    # ── Istok/Zapad ─────────────────────────────────────────
    export_ew(data)
    print(f"\nBar plotovi — Istok/Zapad:")
    for layer in LAYERS:
        if layer not in data or data[layer]["n_valid"] == 0:
            continue
        out_path = os.path.join(OUTPUT_DIR, f"{layer}_ew_bar.png")
        plot_ew_bar(data, layer, out_path)
        print(f"  {layer}_ew_bar.png")

    plot_ew_comparison(data, bg_df, os.path.join(OUTPUT_DIR, "comparison_ew.png"))
    print("  comparison_ew.png")

    print(f"\n{'='*60}")
    print(f"GOTOVO!  Izlaz: {OUTPUT_DIR}")
    print(f"{'='*60}")


# ============================================================
#  2D STRIP PLOTOVI — ASPECT KATEGORIJA vs INTENZITET NAGIBA
# ============================================================

def _load_nagib_df(layer_name):
    """Ucitaj CSV, filtriraj tocke s nagibom < 250, dodaj intenzitet."""
    csv_path = os.path.join(INPUT_DIR, f"{layer_name}.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["aspect"] = pd.to_numeric(df["aspect"], errors="coerce")
    df["nagib"]  = pd.to_numeric(df["nagib"],  errors="coerce")
    df = df[(df["nagib"] < 250) & df["aspect"].notna()].copy()
    if df.empty:
        return None
    df["intenzitet"] = 250.0 - df["nagib"]
    return df


def _strip_plot(ax, groups, colors, title, xlabel):
    """Genericki strip plot: groups je dict {label: array_of_values}."""
    rng = np.random.default_rng(42)
    positions = list(range(len(groups)))

    for pos, (label, vals) in zip(positions, groups.items()):
        if len(vals) == 0:
            continue
        color = colors.get(label, "#555555")
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals,
                   color=color, s=50, edgecolors="k",
                   linewidths=0.4, alpha=0.80, zorder=3)
        # Medijan kao horizontalna crta
        med = float(np.median(vals))
        ax.plot([pos - 0.3, pos + 0.3], [med, med],
                color="#222222", linewidth=2.0, zorder=4)

    ax.set_xticks(positions)
    ax.set_xticklabels(list(groups.keys()), fontsize=10)
    ax.set_ylabel("Intenzitet nagiba  (250 − nagib)", fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_strip_cat4(layer_name):
    """Strip plot: 4 kategorije (NE/SE/SW/NW) vs intenzitet nagiba."""
    df = _load_nagib_df(layer_name)
    if df is None:
        print(f"  {layer_name}: nema podataka — preskacam")
        return

    df["cat4"] = df["aspect"].apply(aspect_to_cat4)
    order  = ["NE", "SE", "SW", "NW"]
    groups = {cat: df.loc[df["cat4"] == cat, "intenzitet"].values for cat in order}

    fig, ax = plt.subplots(figsize=(6, 5))
    _strip_plot(ax, groups, CAT4_COLORS,
                title=f"Smjer padine — intenzitet nagiba\n{LAYER_LABELS.get(layer_name, layer_name)}  (n={len(df)})",
                xlabel="Kategorija aspekta")
    fig.savefig(os.path.join(OUTPUT_DIR, f"{layer_name}_strip_cat4.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {layer_name}_strip_cat4.png  (n={len(df)})")


def plot_strip_sn(layer_name):
    """Strip plot: Sjever/Jug vs intenzitet nagiba."""
    df = _load_nagib_df(layer_name)
    if df is None:
        print(f"  {layer_name}: nema podataka — preskacam")
        return

    df["sn"] = df["aspect"].apply(aspect_to_sn)
    groups = {
        "Sjever": df.loc[df["sn"] == "N", "intenzitet"].values,
        "Jug":    df.loc[df["sn"] == "S", "intenzitet"].values,
    }

    fig, ax = plt.subplots(figsize=(5, 5))
    _strip_plot(ax, groups, {"Sjever": SN_COLORS["N"], "Jug": SN_COLORS["S"]},
                title=f"Sjever / Jug — intenzitet nagiba\n{LAYER_LABELS.get(layer_name, layer_name)}  (n={len(df)})",
                xlabel="")
    fig.savefig(os.path.join(OUTPUT_DIR, f"{layer_name}_strip_sn.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {layer_name}_strip_sn.png  (n={len(df)})")


def plot_strip_ew(layer_name):
    """Strip plot: Istok/Zapad vs intenzitet nagiba."""
    df = _load_nagib_df(layer_name)
    if df is None:
        print(f"  {layer_name}: nema podataka — preskacam")
        return

    df["ew"] = df["aspect"].apply(aspect_to_ew)
    groups = {
        "Istok": df.loc[df["ew"] == "E", "intenzitet"].values,
        "Zapad": df.loc[df["ew"] == "W", "intenzitet"].values,
    }

    fig, ax = plt.subplots(figsize=(5, 5))
    _strip_plot(ax, groups, {"Istok": EW_COLORS["E"], "Zapad": EW_COLORS["W"]},
                title=f"Istok / Zapad — intenzitet nagiba\n{LAYER_LABELS.get(layer_name, layer_name)}  (n={len(df)})",
                xlabel="")
    fig.savefig(os.path.join(OUTPUT_DIR, f"{layer_name}_strip_ew.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {layer_name}_strip_ew.png  (n={len(df)})")


def run_strip():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("STRIP PLOTOVI — ASPECT KATEGORIJA vs INTENZITET NAGIBA")
    print("=" * 60)
    print("\n4-kategorije (NE/SE/SW/NW):")
    for layer in LAYERS:
        plot_strip_cat4(layer)
    print("\nSjever/Jug:")
    for layer in LAYERS:
        plot_strip_sn(layer)
    print("\nIstok/Zapad:")
    for layer in LAYERS:
        plot_strip_ew(layer)
    print(f"\nIzlaz: {OUTPUT_DIR}")


# run()        # zakomentiraj za strip mode
run_strip()
