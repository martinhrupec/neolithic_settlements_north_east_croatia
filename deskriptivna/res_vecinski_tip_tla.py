# -*- coding: utf-8 -*-
"""
RES_VECINSKI_TIP_TLA - Deskriptivna analiza dominantnog tipa tla
=================================================================
Ulaz:  32 CSV-a iz vecinski_tip_tla.py   (stupci: fid, vecinski_tip_tla)
Izlaz: za svaki CSV
         - ispis moda i frekvencijske tablice u konzolu
         - {naziv}_freq.csv   - frekvencijska tablica (n, %)
         - {naziv}_bar.png    - stupcasti dijagram

Pokretanje: python res_vecinski_tip_tla.py
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless - sprema PNG, ne otvara prozor
import matplotlib.pyplot as plt

# ============================================================
#  POSTAVKE
# ============================================================

INPUT_DIR  = r"C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna\vecinski_tip_tla_\csv_output"
OUTPUT_DIR = r"C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna\res_output"

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
    "random_ceste_biased":                     "Random (ceste)",
    "nasumicni_lokaliteti_umjetno_generirani": "Random (nasumični)",
    "neolitik_svi_odredeni":                   "Neolitik (svi određeni)",
    "neolitik_c_starcevacka":                  "Rani neolitik",
    "neolitik_c_sop_kor_len":                  "Srednji i kasni neolitik",
    "kontinuirana_naselja":                    "Kontinuirana naselja",
    "samo_rani":                               "Isključivo rana faza",
    "samo_srednji_kasni":                      "Isključivo srednja i kasna faza",
}

# Konzistentne boje po tipu tla kroz sve dijagrame
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

# ============================================================
#  POMOCNE FUNKCIJE
# ============================================================

def freq_table(series):
    """Vrati DataFrame s kolonama: tip_tla, n, postotak."""
    counts = series.value_counts()
    total  = len(series)
    df = pd.DataFrame({
        "tip_tla":   counts.index,
        "n":         counts.values,
        "postotak":  (counts.values / total * 100).round(2),
    })
    return df


def plot_bar(freq_df, title, out_path):
    """Spremi stupcasti dijagram kao PNG."""
    labels  = freq_df["tip_tla"].tolist()
    counts  = freq_df["n"].tolist()
    percents = freq_df["postotak"].tolist()
    colors  = [SOIL_COLORS.get(t, "#999999") for t in labels]

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.9), 5))
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=0.6)

    # postotak iznad svake sipke
    for bar, pct in zip(bars, percents):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=8, color="#333333",
        )

    ax.set_title(title, fontsize=11, pad=10)
    ax.set_ylabel("Broj točaka (n)")
    ax.set_xlabel("Tip tla (WRB)")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
#  GLAVNA ANALIZA
# ============================================================

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_ok   = 0
    total_skip = 0

    for layer in LAYERS:
        print(f"\n{'='*60}")
        print(f"  {layer}")
        print(f"{'='*60}")

        for radius in RADII:
            csv_name = f"{layer}_vtt_r{radius}.csv"
            csv_path = os.path.join(INPUT_DIR, csv_name)

            if not os.path.exists(csv_path):
                print(f"  r={radius:4d}m  NEDOSTAJE: {csv_name}")
                total_skip += 1
                continue

            df = pd.read_csv(csv_path, encoding="utf-8-sig")

            if "vecinski_tip_tla" not in df.columns:
                print(f"  r={radius:4d}m  GRESKA: stupac 'vecinski_tip_tla' nije u {csv_name}")
                total_skip += 1
                continue

            series = df["vecinski_tip_tla"].fillna("Nepoznato")
            freq   = freq_table(series)
            mode   = freq.iloc[0]["tip_tla"]
            n_tot  = len(series)

            # --- Konzolni ispis ---
            print(f"\n  Polumjer {radius} m   (n = {n_tot})")
            print(f"  Mod: {mode} ({freq.iloc[0]['postotak']:.1f}%)")
            print(f"  {'Tip tla':<16} {'n':>5}  {'%':>6}")
            print(f"  {'-'*32}")
            for _, row in freq.iterrows():
                print(f"  {row['tip_tla']:<16} {int(row['n']):>5}  {row['postotak']:>5.1f}%")

            base = f"{layer}_vtt_r{radius}"

            # --- Spremi frekvencijsku tablicu ---
            freq_path = os.path.join(OUTPUT_DIR, f"{base}_freq.csv")
            freq.to_csv(freq_path, index=False, encoding="utf-8-sig")

            # --- Spremi bar chart ---
            bar_path = os.path.join(OUTPUT_DIR, f"{base}_bar.png")
            title    = f"{LAYER_LABELS.get(layer, layer)}  |  polumjer {radius} m  (n = {n_tot})"
            plot_bar(freq, title, bar_path)

            total_ok += 1

    print(f"\n{'='*60}")
    print(f"GOTOVO!  Obradjeno: {total_ok}   Preskoceno: {total_skip}")
    print(f"Izlaz: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run()
