# -*- coding: utf-8 -*-
"""
RES_SUHO_MOCVARNO - Binarna klasifikacija tla (mocvarno vs suho)
=================================================================
Ulaz:  32 CSV-a iz vecinski_tip_tla.py   (fid, vecinski_tip_tla)
       mapa: deskriptivna/vecinski_tip_tla_/csv_output/

Klasifikacija:
  Mocvarno = Gleysols, Fluvisols, Vertisols
  Suho     = sve ostalo

Izlaz (64 datoteke) u deskriptivna/res_output/suho_mocvarno/:
  {naziv}_freq.csv   - frekvencijska tablica (n, %)
  {naziv}_bar.png    - stupcasti dijagram

Pokretanje: python res_suho_mocvarno.py
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
#  POSTAVKE
# ============================================================

INPUT_DIR  = r"C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna\vecinski_tip_tla_\csv_output"
OUTPUT_DIR = r"C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna\res_output\suho_mocvarno"

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

MOCVARNA_TLA = {"Gleysols", "Fluvisols", "Vertisols"}

COLORS = {
    "Mocvarno": "#4472C4",
    "Suho":     "#B8860B",
}

# ============================================================
#  POMOCNE FUNKCIJE
# ============================================================

def klasificiraj(tip_tla):
    if tip_tla in MOCVARNA_TLA:
        return "Mocvarno"
    return "Suho"


def freq_table(series):
    counts = series.value_counts()
    total  = len(series)
    df = pd.DataFrame({
        "kategorija": counts.index,
        "n":          counts.values,
        "postotak":   (counts.values / total * 100).round(2),
    })
    order = [k for k in ["Mocvarno", "Suho"] if k in df["kategorija"].values]
    df = df.set_index("kategorija").loc[order].reset_index()
    return df


def plot_bar(freq_df, title, out_path):
    labels   = freq_df["kategorija"].tolist()
    counts   = freq_df["n"].tolist()
    percents = freq_df["postotak"].tolist()
    colors   = [COLORS.get(lbl, "#999999") for lbl in labels]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=0.8, width=0.45)

    for bar, pct in zip(bars, percents):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=10, color="#333333",
        )

    ax.set_title(title, fontsize=10, pad=10)
    ax.set_ylabel("Broj točaka (n)")
    ax.set_xlabel("Tip tla")
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

            df["kategorija"] = df["vecinski_tip_tla"].fillna("Nepoznato").apply(klasificiraj)
            freq  = freq_table(df["kategorija"])
            n_tot = len(df)

            print(f"\n  Polumjer {radius} m   (n = {n_tot})")
            print(f"  {'Kategorija':<12} {'n':>5}  {'%':>6}")
            print(f"  {'-'*26}")
            for _, row in freq.iterrows():
                print(f"  {row['kategorija']:<12} {int(row['n']):>5}  {row['postotak']:>5.1f}%")

            base = f"{layer}_sm_r{radius}"

            freq_path = os.path.join(OUTPUT_DIR, f"{base}_freq.csv")
            freq.to_csv(freq_path, index=False, encoding="utf-8-sig")

            bar_path = os.path.join(OUTPUT_DIR, f"{base}_bar.png")
            title = f"{LAYER_LABELS.get(layer, layer)}  |  polumjer {radius} m  (n = {n_tot})"
            plot_bar(freq, title, bar_path)

            total_ok += 1

    print(f"\n{'='*60}")
    print(f"GOTOVO!  Obradjeno: {total_ok}   Preskoceno: {total_skip}")
    print(f"Izlaz: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run()
