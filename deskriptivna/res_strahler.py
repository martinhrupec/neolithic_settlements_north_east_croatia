# -*- coding: utf-8 -*-
"""
RES_STRAHLER - Strahlerov red najblize tekucice (kategorijska varijabla)
=========================================================================
Ulaz:  8 CSV-a iz tekucice.py  (fid, strahler)
       + background_strahler.csv  (strahler, duljina_km, postotak)
       mapa: deskriptivna/strahler_/csv_output/

Analiza I — puni Strahlerov red (1-7):
  {naziv}_freq.csv            - frekvencijska tablica
  {naziv}_bar.png             - stupcastii dijagram
  comparison_strahler.png     - slozeni bar chart: pozadina + 8 slojeva

Analiza II — binarna klasifikacija (male 1-4 vs velike 5-7 rijeke):
  {naziv}_vel_freq.csv        - frekvencijska tablica
  {naziv}_vel_bar.png         - stupcasti dijagram
  comparison_vel.png          - slozeni bar chart: pozadina + 8 slojeva

Pokretanje: python res_strahler.py
"""

import os
import csv as csv_module
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
#  POSTAVKE
# ============================================================

INPUT_DIR  = r"C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna\strahler_\csv_output"
OUTPUT_DIR = r"C:\Users\Martin\Desktop\skripte_za_diplomski\deskriptivna\res_output\strahler"

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
    "random_ceste_biased":                     "Random (ceste)",
    "nasumicni_lokaliteti_umjetno_generirani": "Random (nasumični)",
    "neolitik_svi_odredeni":                   "Neolitik (svi)",
    "neolitik_c_starcevacka":                  "Rani neolitik",
    "neolitik_c_sop_kor_len":                  "Srednji i kasni neolitik",
    "kontinuirana_naselja":                    "Kontinuirana naselja",
    "samo_rani":                               "Isključivo rana faza",
    "samo_srednji_kasni":                      "Isključivo srednja i kasna faza",
    "__background__":                          "Pozadina (cijelo područje)",
}

# Boje za Strahlerov red 1-7 (nijanse od svjetlije do tamnije plave)
STRAHLER_COLORS = {
    1: "#C6E0F5",
    2: "#93C4E8",
    3: "#5FA8DB",
    4: "#2E75B6",
    5: "#1F4E79",
    6: "#0D2B44",
    7: "#050F18",
}

# Boje za binarnu klasifikaciju
VEL_COLORS = {
    "Male (1-4)":   "#70AD47",
    "Velike (5-7)": "#2E75B6",
}

MALE_RIJEKE   = {1, 2, 3, 4}
VELIKE_RIJEKE = {5, 6, 7}

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
        if "strahler" not in df.columns:
            print(f"  GREŠKA: stupac 'strahler' nije u {layer}.csv")
            continue
        series = pd.to_numeric(df["strahler"], errors="coerce").dropna().astype(int)
        data[layer] = series
        print(f"  {layer:<44}  n={len(series)}")
    return data


def load_background_strahler():
    bg_path = os.path.join(INPUT_DIR, "background_strahler.csv")
    if not os.path.exists(bg_path):
        print("  NEDOSTAJE: background_strahler.csv")
        return None
    df = pd.read_csv(bg_path, encoding="utf-8-sig")
    total = df["postotak"].sum()
    return {int(row["strahler"]): float(row["postotak"]) for _, row in df.iterrows()}

# ============================================================
#  POMOCNE FUNKCIJE
# ============================================================

def strahler_freq(series, all_orders=range(1, 8)):
    """Vrati dict {order: postotak} za sve poznate redove."""
    total = len(series)
    counts = series.value_counts()
    return {o: round(counts.get(o, 0) / total * 100, 2) for o in all_orders}


def vel_freq(series):
    """Binarna klasifikacija: Male (1-4) vs Velike (5-7)."""
    total = len(series)
    male   = series[series.isin(MALE_RIJEKE)].count()
    velike = series[series.isin(VELIKE_RIJEKE)].count()
    return {
        "Male (1-4)":   round(male   / total * 100, 2),
        "Velike (5-7)": round(velike / total * 100, 2),
    }

# ============================================================
#  VIZUALIZACIJA — STRAHLER (1-7)
# ============================================================

def plot_strahler_bar(freq, layer_name, n, out_path):
    orders = [o for o in range(1, 8) if freq.get(o, 0) > 0]
    values = [freq.get(o, 0) for o in orders]
    colors = [STRAHLER_COLORS.get(o, "#888888") for o in orders]
    labels = [str(o) for o in orders]

    fig, ax = plt.subplots(figsize=(max(5, len(orders) * 0.9), 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.7)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_title(f"{LAYER_LABELS.get(layer_name, layer_name)}  (n={n})", fontsize=10, pad=10)
    ax.set_xlabel("Strahlerov red")
    ax.set_ylabel("Udio točaka (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_vel_bar(freq, layer_name, n, out_path):
    labels = ["Male (1-4)", "Velike (5-7)"]
    values = [freq.get(l, 0) for l in labels]
    colors = [VEL_COLORS[l] for l in labels]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8, width=0.45)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_title(f"{LAYER_LABELS.get(layer_name, layer_name)}  (n={n})", fontsize=10, pad=10)
    ax.set_xlabel("Veličina rijeke")
    ax.set_ylabel("Udio točaka (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ============================================================
#  VIZUALIZACIJA — USPOREDBA S POZADINOM
# ============================================================

def plot_comparison_strahler(data, bg_freq, out_path):
    """Slozeni horizontalni bar chart: pozadina + 8 slojeva, Strahler 1-7."""
    all_orders = sorted(set(STRAHLER_COLORS.keys()) |
                        {o for s in data.values() for o in s.unique()} |
                        set(bg_freq.keys()))

    groups  = ["__background__"] + list(LAYERS)
    present = ["__background__"] + [l for l in LAYERS if l in data]
    ylabels = [LAYER_LABELS.get(g, g) for g in present]
    n_groups = len(present)

    # Izgradi freq dict za svaku grupu
    freqs = {"__background__": bg_freq}
    for layer in LAYERS:
        if layer in data:
            freqs[layer] = strahler_freq(data[layer], all_orders)

    fig, ax = plt.subplots(figsize=(11, max(5, n_groups * 0.65)))
    lefts = [0.0] * n_groups

    for order in sorted(all_orders):
        vals  = [freqs[g].get(order, 0) for g in present]
        color = STRAHLER_COLORS.get(order, "#888888")
        ax.barh(range(n_groups), vals, left=lefts, color=color,
                edgecolor="white", linewidth=0.4, label=f"Strahler {order}")
        for i, (v, l) in enumerate(zip(vals, lefts)):
            if v >= 8:
                ax.text(l + v / 2, i, f"{v:.0f}%",
                        ha="center", va="center", fontsize=7, color="white", fontweight="bold")
        lefts = [l + v for l, v in zip(lefts, vals)]

    ax.set_yticks(range(n_groups))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xlabel("Udio (%)")
    ax.set_title("Strahlerov red — pozadina vs. slojevi", fontsize=11)
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_vel(data, bg_freq, out_path):
    """Slozeni horizontalni bar chart: binarna klasifikacija, pozadina + 8 slojeva."""
    # bg_freq je {strahler_int: postotak} → trebamo sumu za male i velike
    bg_male   = sum(bg_freq.get(o, 0) for o in MALE_RIJEKE)
    bg_velike = sum(bg_freq.get(o, 0) for o in VELIKE_RIJEKE)
    # Normaliziraj na 100% (može biti <100 ako postoje redovi izvan 1-7)
    bg_tot = bg_male + bg_velike
    if bg_tot > 0:
        bg_male   = bg_male   / bg_tot * 100
        bg_velike = bg_velike / bg_tot * 100

    present = ["__background__"] + [l for l in LAYERS if l in data]
    ylabels = [LAYER_LABELS.get(g, g) for g in present]
    n_groups = len(present)

    freqs_vel = {"__background__": {"Male (1-4)": bg_male, "Velike (5-7)": bg_velike}}
    for layer in LAYERS:
        if layer in data:
            freqs_vel[layer] = vel_freq(data[layer])

    fig, ax = plt.subplots(figsize=(9, max(5, n_groups * 0.65)))
    lefts = [0.0] * n_groups

    for cat in ["Male (1-4)", "Velike (5-7)"]:
        vals  = [freqs_vel[g].get(cat, 0) for g in present]
        color = VEL_COLORS[cat]
        ax.barh(range(n_groups), vals, left=lefts, color=color,
                edgecolor="white", linewidth=0.4, label=cat)
        for i, (v, l) in enumerate(zip(vals, lefts)):
            if v >= 8:
                ax.text(l + v / 2, i, f"{v:.0f}%",
                        ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        lefts = [l + v for l, v in zip(lefts, vals)]

    ax.set_yticks(range(n_groups))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xlabel("Udio (%)")
    ax.set_title("Male (1-4) vs Velike (5-7) rijeke — pozadina vs. slojevi", fontsize=11)
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right", fontsize=8)
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
    print("STRAHLER RED - učitavanje CSV-a")
    print("=" * 60)
    data    = load_all()
    bg_freq = load_background_strahler()

    if not data:
        print("Nema podataka — provjeri INPUT_DIR i pokreni tekucice.py u QGIS-u.")
        return

    # ── Analiza I: Strahler 1-7 ─────────────────────────────
    print(f"\n{'='*60}")
    print("ANALIZA I — STRAHLEROV RED (1-7)")
    print(f"{'='*60}")

    for layer in LAYERS:
        if layer not in data:
            continue
        series = data[layer]
        freq   = strahler_freq(series)
        n      = len(series)

        print(f"\n  {layer}  (n={n})")
        for o in range(1, 8):
            if freq[o] > 0:
                print(f"    Strahler {o}: {freq[o]:.1f}%")

        freq_df = pd.DataFrame([
            {"strahler": o, "n": int(round(freq[o] * n / 100)), "postotak": freq[o]}
            for o in range(1, 8)
        ])
        freq_df.to_csv(os.path.join(OUTPUT_DIR, f"{layer}_freq.csv"),
                       index=False, encoding="utf-8-sig")

        plot_strahler_bar(freq, layer, n,
                          os.path.join(OUTPUT_DIR, f"{layer}_bar.png"))

    print(f"\n  → {len(data)} freq CSV + bar PNG para")

    # ── Analiza II: Male vs Velike ───────────────────────────
    print(f"\n{'='*60}")
    print("ANALIZA II — MALE (1-4) vs VELIKE (5-7) RIJEKE")
    print(f"{'='*60}")

    for layer in LAYERS:
        if layer not in data:
            continue
        series = data[layer]
        freq   = vel_freq(series)
        n      = len(series)

        print(f"  {layer:<44}  Male={freq['Male (1-4)']:.1f}%  "
              f"Velike={freq['Velike (5-7)']:.1f}%")

        vel_df = pd.DataFrame([
            {"kategorija": k, "n": int(round(v * n / 100)), "postotak": v}
            for k, v in freq.items()
        ])
        vel_df.to_csv(os.path.join(OUTPUT_DIR, f"{layer}_vel_freq.csv"),
                      index=False, encoding="utf-8-sig")

        plot_vel_bar(freq, layer, n,
                     os.path.join(OUTPUT_DIR, f"{layer}_vel_bar.png"))

    print(f"\n  → {len(data)} vel_freq CSV + vel_bar PNG para")

    # ── Usporedba s pozadinom ────────────────────────────────
    if bg_freq is not None:
        print(f"\n{'='*60}")
        print("USPOREDBA S POZADINOM")
        print(f"{'='*60}")

        print("  Pozadinski Strahler (% km):")
        for o in sorted(bg_freq.keys()):
            print(f"    Strahler {o}: {bg_freq[o]:.1f}%")

        plot_comparison_strahler(data, bg_freq,
                                 os.path.join(OUTPUT_DIR, "comparison_strahler.png"))
        print("  comparison_strahler.png")

        plot_comparison_vel(data, bg_freq,
                            os.path.join(OUTPUT_DIR, "comparison_vel.png"))
        print("  comparison_vel.png")
    else:
        print("\n  Usporedba preskočena (nema background_strahler.csv)")

    print(f"\n{'='*60}")
    print(f"GOTOVO!  Izlaz: {OUTPUT_DIR}")
    print(f"{'='*60}")


run()
