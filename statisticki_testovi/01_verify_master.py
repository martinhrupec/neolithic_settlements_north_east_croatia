"""
01_verify_master.py
Provjerava poklapaju li se deskriptivne statistike iz master_dataset.csv
s referentnim summary_stats.csv koje vec postoje u deskriptivna/res_output/.

Ne sprema rezultate - samo printa razlike i upozorava ako gdje ne poklapa.
"""

import os
import pandas as pd


ROOT  = r"c:\Users\Martin\Desktop\skripte_za_diplomski"
RES   = os.path.join(ROOT, "deskriptivna", "res_output")
MAS   = os.path.join(ROOT, "statisticki_testovi", "master_dataset.csv")

# referentni naziv sloja -> filter na master
LAYER_MAP = {
    "random_ceste_biased":                    lambda d: d[d.tip_sloja == "nasumicni_ceste"],
    "nasumicni_lokaliteti_umjetno_generirani": lambda d: d[d.tip_sloja == "potpuno_nasumicni"],
    "neolitik_svi_odredeni":                  lambda d: d[d.tip_sloja == "neolitik"],
    "samo_rani":                              lambda d: d[(d.tip_sloja == "neolitik") & d.samo_rano],
    "samo_srednji_kasni":                     lambda d: d[(d.tip_sloja == "neolitik") & d.samo_kasno],
    "kontinuirana_naselja":                   lambda d: d[(d.tip_sloja == "neolitik") & d.kontinuirano],
}

TOL = 0.01  # tolerancija usporedbe (apsolutna)


def cmp(ref, got, label):
    if pd.isna(ref) and pd.isna(got):
        return None
    if pd.isna(ref) or pd.isna(got):
        return f"   X {label}: ref={ref}  master={got}"
    diff = abs(ref - got)
    if diff > TOL:
        return f"   X {label}: ref={ref:.4f}  master={got:.4f}  delta={diff:.4f}"
    return None


def check_simple(master, ref_csv, master_col, label):
    """Standardni summary_stats.csv: stupci sloj, n, mean, median, min, max."""
    ref = pd.read_csv(ref_csv)
    print(f"\n--- {label}  ({os.path.basename(os.path.dirname(ref_csv))}) ---")
    mismatches = 0
    for _, row in ref.iterrows():
        sloj = row["sloj"]
        if sloj not in LAYER_MAP:
            continue
        sub = LAYER_MAP[sloj](master)[master_col].dropna()
        msgs = [
            cmp(row["n"],      len(sub),         "n"),
            cmp(row["mean"],   sub.mean(),       "mean"),
            cmp(row["median"], sub.median(),     "median"),
            cmp(row["min"],    sub.min(),        "min"),
            cmp(row["max"],    sub.max(),        "max"),
        ]
        bad = [m for m in msgs if m]
        if bad:
            print(f"  {sloj}:")
            for m in bad:
                print(m)
                mismatches += 1
        else:
            print(f"  {sloj:42s}  OK  (n={len(sub)})")
    return mismatches


def check_keyed(master, ref_csv, key_col, master_col_template, label):
    """Summary s dodatnim kljucnim stupcem (radijus/kombinacija)."""
    ref = pd.read_csv(ref_csv)
    print(f"\n--- {label}  ({os.path.basename(os.path.dirname(ref_csv))}) ---")
    mismatches = 0
    for _, row in ref.iterrows():
        sloj = row["sloj"]
        if sloj not in LAYER_MAP:
            continue
        key = row[key_col]
        col = master_col_template(key)
        if col not in master.columns:
            continue
        sub = LAYER_MAP[sloj](master)[col].dropna()
        msgs = [
            cmp(row["n"],      len(sub),         "n"),
            cmp(row["mean"],   sub.mean(),       "mean"),
            cmp(row["median"], sub.median(),     "median"),
            cmp(row["min"],    sub.min(),        "min"),
            cmp(row["max"],    sub.max(),        "max"),
        ]
        bad = [m for m in msgs if m]
        if bad:
            print(f"  {sloj} [{key_col}={key}]:")
            for m in bad:
                print(m)
                mismatches += 1
        else:
            print(f"  {sloj:42s} {key_col}={str(key):12s}  OK  (n={len(sub)})")
    return mismatches


def main():
    master = pd.read_csv(MAS)
    print(f"master_dataset.csv: {len(master)} redaka, stupci: {len(master.columns)}")

    total = 0
    total += check_simple(master, os.path.join(RES, "apsolutna_visina", "summary_stats.csv"),
                          "aps_vis", "apsolutna visina")
    total += check_simple(master, os.path.join(RES, "coarse_fragments", "summary_stats.csv"),
                          "coarse_fragments", "coarse fragments")
    total += check_simple(master, os.path.join(RES, "dist_rijeka", "summary_stats.csv"),
                          "dist_rijeka", "dist_rijeka (ORIGINAL)")
    total += check_simple(master, os.path.join(RES, "tri", "summary_stats.csv"),
                          "tri", "TRI")

    total += check_keyed(master,
                         os.path.join(RES, "gustoca_rijeka", "summary_stats.csv"),
                         "radijus",
                         lambda r: f"gustoca_rijeka_{int(r)}",
                         "gustoca rijeka")
    total += check_keyed(master,
                         os.path.join(RES, "relativna_visina", "summary_stats.csv"),
                         "kombinacija",
                         lambda k: f"rel_vis_{k.replace(' m','').replace('/','_').strip()}",
                         "relativna visina")

    print(f"\n========================================")
    if total == 0:
        print(f"SVE OK - sve vrijednosti iz master_dataset poklapaju se s referencom.")
    else:
        print(f"UKUPNO NEPODUDARANJA: {total}")
    print(f"========================================")


if __name__ == "__main__":
    main()
