"""
Provjera Drennanovih uvjeta za 2-uzorkovni chi-square (scenariji 02, 04, 06).
============================================================================

Za razliku od scenarija 1 (goodness-of-fit, 1 uzorak), scenariji 02/04/06
koriste kontingencijske tablice 2 x k (dvije skupine x k kategorija) i
scipy.stats.chi2_contingency. Ocekivane frekvencije racunaju se iz rubnih
zbrojeva (red_total * stupac_total / n), pa se Drennanovi (1996: 197-198)
uvjeti provjeravaju na SVIH 2k celija:
  (a) nijedna ocekivana frekvencija < 1
  (b) najvise 20 % celija s ocekivanom frekvencijom < 5

Provjeravaju se dvije verzije:
  - SIROVO:  bez ikakve prilagodbe (sve izvorne kategorije)
  - FINALNO: kako scenariji SADA stvarno racunaju, tj. s prilagodbama
             primijenjenima u run.py:
               * scenarij 02: vtt -> 4 dominantna tla + 'rijetka_tla'
               * scenarij 04: strahler -> redovi >= 4 spojeni u jednu klasu
               * ostalo / ostali scenariji: bez prilagodbe

Output: provjera_drennan_kontingencija.csv + ispis u konzolu.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT    = r"c:\Users\Martin\Desktop\skripte_za_diplomski\statisticki_testovi"
MASTER  = os.path.join(ROOT, "master_dataset.csv")
BG_DIR  = os.path.join(ROOT, "background")
OUT_CSV = os.path.join(ROOT, "provjera_drennan_kontingencija.csv")

KAT_VARS = (["aspect_cat4", "aspect_ew", "aspect_sn"]
            + [f"vtt_r{r}" for r in [100, 250, 500, 1000]]
            + [f"sm_r{r}"  for r in [100, 250, 500, 1000]]
            + ["strahler"])

STRAHLER_CAP = 4  # isto kao u scenariju 04


def vtt_glavna_kategorije(prag=0.01):
    bg    = pd.read_csv(os.path.join(BG_DIR, "background_vtt.csv"), encoding="utf-8-sig")
    share = bg.set_index("tip_tla")["n_piksela"] / bg["n_piksela"].sum()
    return set(share[share >= prag].index)


def transform(var, a, b, rules, vtt_glavna):
    """Primijeni prilagodbe koje run.py stvarno koristi za taj scenarij."""
    a = pd.Series(a).dropna()
    b = pd.Series(b).dropna()
    if var.startswith("vtt_") and rules.get("group_vtt"):
        remap = lambda c: c if c in vtt_glavna else "rijetka_tla"
        a, b = a.map(remap), b.map(remap)
    if var == "strahler" and rules.get("cap_strahler"):
        cap = rules["cap_strahler"]
        a, b = a.astype(int).clip(upper=cap), b.astype(int).clip(upper=cap)
    return a, b


def build_table(a, b):
    a = pd.Series(a).dropna()
    b = pd.Series(b).dropna()
    cats = sorted(set(a.unique()) | set(b.unique()), key=lambda x: str(x))
    table = np.array([
        [int((a == c).sum()) for c in cats],
        [int((b == c).sum()) for c in cats],
    ], dtype=float)
    keep = table.sum(axis=0) > 0
    return table[:, keep]


def drennan_check(table):
    table = np.asarray(table, dtype=float)
    if table.shape[1] < 2:
        return None
    chi2, p, _, exp = stats.chi2_contingency(table)
    flat = exp.ravel()
    n_lt5 = int((flat < 5).sum())
    pct   = 100.0 * n_lt5 / flat.size
    return {
        "k":       table.shape[1],
        "p_value": float(p),
        "min_exp": float(flat.min()),
        "lt5":     n_lt5,
        "pct_lt5": pct,
        "a_ok":    int((flat < 1).sum()) == 0,
        "b_ok":    pct <= 20.0,
    }


def opis_transform(var, rules):
    if var.startswith("vtt_") and rules.get("group_vtt"):
        return "vtt: 4 glavna + rijetka_tla"
    if var == "strahler" and rules.get("cap_strahler"):
        return f"strahler: redovi >= {rules['cap_strahler']} spojeni"
    return "-"


def run_scenario(label, dfA, dfB, rules, vtt_glavna):
    rows = []
    for var in KAT_VARS:
        sir = drennan_check(build_table(dfA[var], dfB[var]))
        a_t, b_t = transform(var, dfA[var], dfB[var], rules, vtt_glavna)
        fin = drennan_check(build_table(a_t, b_t))
        if sir is None or fin is None:
            continue
        rows.append({
            "scenarij":      label,
            "varijabla":     var,
            "transformacija": opis_transform(var, rules),
            "p_sirovo":      round(sir["p_value"], 6),
            "k_sir":         sir["k"],
            "min_exp_sir":   round(sir["min_exp"], 3),
            "pct_lt5_sir":   round(sir["pct_lt5"], 1),
            "a_ok_sir":      sir["a_ok"],
            "b_ok_sir":      sir["b_ok"],
            "p_finalno":     round(fin["p_value"], 6),
            "k_fin":         fin["k"],
            "min_exp_fin":   round(fin["min_exp"], 3),
            "pct_lt5_fin":   round(fin["pct_lt5"], 1),
            "a_ok_fin":      fin["a_ok"],
            "b_ok_fin":      fin["b_ok"],
        })
    return rows


def main():
    df  = pd.read_csv(MASTER)
    neo = df[df.tip_sloja == "neolitik"]
    vtt_glavna = vtt_glavna_kategorije()

    scenariji = [
        ("02_random_ceste", neo, df[df.tip_sloja == "nasumicni_ceste"],
         {"group_vtt": True}),
        ("04_rano_vs_kasno", neo[neo.samo_rano == True], neo[neo.samo_kasno == True],
         {"cap_strahler": STRAHLER_CAP}),
        ("06_jedno_vs_kont", neo[(neo.samo_rano == True) | (neo.samo_kasno == True)],
         neo[neo.kontinuirano == True], {}),
    ]

    all_rows = []
    for label, dfA, dfB, rules in scenariji:
        all_rows.extend(run_scenario(label, dfA, dfB, rules, vtt_glavna))

    out = pd.DataFrame(all_rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", None)

    print("=" * 100)
    print("DRENNANOVI UVJETI ZA 2-UZORKOVNI CHI-SQUARE (kontingencija 2 x k)")
    print("  (a) nijedna ocekivana < 1     (b) najvise 20% celija s ocekivanom < 5     znacajno = p < 0.05")
    print("  SIROVO = bez prilagodbe   |   FINALNO = kako scenarij sada racuna (run.py)")
    print("=" * 100)

    for label, _, _, _ in scenariji:
        sub = out[out["scenarij"] == label]
        print(f"\n### {label}")
        print("SIROVO:")
        print(sub[["varijabla", "p_sirovo", "k_sir", "min_exp_sir",
                   "pct_lt5_sir", "a_ok_sir", "b_ok_sir"]].to_string(index=False))
        print("FINALNO (run.py):")
        print(sub[["varijabla", "transformacija", "p_finalno", "k_fin", "min_exp_fin",
                   "pct_lt5_fin", "a_ok_fin", "b_ok_fin"]].to_string(index=False))
        # znacajni nalazi koji JOS uvijek krse u finalnom stanju
        loši = sub[(sub["p_finalno"] < 0.05) & ~(sub["a_ok_fin"] & sub["b_ok_fin"])]
        if not loši.empty:
            print(f"  PAZNJA - znacajni A krse Drennana i u finalnom: {', '.join(loši['varijabla'])}")

    print("\n" + "-" * 100)
    print("SAZETAK (samo ZNACAJNI nalazi, p_finalno < 0.05):")
    for label, _, _, _ in scenariji:
        sub = out[out["scenarij"] == label]
        sig = sub[sub["p_finalno"] < 0.05]
        if sig.empty:
            print(f"  {label}: nema znacajnih kategorijskih nalaza")
            continue
        ok = bool((sig["a_ok_fin"] & sig["b_ok_fin"]).all())
        vlist = ", ".join(sig["varijabla"])
        print(f"  {label}: znacajni [{vlist}] -> Drennan {'OK' if ok else 'PADA'}")
    print("-" * 100)
    print(f"\nCSV spremljen: {OUT_CSV}")


if __name__ == "__main__":
    main()
