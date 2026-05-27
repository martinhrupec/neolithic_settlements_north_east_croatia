"""
scenarij 2: random_ceste_biased (kontrolni uzorak s discovery bias-om) vs neoliticka naselja
============================================================================================

Cilj: provjeriti igra li 'discovery bias uz ceste' ulogu u nalazima iz scenarija 1.
Ako neolitik znacajno razlikuje od random_ceste, signal NIJE samo posljedica
toga da su nalazista otkrivena uz ceste — to su prave ekoloske razlike.

Sve usporedbe su 2-uzorkovne (oba uzorka iz mastera):
  - kontinuirane: 2-uzorkovni KS + VDA
  - kategorijske: 2-uzorkovni chi-square + Cramer's V
  - ordinalna (strahler): chi-square na kontingencijskoj tablici + Cliff's delta
  - aspect (stupnjevi): PRESKACEMO (cirkularna)

Bonferroni se primjenjuje unutar ovog scenarija (÷ ukupnim brojem testova).
Output: rezultati.csv.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats


ROOT    = r"c:\Users\Martin\Desktop\skripte_za_diplomski\statisticki_testovi"
MASTER  = os.path.join(ROOT, "master_dataset.csv")
OUT_DIR = os.path.join(ROOT, "02_random_ceste_bias_vs_neolitik")
OUT_CSV = os.path.join(OUT_DIR, "rezultati.csv")


# ---------------------------------------------------------------------------
#  Effect sizes
# ---------------------------------------------------------------------------

def vda(x, y):
    """A = P(X > Y) + 0.5 * P(X = Y)."""
    x = np.asarray(x); y = np.asarray(y)
    n, m = len(x), len(y)
    ranks = stats.rankdata(np.concatenate([x, y]))
    r_x = ranks[:n].sum()
    return float((r_x / n - (n + 1) / 2) / m)


def cliffs_delta(x, y):
    return 2.0 * vda(x, y) - 1.0


def cramers_v_contingency(chi2, n, dof_min):
    return float(np.sqrt(chi2 / (n * max(dof_min, 1))))


# ---------------------------------------------------------------------------
#  Interpretacije
# ---------------------------------------------------------------------------

def interp_vda(a):
    d = abs(a - 0.5)
    if d < 0.06: return "zanemariv"
    if d < 0.14: return "mali"
    if d < 0.21: return "srednji"
    return "velik"


def interp_cliffs(d):
    a = abs(d)
    if a < 0.147: return "zanemariv"
    if a < 0.33:  return "mali"
    if a < 0.474: return "srednji"
    return "velik"


def interp_cramers(v):
    if v < 0.10: return "zanemariv"
    if v < 0.30: return "mali"
    if v < 0.50: return "srednji"
    return "velik"


def smjer_vda(a):
    if a > 0.5: return "vise u neolitiku"
    if a < 0.5: return "manje u neolitiku"
    return "isto"


def smjer_cliff(d):
    if d > 0: return "visi red u neolitiku"
    if d < 0: return "nizi red u neolitiku"
    return "isto"


# ---------------------------------------------------------------------------
#  Test wrappers
# ---------------------------------------------------------------------------

def test_continuous(name, neo_vals, ctrl_vals):
    a = np.asarray(pd.Series(neo_vals).dropna(), dtype=float)
    b = np.asarray(pd.Series(ctrl_vals).dropna(), dtype=float)
    stat, p = stats.ks_2samp(a, b)
    A       = vda(a, b)
    return {
        "varijabla":   name,
        "tip":         "kontinuirana",
        "test":        "KS_2samp",
        "n_neolitik": len(a),
        "n_kontrola": len(b),
        "statistika": float(stat),
        "p_value":    float(p),
        "effect_name":   "VDA",
        "effect_value":  A,
        "effect_interp": interp_vda(A),
        "smjer":         smjer_vda(A),
    }


def test_categorical_2samp(name, neo_vals, ctrl_vals):
    """2-uzorkovni chi-square na kontingencijskoj tablici."""
    neo = pd.Series(neo_vals).dropna()
    ctr = pd.Series(ctrl_vals).dropna()
    # ujednacene kategorije
    cats = sorted(set(neo.unique()) | set(ctr.unique()))
    table = np.array([
        [int((neo == c).sum()) for c in cats],
        [int((ctr == c).sum()) for c in cats],
    ])
    # ako bilo koji stupac ima sve nule, izbacujemo ga (inace chi2 puca)
    keep = table.sum(axis=0) > 0
    table = table[:, keep]
    n = table.sum()
    chi2, p, _, _ = stats.chi2_contingency(table)
    V = cramers_v_contingency(chi2, n, min(table.shape) - 1)
    return {
        "varijabla":   name,
        "tip":         "kategorijska",
        "test":        "chi2_2samp",
        "n_neolitik": int(table[0].sum()),
        "n_kontrola": int(table[1].sum()),
        "statistika": float(chi2),
        "p_value":    float(p),
        "effect_name":   "CramersV",
        "effect_value":  V,
        "effect_interp": interp_cramers(V),
        "smjer":         "",
    }


def test_strahler(neo_vals, ctrl_vals):
    neo = pd.Series(neo_vals).dropna().astype(int)
    ctr = pd.Series(ctrl_vals).dropna().astype(int)
    cats = sorted(set(neo.unique()) | set(ctr.unique()))
    table = np.array([
        [int((neo == c).sum()) for c in cats],
        [int((ctr == c).sum()) for c in cats],
    ])
    keep = table.sum(axis=0) > 0
    table = table[:, keep]
    chi2, p, _, _ = stats.chi2_contingency(table)
    d = cliffs_delta(neo.values, ctr.values)
    return {
        "varijabla":   "strahler",
        "tip":         "ordinalna",
        "test":        "chi2_2samp",
        "n_neolitik": int(table[0].sum()),
        "n_kontrola": int(table[1].sum()),
        "statistika": float(chi2),
        "p_value":    float(p),
        "effect_name":   "CliffsDelta",
        "effect_value":  float(d),
        "effect_interp": interp_cliffs(d),
        "smjer":         smjer_cliff(d),
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df  = pd.read_csv(MASTER)
    neo = df[df.tip_sloja == "neolitik"]
    ctr = df[df.tip_sloja == "nasumicni_ceste"]

    print(f"Neolitik:        n = {len(neo)}")
    print(f"Random_ceste:    n = {len(ctr)}\n")

    results = []

    # 1) aps_vis
    results.append(test_continuous("aps_vis", neo["aps_vis"], ctr["aps_vis"]))

    # 2-7) rel_vis
    for combo in ["100_250", "100_500", "100_1000", "200_500", "200_1000", "500_1000"]:
        col = f"rel_vis_{combo}"
        results.append(test_continuous(col, neo[col], ctr[col]))

    # 8) aspect (stupnjevi) - SKIPPED (cirkularna)

    # 9-11) aspect derivati
    results.append(test_categorical_2samp("aspect_cat4", neo["aspect_cat4"], ctr["aspect_cat4"]))
    results.append(test_categorical_2samp("aspect_ew",   neo["aspect_ew"],   ctr["aspect_ew"]))
    results.append(test_categorical_2samp("aspect_sn",   neo["aspect_sn"],   ctr["aspect_sn"]))

    # 12) nagib
    results.append(test_continuous("nagib", neo["nagib"], ctr["nagib"]))

    # 13) coarse_fragments
    results.append(test_continuous("coarse_fragments", neo["coarse_fragments"], ctr["coarse_fragments"]))

    # 14a-d) vtt_rN
    for r in [100, 250, 500, 1000]:
        col = f"vtt_r{r}"
        results.append(test_categorical_2samp(col, neo[col], ctr[col]))

    # 15a-d) sm_rN
    for r in [100, 250, 500, 1000]:
        col = f"sm_r{r}"
        results.append(test_categorical_2samp(col, neo[col], ctr[col]))

    # 17-18) dist_rijeka, dist_rijeka_korig
    results.append(test_continuous("dist_rijeka",       neo["dist_rijeka"],       ctr["dist_rijeka"]))
    results.append(test_continuous("dist_rijeka_korig", neo["dist_rijeka_korig"], ctr["dist_rijeka_korig"]))

    # 19-20) gustoca_rijeka
    results.append(test_continuous("gustoca_rijeka_1000", neo["gustoca_rijeka_1000"], ctr["gustoca_rijeka_1000"]))
    results.append(test_continuous("gustoca_rijeka_2000", neo["gustoca_rijeka_2000"], ctr["gustoca_rijeka_2000"]))

    # 21) strahler
    results.append(test_strahler(neo["strahler"], ctr["strahler"]))

    # 22) tri
    results.append(test_continuous("tri", neo["tri"], ctr["tri"]))

    # ----- finaliziraj -----
    out      = pd.DataFrame(results)
    n_tests  = len(out)
    out["p_bonferroni"]        = (out["p_value"] * n_tests).clip(upper=1.0)
    out["znacajnost_005"]      = out["p_value"]      < 0.05
    out["znacajnost_005_bonf"] = out["p_bonferroni"] < 0.05

    col_order = [
        "varijabla", "tip", "test", "n_neolitik", "n_kontrola",
        "statistika", "p_value", "p_bonferroni",
        "znacajnost_005", "znacajnost_005_bonf",
        "effect_name", "effect_value", "effect_interp", "smjer",
    ]
    out = out[col_order]
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print(f"GOTOVO. {len(out)} testova  ->  {OUT_CSV}")
    print(f"Bonferroni divisor: {n_tests}\n")
    print(f"Znacajno na p<0.05 (raw):        {int(out['znacajnost_005'].sum())} / {n_tests}")
    print(f"Znacajno na p<0.05 (Bonferroni): {int(out['znacajnost_005_bonf'].sum())} / {n_tests}")
    print("\nSAZETAK:")
    with pd.option_context("display.max_rows", None,
                           "display.width", 200,
                           "display.float_format", "{:.4g}".format):
        print(out[["varijabla", "test", "p_value", "p_bonferroni",
                   "effect_value", "effect_interp", "smjer"]].to_string(index=False))


if __name__ == "__main__":
    main()
