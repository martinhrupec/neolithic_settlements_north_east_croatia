"""
scenarij 1: background distribucija (cijela Slavonija) vs neoliticka naselja
============================================================================

Za svaku varijablu iz master_dataset:
  - kontinuirane s pixel-hist background:   1-uzorkovni weighted KS + weighted VDA
  - kontinuirane bez pixel-hist:            2-uzorkovni KS + VDA (vs potpuno_nasumicni)
  - kategorijske (vtt, sm, aspect_cat4/ew/sn): chi-square goodness-of-fit + Cramer's V
  - ordinalna (strahler):                    chi-square GoF + Cliff's delta
  - aspect (stupnjevi):                      PRESKACEMO (cirkularna varijabla)

Bonferroni se primjenjuje unutar ovog scenarija (dijeli se s ukupnim brojem testova).
Output: rezultati.csv.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats


ROOT    = r"c:\Users\Martin\Desktop\skripte_za_diplomski\statisticki_testovi"
MASTER  = os.path.join(ROOT, "master_dataset.csv")
BG_DIR  = os.path.join(ROOT, "background")
OUT_DIR = os.path.join(ROOT, "01_background_vs_neolitik")
OUT_CSV = os.path.join(OUT_DIR, "rezultati.csv")


# ---------------------------------------------------------------------------
#  Weighted ECDF + KS + VDA (za 1-uzorkovne testove protiv pixel-histograma)
# ---------------------------------------------------------------------------

def weighted_ecdf(values, weights):
    idx = np.argsort(values)
    v   = np.asarray(values, dtype=float)[idx]
    w   = np.asarray(weights, dtype=float)[idx]
    cw  = np.cumsum(w) / w.sum()
    return v, cw


def cdf_from_hist(values, weights):
    v, cw = weighted_ecdf(values, weights)
    def F(x):
        x   = np.atleast_1d(np.asarray(x, dtype=float))
        idx = np.searchsorted(v, x, side="right") - 1
        return np.where(idx < 0, 0.0, cw[np.clip(idx, 0, len(cw) - 1)])
    return F


def ks_1samp_weighted(sample, bg_v, bg_w):
    F = cdf_from_hist(bg_v, bg_w)
    res = stats.ks_1samp(sample, F)
    return float(res.statistic), float(res.pvalue)


def vda_weighted(sample, bg_v, bg_w):
    """A = P(X > Y) + 0.5 * P(X = Y),  X = neolitik,  Y = pixel s tezinom."""
    v, cw  = weighted_ecdf(bg_v, bg_w)
    sample = np.asarray(sample, dtype=float)
    idx_le = np.searchsorted(v, sample, side="right") - 1
    idx_lt = np.searchsorted(v, sample, side="left")  - 1
    P_le   = np.where(idx_le < 0, 0.0, cw[np.clip(idx_le, 0, len(cw) - 1)])
    P_lt   = np.where(idx_lt < 0, 0.0, cw[np.clip(idx_lt, 0, len(cw) - 1)])
    # P(Y < x) + 0.5 * P(Y = x)  =  0.5 * (P_lt + P_le)
    return float(np.mean(0.5 * (P_lt + P_le)))


# ---------------------------------------------------------------------------
#  Klasicni 2-uzorkovni testovi (VDA + Cliff)
# ---------------------------------------------------------------------------

def vda(x, y):
    x = np.asarray(x); y = np.asarray(y)
    n, m = len(x), len(y)
    ranks = stats.rankdata(np.concatenate([x, y]))
    r_x = ranks[:n].sum()
    return float((r_x / n - (n + 1) / 2) / m)


def cliffs_delta(x, y):
    return 2.0 * vda(x, y) - 1.0


def cramers_v(chi2, n, df_min):
    return float(np.sqrt(chi2 / (n * max(df_min, 1))))


# ---------------------------------------------------------------------------
#  Interpretacije effect-size-a
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
#  Aspect binning -> background proporcije za cat4 / ew / sn
# ---------------------------------------------------------------------------

def bin_aspect_cat4(deg):
    d = deg % 360
    if d < 90:  return "NE"
    if d < 180: return "SE"
    if d < 270: return "SW"
    return "NW"


def bin_aspect_ew(deg):
    d = deg % 360
    return "E" if d < 180 else "W"


def bin_aspect_sn(deg):
    d = deg % 360
    return "S" if 90 <= d < 270 else "N"


def aspect_bg_proportions(binner, categories):
    bg = pd.read_csv(os.path.join(BG_DIR, "background_aspect.csv"), encoding="utf-8-sig")
    bg["bin"] = bg["aspect_value"].apply(binner)
    grouped   = bg.groupby("bin")["n_piksela"].sum()
    total     = grouped.sum()
    return {c: float(grouped.get(c, 0)) / total for c in categories}


# ---------------------------------------------------------------------------
#  Test wrapper funkcije
# ---------------------------------------------------------------------------

def test_continuous_weighted(name, neolitik_vals, bg_csv, bg_value_col, bg_weight_col):
    sample = np.asarray(pd.Series(neolitik_vals).dropna(), dtype=float)
    bg     = pd.read_csv(os.path.join(BG_DIR, bg_csv), encoding="utf-8-sig")
    bg_v   = bg[bg_value_col].astype(float).values
    bg_w   = bg[bg_weight_col].astype(float).values
    stat, p = ks_1samp_weighted(sample, bg_v, bg_w)
    A       = vda_weighted(sample, bg_v, bg_w)
    return {
        "varijabla":        name,
        "tip":              "kontinuirana",
        "test":             "KS_1samp_weighted",
        "n_neolitik":       len(sample),
        "background_source": bg_csv,
        "statistika":       stat,
        "p_value":          p,
        "effect_name":      "VDA_weighted",
        "effect_value":     A,
        "effect_interp":    interp_vda(A),
        "smjer":            smjer_vda(A),
    }


def test_continuous_fallback(name, neolitik_vals, fallback_vals):
    a = np.asarray(pd.Series(neolitik_vals).dropna(), dtype=float)
    b = np.asarray(pd.Series(fallback_vals).dropna(), dtype=float)
    stat, p = stats.ks_2samp(a, b)
    A = vda(a, b)
    return {
        "varijabla":        name,
        "tip":              "kontinuirana",
        "test":             "KS_2samp",
        "n_neolitik":       len(a),
        "background_source": f"potpuno_nasumicni_fallback (n={len(b)})",
        "statistika":       float(stat),
        "p_value":          float(p),
        "effect_name":      "VDA",
        "effect_value":     A,
        "effect_interp":    interp_vda(A),
        "smjer":            smjer_vda(A),
    }


def test_categorical(name, neolitik_vals, bg_props, source_label):
    obs_counts = pd.Series(neolitik_vals).value_counts()
    cats       = [c for c in bg_props if bg_props[c] > 0]
    observed   = np.array([obs_counts.get(c, 0) for c in cats], dtype=float)
    n          = observed.sum()
    expected   = np.array([bg_props[c] * n for c in cats], dtype=float)
    chi2, p    = stats.chisquare(f_obs=observed, f_exp=expected)
    V          = cramers_v(chi2, n, len(cats) - 1)
    return {
        "varijabla":        name,
        "tip":              "kategorijska",
        "test":             "chi2_GoF",
        "n_neolitik":       int(n),
        "background_source": source_label,
        "statistika":       float(chi2),
        "p_value":          float(p),
        "effect_name":      "CramersV",
        "effect_value":     V,
        "effect_interp":    interp_cramers(V),
        "smjer":            "",
    }


def test_strahler(neolitik_vals, bg_strahler):
    obs_counts = pd.Series(neolitik_vals).dropna().astype(int).value_counts()
    cats       = sorted(bg_strahler["strahler"].astype(int).unique())
    observed   = np.array([obs_counts.get(c, 0) for c in cats], dtype=float)
    n          = observed.sum()
    total_km   = bg_strahler["duljina_km"].sum()
    expected   = np.array([
        bg_strahler[bg_strahler["strahler"] == c]["duljina_km"].sum() / total_km * n
        for c in cats
    ], dtype=float)
    # Sklonimo eventualne nule u expected (nije problem ovdje, ali safety)
    mask = expected > 0
    chi2, p = stats.chisquare(f_obs=observed[mask], f_exp=expected[mask])
    # Cliff's delta: rekonstruiramo bg sample proporcionalno duljini (10 000 tocaka)
    weights = (bg_strahler["duljina_km"].values / total_km * 10000).round().astype(int)
    bg_sample = np.repeat(bg_strahler["strahler"].astype(int).values, weights)
    d = cliffs_delta(np.asarray(pd.Series(neolitik_vals).dropna().astype(int)), bg_sample)
    return {
        "varijabla":        "strahler",
        "tip":              "ordinalna",
        "test":             "chi2_GoF",
        "n_neolitik":       int(n),
        "background_source": "background_strahler.csv",
        "statistika":       float(chi2),
        "p_value":          float(p),
        "effect_name":      "CliffsDelta",
        "effect_value":     float(d),
        "effect_interp":    interp_cliffs(d),
        "smjer":            smjer_cliff(d),
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df          = pd.read_csv(MASTER)
    neo         = df[df.tip_sloja == "neolitik"]
    bg_fallback = df[df.tip_sloja == "potpuno_nasumicni"]

    results = []

    # 1) aps_vis
    results.append(test_continuous_weighted(
        "aps_vis", neo["aps_vis"], "background_elev.csv", "elev_m", "n_piksela"))

    # 2-7) rel_vis (fallback - nema background fajla)
    for combo in ["100_250", "100_500", "100_1000", "200_500", "200_1000", "500_1000"]:
        col = f"rel_vis_{combo}"
        results.append(test_continuous_fallback(col, neo[col], bg_fallback[col]))

    # 8) aspect (stupnjevi) - SKIPPED (cirkularna)

    # 9-11) aspect_cat4 / ew / sn
    results.append(test_categorical("aspect_cat4", neo["aspect_cat4"].dropna(),
                                    aspect_bg_proportions(bin_aspect_cat4, ["NE", "SE", "SW", "NW"]),
                                    "background_aspect.csv (rebinned)"))
    results.append(test_categorical("aspect_ew",   neo["aspect_ew"].dropna(),
                                    aspect_bg_proportions(bin_aspect_ew, ["E", "W"]),
                                    "background_aspect.csv (rebinned)"))
    results.append(test_categorical("aspect_sn",   neo["aspect_sn"].dropna(),
                                    aspect_bg_proportions(bin_aspect_sn, ["N", "S"]),
                                    "background_aspect.csv (rebinned)"))

    # 12) nagib (fallback)
    results.append(test_continuous_fallback("nagib", neo["nagib"], bg_fallback["nagib"]))

    # 13) coarse_fragments
    results.append(test_continuous_weighted(
        "coarse_fragments", neo["coarse_fragments"],
        "background_cfrag.csv", "value_volpct", "n_piksela"))

    # 14a-d) vtt_rN
    bg_vtt    = pd.read_csv(os.path.join(BG_DIR, "background_vtt.csv"), encoding="utf-8-sig")
    vtt_props = dict(zip(bg_vtt["tip_tla"], bg_vtt["n_piksela"] / bg_vtt["n_piksela"].sum()))
    for r in [100, 250, 500, 1000]:
        col  = f"vtt_r{r}"
        vals = neo[col].dropna()
        vals = vals[vals.isin(vtt_props.keys())]   # samo poznate kategorije
        results.append(test_categorical(col, vals, vtt_props, "background_vtt.csv"))

    # 15a-d) sm_rN
    bg_sm    = pd.read_csv(os.path.join(BG_DIR, "background_sm.csv"), encoding="utf-8-sig")
    sm_props = dict(zip(bg_sm["kategorija"], bg_sm["n_piksela"] / bg_sm["n_piksela"].sum()))
    for r in [100, 250, 500, 1000]:
        col = f"sm_r{r}"
        results.append(test_categorical(col, neo[col].dropna(), sm_props, "background_sm.csv"))

    # 17) dist_rijeka (fallback)
    results.append(test_continuous_fallback("dist_rijeka", neo["dist_rijeka"], bg_fallback["dist_rijeka"]))

    # 18) dist_rijeka_korig (fallback)
    results.append(test_continuous_fallback("dist_rijeka_korig",
                                            neo["dist_rijeka_korig"],
                                            bg_fallback["dist_rijeka_korig"]))

    # 19-20) gustoca (fallback - bg fajl je samo agregat)
    results.append(test_continuous_fallback("gustoca_rijeka_1000",
                                            neo["gustoca_rijeka_1000"],
                                            bg_fallback["gustoca_rijeka_1000"]))
    results.append(test_continuous_fallback("gustoca_rijeka_2000",
                                            neo["gustoca_rijeka_2000"],
                                            bg_fallback["gustoca_rijeka_2000"]))

    # 21) strahler
    bg_strahler = pd.read_csv(os.path.join(BG_DIR, "background_strahler.csv"), encoding="utf-8-sig")
    results.append(test_strahler(neo["strahler"], bg_strahler))

    # 22) tri
    results.append(test_continuous_weighted(
        "tri", neo["tri"], "background_tri.csv", "tri_value", "n_piksela"))

    # ----- finaliziraj -----
    out      = pd.DataFrame(results)
    n_tests  = len(out)
    out["p_bonferroni"]        = (out["p_value"] * n_tests).clip(upper=1.0)
    out["znacajnost_005"]      = out["p_value"]      < 0.05
    out["znacajnost_005_bonf"] = out["p_bonferroni"] < 0.05

    col_order = [
        "varijabla", "tip", "test", "n_neolitik", "background_source",
        "statistika", "p_value", "p_bonferroni",
        "znacajnost_005", "znacajnost_005_bonf",
        "effect_name", "effect_value", "effect_interp", "smjer",
    ]
    out = out[col_order]
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print(f"GOTOVO. {len(out)} testova  ->  {OUT_CSV}")
    print(f"Bonferroni divisor: {n_tests}")
    print(f"\nZnacajno na p<0.05 (raw):        {int(out['znacajnost_005'].sum())} / {n_tests}")
    print(f"Znacajno na p<0.05 (Bonferroni): {int(out['znacajnost_005_bonf'].sum())} / {n_tests}")
    print("\nSAZETAK:")
    with pd.option_context("display.max_rows", None,
                           "display.width", 200,
                           "display.float_format", "{:.4g}".format):
        print(out[["varijabla", "test", "p_value", "p_bonferroni",
                   "effect_value", "effect_interp", "smjer"]].to_string(index=False))


if __name__ == "__main__":
    main()
