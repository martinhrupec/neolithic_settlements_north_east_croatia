"""
Spatial thinning + ponovljeni S3 i S4
======================================

Cilj: ukloniti prostornu pseudoreplikaciju iz svake faze, pa ponoviti
usporedbe.

Metoda: GREEDY DISTANCE-BASED THINNING.
  1. Shuffle redoslijed tocaka (seed=42)
  2. Iteriraj kroz redoslijed:
       - ako tocka NIJE unutar threshold m od neke vec zadrzane → zadrzi
       - ako jest → odbaci
  3. Vraca zadrzane indekse.

Threshold izbor: pokazujemo statistiku za vise vrijednosti (0.5 / 2 / 3 / 5 km).
Primarna analiza koristi 3 km — to je tipicna velicina kataloga
naselja u arheologiji srednjeg neolitika i otprilike velicina koja
razbija unutar-grupne grozdove bez decimacije uzorka.

Ponavljamo:
  S3:  samo_rano (thinned) vs samo_kasno (thinned)
  S4:  jednofazna (thinned) vs kontinuirana (thinned)

Output:
  - rezultati_thin_s3.csv
  - rezultati_thin_s4.csv
  - usporedba_thin_vs_full.csv (pre/post sazetak)
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
#  Konfiguracija
# ---------------------------------------------------------------------------

ROOT       = r"c:\Users\Martin\Desktop\skripte_za_diplomski\statisticki_testovi"
MASTER     = os.path.join(ROOT, "master_dataset.csv")
COORDS_CSV = os.path.join(ROOT, "01_prostorna_autokorelacija", "neolitik_coords.csv")
OUT_DIR    = os.path.join(ROOT, "07_thinned_analize")

PRIMARY_THRESHOLD_M = 500    # 500 m — primarni threshold za testove
THRESHOLDS_M        = [500, 1000, 2000, 3000, 5000]
SEED                = 42


# ---------------------------------------------------------------------------
#  Greedy thinning
# ---------------------------------------------------------------------------

def greedy_thin(coords, threshold_m, seed=42):
    """
    Vraca listu indeksa zadrzanih tocaka.
    Ulazni coords je np.array (n, 2).
    """
    n = len(coords)
    if n == 0:
        return []
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)

    excluded = np.zeros(n, dtype=bool)
    tree     = cKDTree(coords)
    kept     = []

    for i in order:
        if excluded[i]:
            continue
        kept.append(i)
        for j in tree.query_ball_point(coords[i], threshold_m):
            if j != i:
                excluded[j] = True
    return sorted(kept)


# ---------------------------------------------------------------------------
#  Effect sizes (kopirano iz S3/S4)
# ---------------------------------------------------------------------------

def vda(x, y):
    x = np.asarray(x); y = np.asarray(y)
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return float("nan")
    ranks = stats.rankdata(np.concatenate([x, y]))
    return float((ranks[:n].sum() / n - (n + 1) / 2) / m)


def cliffs_delta(x, y):
    a = vda(x, y)
    return 2.0 * a - 1.0 if not np.isnan(a) else float("nan")


def cramers_v_contingency(chi2, n, dof_min):
    return float(np.sqrt(chi2 / (n * max(dof_min, 1))))


def interp_vda(a):
    if np.isnan(a): return "—"
    d = abs(a - 0.5)
    if d < 0.06: return "zanemariv"
    if d < 0.14: return "mali"
    if d < 0.21: return "srednji"
    return "velik"


def interp_cliffs(d):
    if np.isnan(d): return "—"
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


# ---------------------------------------------------------------------------
#  Test wrappers — labeli prilagodljivi (rano/kasno ili jednofazna/kontinuirana)
# ---------------------------------------------------------------------------

def test_continuous(name, A_vals, B_vals, A_label, B_label):
    a = np.asarray(pd.Series(A_vals).dropna(), dtype=float)
    b = np.asarray(pd.Series(B_vals).dropna(), dtype=float)
    if len(a) < 3 or len(b) < 3:
        return {"varijabla": name, "tip": "kontinuirana", "test": "KS_2samp",
                f"n_{A_label}": len(a), f"n_{B_label}": len(b),
                "statistika": np.nan, "p_value": np.nan,
                "effect_name": "VDA", "effect_value": np.nan,
                "effect_interp": "—", "smjer": "premalo podataka"}
    stat, p = stats.ks_2samp(a, b)
    A = vda(a, b)
    smjer = ("vise u " + A_label) if A > 0.5 else \
            ("vise u " + B_label) if A < 0.5 else "isto"
    return {
        "varijabla":   name, "tip": "kontinuirana", "test": "KS_2samp",
        f"n_{A_label}": len(a), f"n_{B_label}": len(b),
        "statistika": float(stat), "p_value": float(p),
        "effect_name": "VDA", "effect_value": A,
        "effect_interp": interp_vda(A), "smjer": smjer,
    }


def test_categorical_2samp(name, A_vals, B_vals, A_label, B_label):
    A_ser = pd.Series(A_vals).dropna()
    B_ser = pd.Series(B_vals).dropna()
    cats  = sorted(set(A_ser.unique()) | set(B_ser.unique()))
    table = np.array([
        [int((A_ser == c).sum()) for c in cats],
        [int((B_ser == c).sum()) for c in cats],
    ])
    keep  = table.sum(axis=0) > 0
    table = table[:, keep]
    if table.shape[1] < 2 or table.sum() < 5:
        return {"varijabla": name, "tip": "kategorijska", "test": "chi2_2samp",
                f"n_{A_label}": int(table[0].sum()) if table.size else 0,
                f"n_{B_label}": int(table[1].sum()) if table.size else 0,
                "statistika": np.nan, "p_value": np.nan,
                "effect_name": "CramersV", "effect_value": np.nan,
                "effect_interp": "—", "smjer": "premalo podataka"}
    n = table.sum()
    chi2, p, _, _ = stats.chi2_contingency(table)
    V = cramers_v_contingency(chi2, n, min(table.shape) - 1)
    return {
        "varijabla": name, "tip": "kategorijska", "test": "chi2_2samp",
        f"n_{A_label}": int(table[0].sum()), f"n_{B_label}": int(table[1].sum()),
        "statistika": float(chi2), "p_value": float(p),
        "effect_name": "CramersV", "effect_value": V,
        "effect_interp": interp_cramers(V), "smjer": "",
    }


def test_strahler(A_vals, B_vals, A_label, B_label):
    A_ser = pd.Series(A_vals).dropna().astype(int)
    B_ser = pd.Series(B_vals).dropna().astype(int)
    cats  = sorted(set(A_ser.unique()) | set(B_ser.unique()))
    table = np.array([
        [int((A_ser == c).sum()) for c in cats],
        [int((B_ser == c).sum()) for c in cats],
    ])
    keep  = table.sum(axis=0) > 0
    table = table[:, keep]
    chi2, p, _, _ = stats.chi2_contingency(table)
    d = cliffs_delta(A_ser.values, B_ser.values)
    smjer = ("visi red u " + A_label) if d > 0 else \
            ("visi red u " + B_label) if d < 0 else "isto"
    return {
        "varijabla":  "strahler", "tip": "ordinalna", "test": "chi2_2samp",
        f"n_{A_label}": int(table[0].sum()), f"n_{B_label}": int(table[1].sum()),
        "statistika": float(chi2), "p_value": float(p),
        "effect_name": "CliffsDelta", "effect_value": float(d),
        "effect_interp": interp_cliffs(d), "smjer": smjer,
    }


# ---------------------------------------------------------------------------
#  Pune baterije testova za jedan par grupa
# ---------------------------------------------------------------------------

def run_battery(group_A, group_B, A_label, B_label):
    results = []
    results.append(test_continuous("aps_vis", group_A["aps_vis"], group_B["aps_vis"], A_label, B_label))
    for combo in ["100_250", "100_500", "100_1000", "200_500", "200_1000", "500_1000"]:
        col = f"rel_vis_{combo}"
        results.append(test_continuous(col, group_A[col], group_B[col], A_label, B_label))
    results.append(test_categorical_2samp("aspect_cat4", group_A["aspect_cat4"], group_B["aspect_cat4"], A_label, B_label))
    results.append(test_categorical_2samp("aspect_ew",   group_A["aspect_ew"],   group_B["aspect_ew"],   A_label, B_label))
    results.append(test_categorical_2samp("aspect_sn",   group_A["aspect_sn"],   group_B["aspect_sn"],   A_label, B_label))
    results.append(test_continuous("nagib",            group_A["nagib"],            group_B["nagib"],            A_label, B_label))
    results.append(test_continuous("coarse_fragments", group_A["coarse_fragments"], group_B["coarse_fragments"], A_label, B_label))
    for r in [100, 250, 500, 1000]:
        col = f"vtt_r{r}"
        results.append(test_categorical_2samp(col, group_A[col], group_B[col], A_label, B_label))
    for r in [100, 250, 500, 1000]:
        col = f"sm_r{r}"
        results.append(test_categorical_2samp(col, group_A[col], group_B[col], A_label, B_label))
    results.append(test_continuous("dist_rijeka",          group_A["dist_rijeka"],       group_B["dist_rijeka"],       A_label, B_label))
    results.append(test_continuous("dist_rijeka_korig",    group_A["dist_rijeka_korig"], group_B["dist_rijeka_korig"], A_label, B_label))
    results.append(test_continuous("gustoca_rijeka_1000",  group_A["gustoca_rijeka_1000"], group_B["gustoca_rijeka_1000"], A_label, B_label))
    results.append(test_continuous("gustoca_rijeka_2000",  group_A["gustoca_rijeka_2000"], group_B["gustoca_rijeka_2000"], A_label, B_label))
    results.append(test_strahler(group_A["strahler"], group_B["strahler"], A_label, B_label))
    results.append(test_continuous("tri", group_A["tri"], group_B["tri"], A_label, B_label))

    out = pd.DataFrame(results)
    n_tests = len(out)
    out["p_bonferroni"]        = (out["p_value"] * n_tests).clip(upper=1.0)
    out["znacajnost_005"]      = out["p_value"]      < 0.05
    out["znacajnost_005_bonf"] = out["p_bonferroni"] < 0.05
    return out


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df     = pd.read_csv(MASTER)
    coords = pd.read_csv(COORDS_CSV)
    neo    = df[df.tip_sloja == "neolitik"].copy()
    merged = neo.merge(coords, left_on="fid_raw", right_on="fid", how="inner")

    rano_full  = merged[merged.samo_rano    == True].copy()
    kasno_full = merged[merged.samo_kasno   == True].copy()
    kont_full  = merged[merged.kontinuirano == True].copy()
    jedno_full = pd.concat([rano_full, kasno_full], ignore_index=True)

    print(f"PRIJE THINNINGA:")
    print(f"  samo_rano:        n = {len(rano_full)}")
    print(f"  samo_kasno:       n = {len(kasno_full)}")
    print(f"  jednofazni:       n = {len(jedno_full)}")
    print(f"  kontinuirana:     n = {len(kont_full)}\n")

    # ---- 1) Statistika za vise thresholdova ----
    print("=" * 72)
    print("1) BROJ ZADRZANIH TOCAKA pri razlicitim thresholdovima")
    print("=" * 72)
    print(f"{'threshold':<12}{'rano':>8}{'kasno':>8}{'jednofazni':>14}{'kont':>8}")
    print("-" * 50)
    thresh_stats = []
    for thr in THRESHOLDS_M:
        n_r = len(greedy_thin(rano_full [["x","y"]].values, thr, SEED))
        n_k = len(greedy_thin(kasno_full[["x","y"]].values, thr, SEED))
        n_j = len(greedy_thin(jedno_full[["x","y"]].values, thr, SEED))
        n_c = len(greedy_thin(kont_full [["x","y"]].values, thr, SEED))
        print(f"{thr/1000:>6.1f} km   {n_r:>5d}   {n_k:>5d}   {n_j:>11d}   {n_c:>5d}")
        thresh_stats.append({"threshold_m": thr, "n_rano": n_r, "n_kasno": n_k,
                             "n_jednofazni": n_j, "n_kontinuirana": n_c})
    pd.DataFrame(thresh_stats).to_csv(os.path.join(OUT_DIR, "00_thinning_stats.csv"),
                                       index=False, encoding="utf-8")
    print()

    # ---- 2) Primjeni primarni threshold ----
    thr = PRIMARY_THRESHOLD_M
    print(f"Primarni threshold: {thr/1000:.1f} km\n")

    keep_r = greedy_thin(rano_full [["x","y"]].values, thr, SEED)
    keep_k = greedy_thin(kasno_full[["x","y"]].values, thr, SEED)
    keep_j = greedy_thin(jedno_full[["x","y"]].values, thr, SEED)
    keep_c = greedy_thin(kont_full [["x","y"]].values, thr, SEED)

    rano_thin  = rano_full .iloc[keep_r].reset_index(drop=True)
    kasno_thin = kasno_full.iloc[keep_k].reset_index(drop=True)
    jedno_thin = jedno_full.iloc[keep_j].reset_index(drop=True)
    kont_thin  = kont_full .iloc[keep_c].reset_index(drop=True)

    print("=" * 72)
    print(f"PRIMJENJEN THINNING ({thr/1000:.1f} km):")
    print(f"  samo_rano:        {len(rano_full):>3d}  ->  {len(rano_thin):>3d}  "
          f"({100*len(rano_thin)/len(rano_full):.0f}%)")
    print(f"  samo_kasno:       {len(kasno_full):>3d}  ->  {len(kasno_thin):>3d}  "
          f"({100*len(kasno_thin)/len(kasno_full):.0f}%)")
    print(f"  jednofazni:       {len(jedno_full):>3d}  ->  {len(jedno_thin):>3d}  "
          f"({100*len(jedno_thin)/len(jedno_full):.0f}%)")
    print(f"  kontinuirana:     {len(kont_full):>3d}  ->  {len(kont_thin):>3d}  "
          f"({100*len(kont_thin)/len(kont_full):.0f}%)")
    print("=" * 72)
    print()

    # ---- 3) S3 thinned: rano vs kasno ----
    print("=" * 72)
    print("3) S3 THINNED: samo_rano vs samo_kasno (3 km thinning)")
    print("=" * 72)
    s3_thin = run_battery(rano_thin, kasno_thin, "rano", "kasno")
    s3_thin.to_csv(os.path.join(OUT_DIR, "rezultati_thin_s3.csv"),
                   index=False, encoding="utf-8")
    print(f"Znacajno raw:   {int(s3_thin['znacajnost_005'].sum())}/{len(s3_thin)}")
    print(f"Znacajno Bonf:  {int(s3_thin['znacajnost_005_bonf'].sum())}/{len(s3_thin)}")
    print()
    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.float_format", "{:.4g}".format):
        print(s3_thin[["varijabla","p_value","p_bonferroni",
                       "effect_value","effect_interp","smjer"]].to_string(index=False))

    # ---- 4) S4 thinned: jednofazna vs kontinuirana ----
    print()
    print("=" * 72)
    print("4) S4 THINNED: jednofazni (rano+kasno) vs kontinuirana (3 km)")
    print("=" * 72)
    s4_thin = run_battery(jedno_thin, kont_thin, "jednofazna", "kontinuirana")
    s4_thin.to_csv(os.path.join(OUT_DIR, "rezultati_thin_s4.csv"),
                   index=False, encoding="utf-8")
    print(f"Znacajno raw:   {int(s4_thin['znacajnost_005'].sum())}/{len(s4_thin)}")
    print(f"Znacajno Bonf:  {int(s4_thin['znacajnost_005_bonf'].sum())}/{len(s4_thin)}")
    print()
    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.float_format", "{:.4g}".format):
        print(s4_thin[["varijabla","p_value","p_bonferroni",
                       "effect_value","effect_interp","smjer"]].to_string(index=False))

    # ---- 5) Usporedba pre/post thinninga (samo S4 koji je suspect) ----
    print()
    print("=" * 72)
    print("5) USPOREDBA PRE/POST THINNING — SCENARIJ 4 (suspect)")
    print("=" * 72)

    full_s4_path = os.path.join(ROOT, "06_jednofazni_vs_kontinuirani", "rezultati.csv")
    if os.path.exists(full_s4_path):
        full_s4 = pd.read_csv(full_s4_path)
        cmp = full_s4[["varijabla", "p_value", "p_bonferroni",
                       "effect_value", "znacajnost_005_bonf"]].rename(
                columns={"p_value":"p_full", "p_bonferroni":"p_bonf_full",
                         "effect_value":"eff_full", "znacajnost_005_bonf":"bonf_full"})
        thin = s4_thin[["varijabla", "p_value", "p_bonferroni",
                        "effect_value", "znacajnost_005_bonf"]].rename(
                columns={"p_value":"p_thin", "p_bonferroni":"p_bonf_thin",
                         "effect_value":"eff_thin", "znacajnost_005_bonf":"bonf_thin"})
        usp = cmp.merge(thin, on="varijabla")
        usp["preserved"] = usp["bonf_full"] & usp["bonf_thin"]
        usp["lost"]      = usp["bonf_full"] & ~usp["bonf_thin"]
        usp["gained"]    = ~usp["bonf_full"] & usp["bonf_thin"]
        usp.to_csv(os.path.join(OUT_DIR, "usporedba_thin_vs_full.csv"),
                   index=False, encoding="utf-8")

        n_pres = int(usp["preserved"].sum())
        n_lost = int(usp["lost"].sum())
        n_gain = int(usp["gained"].sum())

        print(f"  Bonferroni-znacajne PRIJE thinninga:  {int(usp['bonf_full'].sum())}")
        print(f"  Bonferroni-znacajne POSLIJE thinninga: {int(usp['bonf_thin'].sum())}")
        print(f"  Preservirane (sig pre I post):         {n_pres}")
        print(f"  Izgubljene  (sig samo pre):            {n_lost}")
        print(f"  Dobivene    (sig samo post):           {n_gain}")
        print()
        if n_lost > 0:
            print("  IZGUBLJENE VARIJABLE (vjerojatno prostorni artefakt):")
            for _, r in usp[usp["lost"]].iterrows():
                print(f"    - {r['varijabla']:<22s}  p_full={r['p_bonf_full']:.4f}  "
                      f"p_thin={r['p_bonf_thin']:.4f}")
        if n_pres > 0:
            print("  PRESERVIRANE VARIJABLE (vjerojatno realne preferencije):")
            for _, r in usp[usp["preserved"]].iterrows():
                print(f"    - {r['varijabla']:<22s}  p_full={r['p_bonf_full']:.4f}  "
                      f"p_thin={r['p_bonf_thin']:.4f}")


if __name__ == "__main__":
    main()
