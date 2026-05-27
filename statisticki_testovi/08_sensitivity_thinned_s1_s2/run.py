"""
Sensitivity check za S1 i S2: 500 m thinning na robusne varijable
==================================================================

Pitanje: jesu li glavni nalazi S1 (background vs neolitik) i S2
(random_ceste vs neolitik) robusni na prostornu pseudoreplikaciju?

Strategija:
  - Thinning samo na NEOLITIK stranu (random_ceste nemamo kao coords sloj)
  - Tretira se kao "konzervativan" pristup: reduciramo "test" stranu,
    ali ne i kontrolnu — ako signal preživi, otporan je na klasterizaciju
  - 500m threshold (kao u 07_thinned_analize) — uklanja samo najbliske
    pseudoreplikate (vjerojatno isti naseobinski kompleks)

Testira se SAMO 5 robusnih varijabli iz S1/S2 (one koje su prosle
najmanje jedan od ta dva scenarija):
  - dist_rijeka_korig   (kontinuirana — fallback na potpuno_nasumicni)
  - strahler            (ordinalna     — background_strahler.csv)
  - rel_vis_100_250     (kontinuirana — fallback na potpuno_nasumicni)
  - sm_r100             (kategorijska — background_sm.csv)
  - vtt_r100            (kategorijska — background_vtt.csv; S2 emergent)

Output: rezultati_sensitivity.csv — pre/post usporedba
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
BG_DIR     = os.path.join(ROOT, "background")
COORDS_CSV = os.path.join(ROOT, "01_prostorna_autokorelacija", "neolitik_coords.csv")
OUT_DIR    = os.path.join(ROOT, "08_sensitivity_thinned_s1_s2")
OUT_CSV    = os.path.join(OUT_DIR, "rezultati_sensitivity.csv")

THRESHOLD_M = 500
SEED        = 42

# Originalne p-vrijednosti iz S1 i S2 — za usporedbu pre/post
ORIGINAL_S1 = {
    "dist_rijeka_korig":   5.9e-9,
    "strahler":            2.3e-6,
    "rel_vis_100_250":     2.3e-6,
    "sm_r100":             None,   # popunit cu iz CSV-a
    "vtt_r100":            0.034,  # nije bila znacajna Bonf
}
ORIGINAL_S2 = {
    "dist_rijeka_korig":   0.003,
    "strahler":            5e-4,
    "rel_vis_100_250":     9.4e-5,
    "sm_r100":             None,
    "vtt_r100":            8.1e-10,
}


# ---------------------------------------------------------------------------
#  Greedy thinning (kao u 07_thinned_analize)
# ---------------------------------------------------------------------------

def greedy_thin(coords, threshold_m, seed=42):
    n = len(coords)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    excluded = np.zeros(n, dtype=bool)
    tree = cKDTree(coords)
    kept = []
    for i in order:
        if excluded[i]:
            continue
        kept.append(i)
        for j in tree.query_ball_point(coords[i], threshold_m):
            if j != i:
                excluded[j] = True
    return sorted(kept)


# ---------------------------------------------------------------------------
#  S1 test helpers — weighted 1-sample
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


# ---------------------------------------------------------------------------
#  Chi-square goodness-of-fit (kategorijska vs background proporcije)
# ---------------------------------------------------------------------------

def chi2_gof(observed_counts, expected_proportions):
    """
    1-sample chi-square GoF: opazene frekvencije vs ocekivane proporcije.
    Spaja rijetke kategorije (exp < 5) u jednu 'rare' grupu da bi se
    izbjegao nan i poboljsala validnost chi² aproksimacije.
    """
    obs = np.asarray(observed_counts, dtype=float)
    p   = np.asarray(expected_proportions, dtype=float)
    p   = p / p.sum()
    n   = obs.sum()
    exp = n * p

    # Spoji rijetke kategorije u jednu
    rare = exp < 5
    if rare.any() and (~rare).sum() >= 2:
        obs_keep = list(obs[~rare]) + [obs[rare].sum()]
        exp_keep = list(exp[~rare]) + [exp[rare].sum()]
        obs, exp = np.array(obs_keep), np.array(exp_keep)
    else:
        obs, exp = obs[exp > 0], exp[exp > 0]

    if len(obs) < 2:
        return float("nan"), float("nan")
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    dof  = len(obs) - 1
    p_val = float(stats.chi2.sf(chi2, dof))
    return chi2, p_val


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df     = pd.read_csv(MASTER)
    coords = pd.read_csv(COORDS_CSV)
    neo    = df[df.tip_sloja == "neolitik"].copy()
    ctr_p  = df[df.tip_sloja == "potpuno_nasumicni"].copy()
    ctr_c  = df[df.tip_sloja == "nasumicni_ceste"].copy()

    merged = neo.merge(coords, left_on="fid_raw", right_on="fid", how="inner")
    print(f"Spojeno: {len(merged)} neolitickih nalazista")

    # Thinning na neolitik
    keep   = greedy_thin(merged[["x","y"]].values, THRESHOLD_M, SEED)
    neo_thin = merged.iloc[keep].reset_index(drop=True)
    print(f"Thinning {THRESHOLD_M} m: {len(merged)} -> {len(neo_thin)} "
          f"({100*len(neo_thin)/len(merged):.0f}%)\n")

    n_full = len(merged)
    n_thin = len(neo_thin)

    rows = []

    # ===========================================================
    # S1: 1-sample weighted KS / chi-square GoF
    # ===========================================================
    print("=" * 72)
    print("S1: NEOLITIK (thinned) vs BACKGROUND")
    print("=" * 72)

    # --- dist_rijeka_korig: fallback na potpuno_nasumicni ---
    a_full = merged["dist_rijeka_korig"].dropna().values
    a_thin = neo_thin["dist_rijeka_korig"].dropna().values
    b      = ctr_p["dist_rijeka_korig"].dropna().values
    _,  p_full_s1 = stats.ks_2samp(a_full, b)
    _,  p_thin_s1 = stats.ks_2samp(a_thin, b)
    rows.append({
        "scenarij": "S1", "varijabla": "dist_rijeka_korig",
        "test":     "KS_2samp_vs_potpuno_nasumicni",
        "n_full":   len(a_full), "n_thin": len(a_thin),
        "p_full":   p_full_s1,   "p_thin": p_thin_s1,
    })
    print(f"  dist_rijeka_korig:   p_full={p_full_s1:.2e}  p_thin={p_thin_s1:.2e}")

    # --- rel_vis_100_250: fallback ---
    a_full = merged["rel_vis_100_250"].dropna().values
    a_thin = neo_thin["rel_vis_100_250"].dropna().values
    b      = ctr_p["rel_vis_100_250"].dropna().values
    _,  p_full_s1 = stats.ks_2samp(a_full, b)
    _,  p_thin_s1 = stats.ks_2samp(a_thin, b)
    rows.append({
        "scenarij": "S1", "varijabla": "rel_vis_100_250",
        "test":     "KS_2samp_vs_potpuno_nasumicni",
        "n_full":   len(a_full), "n_thin": len(a_thin),
        "p_full":   p_full_s1,   "p_thin": p_thin_s1,
    })
    print(f"  rel_vis_100_250:     p_full={p_full_s1:.2e}  p_thin={p_thin_s1:.2e}")

    # --- strahler: chi2 GoF vs background_strahler.csv ---
    bg = pd.read_csv(os.path.join(BG_DIR, "background_strahler.csv"))
    bg_props = bg.set_index("strahler")["duljina_km"]  # tezine = duljina
    bg_props = bg_props / bg_props.sum()

    def strahler_gof(neo_series):
        cats = sorted(bg_props.index.tolist())
        obs  = np.array([(neo_series.astype(int) == c).sum() for c in cats], dtype=float)
        exp  = bg_props.reindex(cats).values
        return chi2_gof(obs, exp)

    _,  p_full_s1 = strahler_gof(merged["strahler"].dropna())
    _,  p_thin_s1 = strahler_gof(neo_thin["strahler"].dropna())
    rows.append({
        "scenarij": "S1", "varijabla": "strahler",
        "test":     "chi2_GoF_vs_background",
        "n_full":   int(merged["strahler"].dropna().shape[0]),
        "n_thin":   int(neo_thin["strahler"].dropna().shape[0]),
        "p_full":   p_full_s1, "p_thin": p_thin_s1,
    })
    print(f"  strahler:            p_full={p_full_s1:.2e}  p_thin={p_thin_s1:.2e}")

    # --- sm_r100: chi2 GoF vs background_sm.csv ---
    bg_sm = pd.read_csv(os.path.join(BG_DIR, "background_sm.csv"))
    bg_sm_props = bg_sm.set_index("kategorija")["n_piksela"]
    bg_sm_props = bg_sm_props / bg_sm_props.sum()

    def sm_gof(neo_series):
        cats = bg_sm_props.index.tolist()
        obs  = np.array([(neo_series == c).sum() for c in cats], dtype=float)
        exp  = bg_sm_props.values
        return chi2_gof(obs, exp)

    _,  p_full_s1 = sm_gof(merged["sm_r100"].dropna())
    _,  p_thin_s1 = sm_gof(neo_thin["sm_r100"].dropna())
    rows.append({
        "scenarij": "S1", "varijabla": "sm_r100",
        "test":     "chi2_GoF_vs_background",
        "n_full":   int(merged["sm_r100"].dropna().shape[0]),
        "n_thin":   int(neo_thin["sm_r100"].dropna().shape[0]),
        "p_full":   p_full_s1, "p_thin": p_thin_s1,
    })
    print(f"  sm_r100:             p_full={p_full_s1:.2e}  p_thin={p_thin_s1:.2e}")

    # --- vtt_r100: chi2 GoF vs background_vtt.csv ---
    bg_vtt = pd.read_csv(os.path.join(BG_DIR, "background_vtt.csv"))
    bg_vtt_props = bg_vtt.set_index("tip_tla")["n_piksela"]
    bg_vtt_props = bg_vtt_props / bg_vtt_props.sum()

    def vtt_gof(neo_series):
        cats = bg_vtt_props.index.tolist()
        obs  = np.array([(neo_series == c).sum() for c in cats], dtype=float)
        exp  = bg_vtt_props.values
        return chi2_gof(obs, exp)

    _,  p_full_s1 = vtt_gof(merged["vtt_r100"].dropna())
    _,  p_thin_s1 = vtt_gof(neo_thin["vtt_r100"].dropna())
    rows.append({
        "scenarij": "S1", "varijabla": "vtt_r100",
        "test":     "chi2_GoF_vs_background",
        "n_full":   int(merged["vtt_r100"].dropna().shape[0]),
        "n_thin":   int(neo_thin["vtt_r100"].dropna().shape[0]),
        "p_full":   p_full_s1, "p_thin": p_thin_s1,
    })
    print(f"  vtt_r100:            p_full={p_full_s1:.2e}  p_thin={p_thin_s1:.2e}")

    # ===========================================================
    # S2: 2-sample tests vs nasumicni_ceste
    # ===========================================================
    print()
    print("=" * 72)
    print("S2: NEOLITIK (thinned) vs NASUMICNI_CESTE (full)")
    print("=" * 72)

    # --- dist_rijeka_korig ---
    a_full = merged["dist_rijeka_korig"].dropna().values
    a_thin = neo_thin["dist_rijeka_korig"].dropna().values
    b      = ctr_c["dist_rijeka_korig"].dropna().values
    _,  p_full_s2 = stats.ks_2samp(a_full, b)
    _,  p_thin_s2 = stats.ks_2samp(a_thin, b)
    rows.append({
        "scenarij": "S2", "varijabla": "dist_rijeka_korig",
        "test":     "KS_2samp",
        "n_full":   len(a_full), "n_thin": len(a_thin),
        "p_full":   p_full_s2, "p_thin": p_thin_s2,
    })
    print(f"  dist_rijeka_korig:   p_full={p_full_s2:.2e}  p_thin={p_thin_s2:.2e}")

    # --- rel_vis_100_250 ---
    a_full = merged["rel_vis_100_250"].dropna().values
    a_thin = neo_thin["rel_vis_100_250"].dropna().values
    b      = ctr_c["rel_vis_100_250"].dropna().values
    _,  p_full_s2 = stats.ks_2samp(a_full, b)
    _,  p_thin_s2 = stats.ks_2samp(a_thin, b)
    rows.append({
        "scenarij": "S2", "varijabla": "rel_vis_100_250",
        "test":     "KS_2samp",
        "n_full":   len(a_full), "n_thin": len(a_thin),
        "p_full":   p_full_s2, "p_thin": p_thin_s2,
    })
    print(f"  rel_vis_100_250:     p_full={p_full_s2:.2e}  p_thin={p_thin_s2:.2e}")

    # --- strahler ---
    def strahler_2samp(a, b):
        cats = sorted(set(a.astype(int).unique()) | set(b.astype(int).unique()))
        table = np.array([
            [int((a.astype(int) == c).sum()) for c in cats],
            [int((b.astype(int) == c).sum()) for c in cats],
        ])
        keep = table.sum(axis=0) > 0
        table = table[:, keep]
        chi2, p, _, _ = stats.chi2_contingency(table)
        return chi2, p

    _,  p_full_s2 = strahler_2samp(merged["strahler"].dropna(), ctr_c["strahler"].dropna())
    _,  p_thin_s2 = strahler_2samp(neo_thin["strahler"].dropna(), ctr_c["strahler"].dropna())
    rows.append({
        "scenarij": "S2", "varijabla": "strahler",
        "test":     "chi2_2samp",
        "n_full":   int(merged["strahler"].dropna().shape[0]),
        "n_thin":   int(neo_thin["strahler"].dropna().shape[0]),
        "p_full":   p_full_s2, "p_thin": p_thin_s2,
    })
    print(f"  strahler:            p_full={p_full_s2:.2e}  p_thin={p_thin_s2:.2e}")

    # --- sm_r100 ---
    def cat_2samp(a, b):
        cats = sorted(set(a.unique()) | set(b.unique()))
        table = np.array([
            [int((a == c).sum()) for c in cats],
            [int((b == c).sum()) for c in cats],
        ])
        keep = table.sum(axis=0) > 0
        table = table[:, keep]
        chi2, p, _, _ = stats.chi2_contingency(table)
        return chi2, p

    _,  p_full_s2 = cat_2samp(merged["sm_r100"].dropna(), ctr_c["sm_r100"].dropna())
    _,  p_thin_s2 = cat_2samp(neo_thin["sm_r100"].dropna(), ctr_c["sm_r100"].dropna())
    rows.append({
        "scenarij": "S2", "varijabla": "sm_r100",
        "test":     "chi2_2samp",
        "n_full":   int(merged["sm_r100"].dropna().shape[0]),
        "n_thin":   int(neo_thin["sm_r100"].dropna().shape[0]),
        "p_full":   p_full_s2, "p_thin": p_thin_s2,
    })
    print(f"  sm_r100:             p_full={p_full_s2:.2e}  p_thin={p_thin_s2:.2e}")

    # --- vtt_r100 ---
    _,  p_full_s2 = cat_2samp(merged["vtt_r100"].dropna(), ctr_c["vtt_r100"].dropna())
    _,  p_thin_s2 = cat_2samp(neo_thin["vtt_r100"].dropna(), ctr_c["vtt_r100"].dropna())
    rows.append({
        "scenarij": "S2", "varijabla": "vtt_r100",
        "test":     "chi2_2samp",
        "n_full":   int(merged["vtt_r100"].dropna().shape[0]),
        "n_thin":   int(neo_thin["vtt_r100"].dropna().shape[0]),
        "p_full":   p_full_s2, "p_thin": p_thin_s2,
    })
    print(f"  vtt_r100:            p_full={p_full_s2:.2e}  p_thin={p_thin_s2:.2e}")

    # ---- Finaliziraj ----
    out = pd.DataFrame(rows)

    # Dodaj Bonferroni (26 testova u svakom originalnom scenariju)
    BONF_N = 26
    out["p_full_bonf"] = (out["p_full"] * BONF_N).clip(upper=1.0)
    out["p_thin_bonf"] = (out["p_thin"] * BONF_N).clip(upper=1.0)
    out["bonf_full"]   = out["p_full_bonf"] < 0.05
    out["bonf_thin"]   = out["p_thin_bonf"] < 0.05
    out["status"]      = out.apply(lambda r:
        "PREZIVJELA" if r["bonf_full"] and r["bonf_thin"] else
        "IZGUBLJENA" if r["bonf_full"] and not r["bonf_thin"] else
        "I dalje n.s." if not r["bonf_full"] and not r["bonf_thin"] else
        "EMERGENT (post)", axis=1)

    out.to_csv(OUT_CSV, index=False, encoding="utf-8")

    # ---- Sazetak ----
    print()
    print("=" * 72)
    print("SAZETAK — PRE/POST USPOREDBA (Bonferroni adjusted, divisor=26)")
    print("=" * 72)
    print()
    with pd.option_context("display.width", 200, "display.max_rows", None,
                           "display.float_format", "{:.3g}".format):
        print(out[["scenarij","varijabla","n_full","n_thin",
                   "p_full_bonf","p_thin_bonf","status"]].to_string(index=False))

    n_preserved  = int((out["status"] == "PREZIVJELA").sum())
    n_lost       = int((out["status"] == "IZGUBLJENA").sum())
    n_ns_both    = int((out["status"] == "I dalje n.s.").sum())

    print()
    print(f"Preservirane (Bonf. sig pre I post): {n_preserved}")
    print(f"Izgubljene  (Bonf. sig samo pre):    {n_lost}")
    print(f"Ostaju n.s.:                          {n_ns_both}")
    print()
    print(f"Spremi u: {OUT_CSV}")


if __name__ == "__main__":
    main()
