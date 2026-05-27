"""
morans_i.py
===========
Globalni Moran's I za sumnjive varijable na neolitickim nalazistima.

Cilj: provjeriti je li signal iz scenarija 1 posljedica prave ekoloske preferencije
ili je geografski artefakt (npr. Fruska gora north-facing klaster).

Pristup:
  - prostorne tezine: k-najblizih susjeda (k=8), row-standardizirano
  - znacajnost: permutacijski test (999 permutacija)
  - kontinuirane varijable: direktan Moran's I na vrijednostima
  - binarne kategorijske (aspect_sn, sm_rN): Moran's I na 0/1 kodiranju
  - aspect_cat4: Moran's I na svakoj kategoriji kao binarni indikator

Interpretacija:
  I > 0  -> slicne vrijednosti se prostorno klasteriraju (vise sumnje na artefakt)
  I < 0  -> slicne vrijednosti se izbjegavaju (rijetko u arheologiji)
  I ~ 0  -> nema prostorne strukture, signal je prostorno distribuiran (robustan)

  |I| < 0.10  zanemariv
  |I| < 0.30  umjeren
  |I| >= 0.30 jak

Output: rezultati.csv

Pokretanje: obican Python (NE u QGIS-u). Treba samo numpy, scipy, pandas.
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ROOT       = r"c:\Users\Martin\Desktop\skripte_za_diplomski\statisticki_testovi"
MASTER     = os.path.join(ROOT, "master_dataset.csv")
COORDS_CSV = os.path.join(ROOT, "02_prostorna_autokorelacija", "neolitik_coords.csv")
OUT_DIR    = os.path.join(ROOT, "02_prostorna_autokorelacija")
OUT_CSV    = os.path.join(OUT_DIR, "rezultati.csv")

K_NEIGHBORS = 8
N_PERMUTATIONS = 999
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
#  Moran's I implementacija (pure numpy)
# ---------------------------------------------------------------------------

def build_knn_weights(coords, k):
    """Row-standardizirana k-NN matrica tezina."""
    n = len(coords)
    tree = cKDTree(coords)
    # k+1 jer prvi sused je sama tocka
    _, idx = tree.query(coords, k=k + 1)
    W = np.zeros((n, n))
    for i in range(n):
        for j in idx[i, 1:]:        # preskoci self
            W[i, j] = 1.0
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0   # safety
    return W / row_sums


def morans_i(values, W, n_permutations=999, seed=42):
    """Globalni Moran's I + permutacijski p-value (dvostrani)."""
    values = np.asarray(values, dtype=float)
    n      = len(values)
    z      = values - values.mean()
    s2     = np.sum(z * z)
    if s2 == 0:
        return float("nan"), float("nan")
    W_sum  = W.sum()
    num    = float(z @ W @ z)
    I_obs  = (n / W_sum) * (num / s2)

    rng = np.random.default_rng(seed)
    I_perm = np.empty(n_permutations)
    for p in range(n_permutations):
        z_p = rng.permutation(z)
        num_p = float(z_p @ W @ z_p)
        I_perm[p] = (n / W_sum) * (num_p / s2)

    # dvostrani p-value: koliko apsolutnih permutiranih |I| >= |I_obs|
    p_val = (np.sum(np.abs(I_perm) >= abs(I_obs)) + 1) / (n_permutations + 1)
    return float(I_obs), float(p_val)


def interp_morans(I):
    a = abs(I)
    if a < 0.10: return "zanemariv"
    if a < 0.30: return "umjeren"
    return "jak"


def smjer(I):
    if I > 0.05:  return "klasteriran (slicne vrijednosti su prostorno blizu)"
    if I < -0.05: return "raspršen (slicne vrijednosti se izbjegavaju)"
    return "slucajan"


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

# Sumnjive varijable iz scenarija 1.
# Za svaku: ime stupca u masteru, tip (continuous/binary/multicat)
SUMNJIVE = [
    ("aspect_sn",          "binary",       {"N": 0, "S": 1}),
    ("aspect_cat4",        "multicat",     ["NE", "SE", "SW", "NW"]),
    ("nagib",              "continuous",   None),
    ("tri",                "continuous",   None),
    ("coarse_fragments",   "continuous",   None),
    ("rel_vis_100_250",    "continuous",   None),
    ("sm_r100",            "binary",       {"Suho": 0, "Mocvarno": 1}),
    ("sm_r250",            "binary",       {"Suho": 0, "Mocvarno": 1}),
]


def encode_binary(series, mapping):
    return series.map(mapping)


def encode_indicator(series, category):
    return (series == category).astype(int)


def main():
    if not os.path.exists(COORDS_CSV):
        print("GRESKA: neolitik_coords.csv ne postoji.")
        print(f"Ocekivana lokacija: {COORDS_CSV}")
        print("Prvo pokreni 'export_coords_qgis.py' u QGIS Python konzoli.")
        return

    df     = pd.read_csv(MASTER)
    coords = pd.read_csv(COORDS_CSV)
    neo    = df[df.tip_sloja == "neolitik"].copy()

    # spoj na fid_raw <-> fid
    merged = neo.merge(coords, left_on="fid_raw", right_on="fid", how="inner")
    print(f"Spojeno {len(merged)} nalazista (od {len(neo)} u masteru, "
          f"{len(coords)} u coords).")
    if len(merged) < len(neo):
        missing = set(neo["fid_raw"]) - set(coords["fid"])
        print(f"   nedostaju koordinate za fid: {sorted(missing)[:10]}{'...' if len(missing)>10 else ''}")

    xy = merged[["x", "y"]].values
    W  = build_knn_weights(xy, K_NEIGHBORS)
    print(f"k-NN matrica tezina izgraena (k={K_NEIGHBORS}).\n")

    rows = []
    for col, kind, meta in SUMNJIVE:
        if col not in merged.columns:
            print(f"  PRESKACEM: {col} nema u masteru")
            continue

        if kind == "continuous":
            s = merged[col].dropna()
            sub_idx = s.index
            if len(s) < 20:
                print(f"  PRESKACEM: {col} ima samo n={len(s)}")
                continue
            W_sub = _subset_weights(W, sub_idx, merged.index)
            I, p = morans_i(s.values, W_sub, N_PERMUTATIONS, RANDOM_SEED)
            rows.append(_row(col, "kontinuirana", "—", len(s), I, p))

        elif kind == "binary":
            encoded = encode_binary(merged[col], meta).dropna()
            if encoded.nunique() < 2 or len(encoded) < 20:
                print(f"  PRESKACEM: {col} ima samo n={len(encoded)} ili jednu vrijednost")
                continue
            W_sub = _subset_weights(W, encoded.index, merged.index)
            I, p = morans_i(encoded.values, W_sub, N_PERMUTATIONS, RANDOM_SEED)
            rows.append(_row(col, "binarna", "—", len(encoded), I, p))

        elif kind == "multicat":
            for cat in meta:
                indicator = encode_indicator(merged[col], cat)
                # dropna na originalu (NaN u kategorijskom)
                valid = merged[col].dropna().index
                ind = indicator.loc[valid]
                if ind.sum() < 5 or len(ind) < 20:
                    print(f"  PRESKACEM: {col}={cat} (n_pozitivnih={int(ind.sum())})")
                    continue
                W_sub = _subset_weights(W, ind.index, merged.index)
                I, p  = morans_i(ind.values, W_sub, N_PERMUTATIONS, RANDOM_SEED)
                rows.append(_row(f"{col} = {cat}", "binarna (indikator)", cat,
                                 len(ind), I, p))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nGOTOVO. {len(out)} testova  ->  {OUT_CSV}\n")

    with pd.option_context("display.width", 200,
                           "display.max_rows", None,
                           "display.float_format", "{:.4f}".format):
        print(out.to_string(index=False))


def _subset_weights(W, sub_idx, all_idx):
    """Iz pune matrice izvuci podmatricu samo za odabrane retke/stupce."""
    pos = [list(all_idx).index(i) for i in sub_idx]
    Wsub = W[np.ix_(pos, pos)]
    rs = Wsub.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return Wsub / rs


def _row(varijabla, tip, kategorija, n, I, p):
    sig = p < 0.05
    artefakt = sig and abs(I) >= 0.10
    return {
        "varijabla":   varijabla,
        "tip":         tip,
        "n":           n,
        "Moran_I":     round(I, 4),
        "p_value":     round(p, 4),
        "znacajno":    sig,
        "jacina":      interp_morans(I),
        "smjer":       smjer(I),
        "mozda_artefakt": artefakt,
    }


if __name__ == "__main__":
    main()
