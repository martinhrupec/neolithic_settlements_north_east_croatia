"""
Prostorno klasteriranje kontinuiranih naselja
=============================================

Cilj: testirati je li grupiranje kontinuiranih naselja prostorni fenomen
(tj. klasteriraju li se kontinuirana nalazišta medu sobom više nego što
bi se očekivalo slučajem), ili je to artefakt thinned/small-n analize.

Ako je klasteriranje realno -> bliska kontinuirana nalazišta su nezavisni
dokazi istog ekološkog niša, a ne pseudoreplikati -> spatial thinning nije
prikladan arbitar za S4.

Testovi:
  1) Deskriptiva: konveksni hull, mean NN, centroid za kontinuirana
     vs jednofazna (rano + kasno zajedno)
  2) Moran's I na binarnom indikatoru 'is_kontinuirano' unutar svih
     274 neolitičkih nalazišta — mjeri prostorno klasteriranje
  3) Join count test (k=8 NN): kontinuirano-kontinuirano parovi vs slučaj
  4) NN analiza: je li prosječna udaljenost kontinuirano->najblizi_kontinuirani
     manja nego što bi se očekivalo kod slučajnog rasporeivanja iste
     n-ice unutar neolitičkog uzorka? (permutacijski test)

Output: konzola + 02_klasteriranje_summary.csv
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree, ConvexHull


ROOT       = r"c:\Users\Martin\Desktop\skripte_za_diplomski\statisticki_testovi"
MASTER     = os.path.join(ROOT, "master_dataset.csv")
COORDS_CSV = os.path.join(ROOT, "01_prostorna_autokorelacija", "neolitik_coords.csv")
OUT_DIR    = os.path.join(ROOT, "05_prostorna_po_fazama")
OUT_CSV    = os.path.join(OUT_DIR, "02_klasteriranje_summary.csv")

K_NEIGHBORS    = 8
N_PERMUTATIONS = 999
RANDOM_SEED    = 42


# ---------------------------------------------------------------------------
#  Pomocne funkcije
# ---------------------------------------------------------------------------

def build_knn_weights(coords, k):
    n = len(coords)
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)
    W = np.zeros((n, n))
    for i in range(n):
        for j in idx[i, 1:]:
            W[i, j] = 1.0
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return W / rs, idx[:, 1:]


def morans_i(values, W, n_perm=999, seed=42):
    values = np.asarray(values, dtype=float)
    n  = len(values)
    z  = values - values.mean()
    s2 = (z * z).sum()
    if s2 == 0:
        return float("nan"), float("nan")
    W_sum = W.sum()
    I_obs = (n / W_sum) * float(z @ W @ z) / s2
    rng = np.random.default_rng(seed)
    I_perm = np.empty(n_perm)
    for p in range(n_perm):
        z_p = rng.permutation(z)
        I_perm[p] = (n / W_sum) * float(z_p @ W @ z_p) / s2
    p_val = (np.sum(np.abs(I_perm) >= abs(I_obs)) + 1) / (n_perm + 1)
    return float(I_obs), float(p_val)


def join_count(labels, neighbors, n_perm=999, seed=42):
    labels = np.asarray(labels, dtype=int)

    def counts(lab):
        BB = WW = BW = 0
        for i, neigh in enumerate(neighbors):
            for j in neigh:
                if j <= i:
                    continue
                if lab[i] == 1 and lab[j] == 1:
                    BB += 1
                elif lab[i] == 0 and lab[j] == 0:
                    WW += 1
                else:
                    BW += 1
        return BB, WW, BW

    BB_obs, WW_obs, BW_obs = counts(labels)
    rng = np.random.default_rng(seed)
    BB_p = np.empty(n_perm); WW_p = np.empty(n_perm); BW_p = np.empty(n_perm)
    for p in range(n_perm):
        lab_p = rng.permutation(labels)
        BB_p[p], WW_p[p], BW_p[p] = counts(lab_p)

    p_BB = (np.sum(BB_p >= BB_obs) + 1) / (n_perm + 1)
    p_WW = (np.sum(WW_p >= WW_obs) + 1) / (n_perm + 1)
    p_BW = (np.sum(BW_p <= BW_obs) + 1) / (n_perm + 1)
    return {
        "BB_obs": int(BB_obs), "WW_obs": int(WW_obs), "BW_obs": int(BW_obs),
        "BB_exp": float(BB_p.mean()), "WW_exp": float(WW_p.mean()),
        "BW_exp": float(BW_p.mean()),
        "p_BB": float(p_BB), "p_WW": float(p_WW), "p_BW": float(p_BW),
    }


def mean_nn_within_group(group_coords, all_coords):
    """
    Za svaku tocku u group_coords: udaljenost do najblizeg susjeda
    koji je TAKOER u group_coords (ne u all_coords).
    """
    if len(group_coords) < 2:
        return float("nan")
    tree = cKDTree(group_coords)
    # k=2: prvi je sama tocka (dist=0), drugi je pravi susjed
    d, _ = tree.query(group_coords, k=2)
    return float(d[:, 1].mean())


def permutation_nn_test(is_kont, xy, n_perm=999, seed=42):
    """
    Permutacijski test: je li mean_NN unutar kontinuiranih manji nego
    što bi se ocekivalo ako bi se isti broj tocaka rasporedio slucajno
    medu svim neolitickim nalazistima?

    Vraca (obs_nn_km, p_value).
    obs_nn_km: opazena mean NN unutar kontinuiranih (km)
    p_value:   udio permutacija s jednako malim ili manjim mean_NN
    """
    kont_idx = np.where(is_kont)[0]
    n_kont   = len(kont_idx)
    kont_xy  = xy[kont_idx]
    obs_nn   = mean_nn_within_group(kont_xy, xy)

    rng = np.random.default_rng(seed)
    perm_nn = np.empty(n_perm)
    for p in range(n_perm):
        perm_idx = rng.choice(len(xy), size=n_kont, replace=False)
        perm_nn[p] = mean_nn_within_group(xy[perm_idx], xy)

    # mali mean_NN = jace klasteriranje; p = koliko permutacija je <= obs
    p_val = (np.sum(perm_nn <= obs_nn) + 1) / (n_perm + 1)
    return float(obs_nn), float(perm_nn.mean()), float(p_val)


def convex_hull_area_km2(coords):
    if len(coords) < 3:
        return float("nan")
    return ConvexHull(coords).volume / 1e6   # 2D: volume = area


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df     = pd.read_csv(MASTER)
    coords = pd.read_csv(COORDS_CSV)

    neo    = df[df.tip_sloja == "neolitik"].copy()
    merged = neo.merge(coords, left_on="fid_raw", right_on="fid", how="inner")

    print(f"Neoliticka nalazista ucitana: n = {len(merged)}")

    kont  = merged[merged.kontinuirano == True]
    jedno = merged[(merged.samo_rano == True) | (merged.samo_kasno == True)]

    print(f"  kontinuirano: n = {len(kont)}")
    print(f"  jednofazna (rano+kasno): n = {len(jedno)}")
    print()

    xy_all  = merged[["x", "y"]].values
    xy_kont = kont[["x", "y"]].values
    xy_jed  = jedno[["x", "y"]].values

    # ---- 1) Deskriptiva ----
    print("=" * 70)
    print("1) PROSTORNA DESKRIPTIVA")
    print("=" * 70)
    print(f"\n{'grupa':<20}{'n':>5}{'hull_km2':>12}{'mean_NN_km':>13}"
          f"{'x_mean':>12}{'y_mean':>12}{'x_sd':>10}{'y_sd':>10}")
    print("-" * 84)

    desc_rows = []
    for label, xy in [("kontinuirano", xy_kont), ("jednofazna", xy_jed),
                      ("sva_neoliticka", xy_all)]:
        if len(xy) < 3:
            continue
        hull = convex_hull_area_km2(xy)
        nn   = mean_nn_within_group(xy, xy) / 1000
        xm, ym = xy[:, 0].mean(), xy[:, 1].mean()
        xs, ys = xy[:, 0].std(),  xy[:, 1].std()
        print(f"{label:<20}{len(xy):>5}{hull:>12.1f}{nn:>13.2f}"
              f"{xm:>12.0f}{ym:>12.0f}{xs:>10.0f}{ys:>10.0f}")
        desc_rows.append({"grupa": label, "n": len(xy), "hull_km2": round(hull, 1),
                           "mean_nn_km": round(nn, 3),
                           "x_mean": round(xm), "y_mean": round(ym),
                           "x_sd": round(xs), "y_sd": round(ys)})

    # ---- 2) Moran's I na is_kontinuirano ----
    print()
    print("=" * 70)
    print("2) MORAN'S I na 'is_kontinuirano' (sva 274 neoliticka nalazista)")
    print("=" * 70)

    W, neighbors = build_knn_weights(xy_all, K_NEIGHBORS)
    is_kont_bin  = merged["kontinuirano"].astype(int).values

    I, p = morans_i(is_kont_bin, W, N_PERMUTATIONS, RANDOM_SEED)
    print(f"\n  n = {len(merged)}  (k = {K_NEIGHBORS} susjeda)")
    print(f"  Moran's I = {I:+.4f},  p = {p:.4g}")

    if p < 0.05:
        if abs(I) >= 0.30:
            strength = "jak"
        elif abs(I) >= 0.10:
            strength = "umjeren"
        else:
            strength = "slab"
        interp = (f"ZNACAJNO ({strength}) prostorno klasteriranje kontinuiranih nalazista.\n"
                  f"  -> Kontinuirana nalazista nisu slucajno rasprsena medu neolitickim\n"
                  f"     uzorkom — grupiraju se u prostoru vise nego sto bi se ocekivalo\n"
                  f"     slucajem. Thinning kao arbitar robusnosti upitan za S4.")
    else:
        interp = ("Nema znacajnog prostornog klasteriranja kontinuiranih nalazista.\n"
                  "  -> Rasporedena su slucajno medu neolitickim uzorkom.\n"
                  "     Thinning kao kontrola pseudoreplikacije je opravdan.")
    print(f"\n  Interpretacija: {interp}")

    # ---- 3) Join count test ----
    print()
    print("=" * 70)
    print("3) JOIN COUNT TEST (k=8 NN susjedi)")
    print("   K-K = kontinuirano-kontinuirano parovi")
    print("   J-J = jednofazno-jednofazno parovi")
    print("   K-J = mijesani parovi")
    print("=" * 70)

    jc = join_count(is_kont_bin, neighbors, N_PERMUTATIONS, RANDOM_SEED)
    print(f"\n             Opazeno   Ocekivano      p")
    print(f"  K-K (BB): {jc['BB_obs']:>7d}   {jc['BB_exp']:>9.1f}   {jc['p_BB']:.4g}  "
          f"{'*' if jc['p_BB'] < 0.05 else ''}")
    print(f"  J-J (WW): {jc['WW_obs']:>7d}   {jc['WW_exp']:>9.1f}   {jc['p_WW']:.4g}  "
          f"{'*' if jc['p_WW'] < 0.05 else ''}")
    print(f"  K-J (BW): {jc['BW_obs']:>7d}   {jc['BW_exp']:>9.1f}   {jc['p_BW']:.4g}  "
          f"{'*' if jc['p_BW'] < 0.05 else ''}")

    if jc["p_BB"] < 0.05:
        jc_interp = ("Kontinuirana nalazista imaju VISE medusobnih susjedstava nego\n"
                     "  sto bi bilo slucajno -> potvrda prostornog klasteriranja.")
    else:
        jc_interp = ("Kontinuirana nalazista NEMAJU vise medusobnih susjedstava\n"
                     "  nego sto bi bilo slucajno.")
    print(f"\n  -> {jc_interp}")

    # ---- 4) Permutacijski test mean_NN ----
    print()
    print("=" * 70)
    print("4) PERMUTACIJSKI TEST mean NN UNUTAR KONTINUIRANIH")
    print(f"   ({N_PERMUTATIONS} permutacija, n_kontinuiranih = {len(kont)})")
    print("=" * 70)

    obs_nn, exp_nn, p_nn = permutation_nn_test(
        merged["kontinuirano"].values, xy_all, N_PERMUTATIONS, RANDOM_SEED
    )
    print(f"\n  Opazena mean NN unutar kontinuiranih: {obs_nn/1000:.2f} km")
    print(f"  Ocekivana mean NN (permutacije):      {exp_nn/1000:.2f} km")
    print(f"  p-value (jednosmjerni, manji=klaster): {p_nn:.4g}")

    if p_nn < 0.05:
        nn_interp = ("Kontinuirana nalazista su BLIZA jedna drugima nego sto bi\n"
                     "  bilo slucajno za isti n unutar neolitickog uzorka.\n"
                     "  -> Klasteriranje je statisticki znacajno.")
    else:
        nn_interp = ("Nema dokaza da su kontinuirana nalazista bliza jedna drugima\n"
                     "  nego sto bi bilo slucajno.")
    print(f"\n  -> {nn_interp}")

    # ---- 5) Centroid distance kontinuirano vs jednofazna ----
    print()
    print("=" * 70)
    print("5) UDALJENOST CENTROIDA")
    print("=" * 70)
    c_kont = xy_kont.mean(axis=0)
    c_jed  = xy_jed.mean(axis=0)
    c_all  = xy_all.mean(axis=0)
    d_kj   = np.sqrt(((c_kont - c_jed)**2).sum()) / 1000
    d_ka   = np.sqrt(((c_kont - c_all)**2).sum()) / 1000
    print(f"\n  Centroid kontinuirano: x={c_kont[0]:.0f}, y={c_kont[1]:.0f}")
    print(f"  Centroid jednofazna:   x={c_jed[0]:.0f},  y={c_jed[1]:.0f}")
    print(f"  Centroid svi neolitik: x={c_all[0]:.0f},  y={c_all[1]:.0f}")
    print(f"\n  d(kontinuirano, jednofazna) = {d_kj:.1f} km")
    print(f"  d(kontinuirano, svi neo)    = {d_ka:.1f} km")
    if d_kj > 10:
        print("  -> Kontinuirana nalazista su geografski odvojena od jednofaznih")
        print("     (centroidi > 10 km udaljeni). Moguci zasebni geografski niz.")
    else:
        print("  -> Centroidi su relativno blizu — geografska separacija nije jaka.")

    # ---- Spremi summary ----
    summary = {
        "morans_I": round(I, 4),
        "morans_p": round(p, 4),
        "morans_znacajno": p < 0.05,
        "joincount_KK_obs": jc["BB_obs"],
        "joincount_KK_exp": round(jc["BB_exp"], 1),
        "joincount_KK_p": round(jc["p_BB"], 4),
        "joincount_KJ_p": round(jc["p_BW"], 4),
        "perm_nn_obs_km": round(obs_nn / 1000, 3),
        "perm_nn_exp_km": round(exp_nn / 1000, 3),
        "perm_nn_p": round(p_nn, 4),
        "centroid_dist_kont_jedno_km": round(d_kj, 1),
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(OUT_DIR, "02_klasteriranje_summary.csv"),
        index=False, encoding="utf-8"
    )
    print(f"\nSpremi u: {OUT_DIR}/02_klasteriranje_summary.csv")
    print("\n" + "=" * 70)
    print("GOTOVO.")
    print("=" * 70)


if __name__ == "__main__":
    main()
