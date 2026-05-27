"""
Prostorna distribucija: samo_rano vs samo_kasno
================================================

Cilj: prije provedbe analiza po fazama, provjeriti jesu li rano i kasno
prostorno razlikujuca (geografski klasterirana) ili wel-pomijesana.

Ako su razlikujuca: razlike u S3 (aps_vis, coarse_fragments, strahler)
su vjerojatno GEOGRAFSKI artefakt, ne temporalna promjena preferencije.
Ako su pomijesana: razlike su vremenske/kulturne.

Testovi:
  1) Centroid, sirina i konveksna ljuska po fazi (osnovna deskriptiva)
  2) Moran's I na binarnom indikatoru "is_rano" — direktno mjeri
     prostornu odvojenost rano vs kasno
  3) Join count test (k-NN, k=8): koliko parova susjeda dijeli istu fazu
     - BB = rano-rano susjedi
     - WW = kasno-kasno susjedi
     - BW = mijesani susjedi
     Visok BB i WW + nizak BW = jaka prostorna segregacija
  4) Udaljenost izmedu centroida grupa

Output: konzola + summary CSV
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree, ConvexHull


ROOT       = r"c:\Users\Martin\Desktop\skripte_za_diplomski\statisticki_testovi"
MASTER     = os.path.join(ROOT, "master_dataset.csv")
COORDS_CSV = os.path.join(ROOT, "01_prostorna_autokorelacija", "neolitik_coords.csv")
OUT_DIR    = os.path.join(ROOT, "05_prostorna_po_fazama")
OUT_CSV    = os.path.join(OUT_DIR, "01_distribucija_summary.csv")

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
    return W / rs, idx[:, 1:]   # i listu susjeda za join count


def morans_i(values, W, n_perm=999, seed=42):
    values = np.asarray(values, dtype=float)
    n = len(values)
    z = values - values.mean()
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
    """
    Join count test na binarnoj varijabli (0/1) s k-NN susjedstvom.
    Vraca BB (1-1), WW (0-0), BW (mijesani) plus permutirane p-vrijednosti.
    """
    labels = np.asarray(labels, dtype=int)

    def counts(lab):
        BB = WW = BW = 0
        for i, neigh in enumerate(neighbors):
            for j in neigh:
                if j <= i:    # da ne brojimo par dva puta
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

    # za segregaciju: BB i WW veci od ocekivanog, BW manji
    p_BB = (np.sum(BB_p >= BB_obs) + 1) / (n_perm + 1)
    p_WW = (np.sum(WW_p >= WW_obs) + 1) / (n_perm + 1)
    p_BW = (np.sum(BW_p <= BW_obs) + 1) / (n_perm + 1)

    return {
        "BB_obs": int(BB_obs), "WW_obs": int(WW_obs), "BW_obs": int(BW_obs),
        "BB_exp": float(BB_p.mean()), "WW_exp": float(WW_p.mean()), "BW_exp": float(BW_p.mean()),
        "p_BB":   float(p_BB), "p_WW": float(p_WW), "p_BW": float(p_BW),
    }


def convex_hull_area(coords):
    if len(coords) < 3:
        return 0.0
    hull = ConvexHull(coords)
    return float(hull.volume)   # u 2D je 'volume' zapravo povrsina


def mean_nn_distance(coords):
    if len(coords) < 2:
        return float("nan")
    tree = cKDTree(coords)
    d, _ = tree.query(coords, k=2)
    return float(d[:, 1].mean())


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df     = pd.read_csv(MASTER)
    coords = pd.read_csv(COORDS_CSV)
    neo    = df[df.tip_sloja == "neolitik"].copy()
    merged = neo.merge(coords, left_on="fid_raw", right_on="fid", how="inner")

    rano  = merged[merged.samo_rano  == True].copy()
    kasno = merged[merged.samo_kasno == True].copy()
    kont  = merged[merged.kontinuirano == True].copy()

    print(f"Ukupno spojeno:   n = {len(merged)} (od {len(neo)} u masteru)")
    print(f"  samo_rano:      n = {len(rano)}")
    print(f"  samo_kasno:     n = {len(kasno)}")
    print(f"  kontinuirano:   n = {len(kont)}")
    print()

    # ---- 1) Deskriptiva po fazi ----
    print("=" * 70)
    print("1) PROSTORNA DESKRIPTIVA PO FAZI")
    print("=" * 70)
    print()
    print(f"{'faza':<14}{'n':>5}{'x_mean':>12}{'y_mean':>12}"
          f"{'x_sd':>10}{'y_sd':>10}{'hull_km2':>12}{'mean_nn_km':>13}")
    print("-" * 80)

    desc_rows = []
    for label, sub in [("rano", rano), ("kasno", kasno), ("kontinuirano", kont)]:
        if len(sub) < 3:
            continue
        xy = sub[["x", "y"]].values
        x_m, y_m = xy[:, 0].mean(), xy[:, 1].mean()
        x_s, y_s = xy[:, 0].std(),  xy[:, 1].std()
        area_km2 = convex_hull_area(xy) / 1e6
        nn_km    = mean_nn_distance(xy) / 1000
        print(f"{label:<14}{len(sub):>5}{x_m:>12.0f}{y_m:>12.0f}"
              f"{x_s:>10.0f}{y_s:>10.0f}{area_km2:>12.1f}{nn_km:>13.2f}")
        desc_rows.append({
            "faza": label, "n": len(sub),
            "x_mean": x_m, "y_mean": y_m,
            "x_sd":   x_s, "y_sd":   y_s,
            "hull_km2": area_km2,
            "mean_nn_km": nn_km,
        })

    # ---- 2) Udaljenost izmedu centroida ----
    print()
    print("=" * 70)
    print("2) UDALJENOSTI IZMEDU CENTROIDA")
    print("=" * 70)
    cents = {r["faza"]: (r["x_mean"], r["y_mean"]) for r in desc_rows}
    pairs = [("rano", "kasno"), ("rano", "kontinuirano"), ("kasno", "kontinuirano")]
    for a, b in pairs:
        if a in cents and b in cents:
            dx = cents[a][0] - cents[b][0]
            dy = cents[a][1] - cents[b][1]
            d  = np.sqrt(dx*dx + dy*dy) / 1000
            print(f"  d({a:<14}, {b:<14}) = {d:7.2f} km")

    # ---- 3) Moran's I na is_rano (unutar rano+kasno subseta) ----
    print()
    print("=" * 70)
    print("3) MORAN'S I na 'is_rano' (binarni indikator unutar rano+kasno)")
    print("=" * 70)

    sub = pd.concat([rano, kasno], ignore_index=True)
    sub["is_rano"] = sub["samo_rano"].astype(int)
    xy_sub = sub[["x", "y"]].values
    W_sub, neighbors_sub = build_knn_weights(xy_sub, K_NEIGHBORS)

    I, p = morans_i(sub["is_rano"].values, W_sub, N_PERMUTATIONS, RANDOM_SEED)
    print(f"  n = {len(sub)}  (k = {K_NEIGHBORS})")
    print(f"  Moran's I = {I:+.4f},  p = {p:.4g}")
    if p < 0.05:
        if I > 0.10:
            tum = ("rano i kasno SU prostorno klasterirani — razlike u S3 su "
                   "vjerojatno geografski artefakt")
        else:
            tum = "blagi klaster, ali nije osobito jak"
    else:
        tum = "nema znacajne prostorne strukture — rano i kasno su pomijesani"
    print(f"  Tumacenje: {tum}")

    # ---- 4) Join count test ----
    print()
    print("=" * 70)
    print("4) JOIN COUNT TEST (k=8 susjedi)")
    print("=" * 70)

    jc = join_count(sub["is_rano"].values, neighbors_sub,
                    N_PERMUTATIONS, RANDOM_SEED)
    print(f"             Obs       Exp       p")
    print(f"  BB (R-R): {jc['BB_obs']:>4d}   {jc['BB_exp']:>7.1f}   {jc['p_BB']:.4g}")
    print(f"  WW (K-K): {jc['WW_obs']:>4d}   {jc['WW_exp']:>7.1f}   {jc['p_WW']:.4g}")
    print(f"  BW (mij): {jc['BW_obs']:>4d}   {jc['BW_exp']:>7.1f}   {jc['p_BW']:.4g}")
    print()
    if jc["p_BB"] < 0.05 and jc["p_BW"] < 0.05:
        print("  -> RANO i KASNO se medusobno SEGREGIRAJU prostorno")
        print("     (vise R-R i K-K susjedstava nego sto bi bilo slucajno,")
        print("      manje mijesanih)")
    elif jc["p_BB"] >= 0.05 and jc["p_BW"] >= 0.05:
        print("  -> Nema znacajne segregacije — rano i kasno su prostorno pomijesani")
    else:
        print("  -> Djelomicna segregacija (provjeri pojedine p-vrijednosti)")

    # ---- Spremi summary ----
    pd.DataFrame(desc_rows).to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nSpremi u: {OUT_CSV}")


if __name__ == "__main__":
    main()
