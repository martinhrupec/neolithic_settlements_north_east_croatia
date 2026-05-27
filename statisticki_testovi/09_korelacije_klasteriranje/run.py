"""
Faza A: Korelacijska matrica + hijerarhijsko klasteriranje
============================================================

Cilj: identificirati grupe varijabli koje mjere istu dimenziju krajobraza
(redundancija) prije provedbe Random Forest analize.

Metoda — mixed-type association strength matrica, vrijednost u [0, 1]:
  - cont x cont   → |Spearman ρ|
  - cat  x cat    → Cramer's V
  - mixed         → correlation ratio (eta = sqrt(SSb / SSt))

Sve metrike imaju usporedivu skalu [0, 1] gdje:
  0   = nema povezanosti
  1   = savrsena povezanost
  ~0.7+ = jaka povezanost (kandidati za istom dimenzijom)

Hijerarhijsko klasteriranje:
  - distance = 1 - association
  - linkage: average (UPGMA) — robusan na mixed-type
  - prikazujemo klastere pri visinama 0.2 (jako stroga), 0.3, 0.5

Podaci za korelaciju: neolitik + nasumicni_ceste (n=548).
To je isti uzorak koji ce ici u Random Forest, pa su korelacije
direktno relevantne za prediktivni model.

Output:
  - matrica_korelacija.csv         (kvadrat. matrica asocijacija)
  - klasteri_dendrogram.png        (vizualizacija)
  - klasteri_pripadnost.csv        (varijabla → cluster ID po pragu)
  - reprezentanti_prijedlozi.csv   (prijedlog reprezentanta po klasteru)
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list, dendrogram
from scipy.spatial.distance import squareform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT    = r"c:\Users\Martin\Desktop\skripte_za_diplomski\statisticki_testovi"
MASTER  = os.path.join(ROOT, "master_dataset.csv")
OUT_DIR = os.path.join(ROOT, "09_korelacije_klasteriranje")

# Varijable
CONT_VARS = [
    "aps_vis",
    "rel_vis_100_250", "rel_vis_100_500", "rel_vis_100_1000",
    "rel_vis_200_500", "rel_vis_200_1000", "rel_vis_500_1000",
    "nagib", "coarse_fragments", "tri",
    "dist_rijeka", "dist_rijeka_korig",
    "gustoca_rijeka_1000", "gustoca_rijeka_2000",
    "strahler",   # ordinal, tretirana kao numericka
]
CAT_VARS = [
    "aspect_cat4", "aspect_ew", "aspect_sn",
    "vtt_r100", "vtt_r250", "vtt_r500", "vtt_r1000",
    "sm_r100",  "sm_r250",  "sm_r500",  "sm_r1000",
]
ALL_VARS = CONT_VARS + CAT_VARS

# Robusne varijable iz univariate analize (S1 + S2 + thinning sensitivity)
ROBUSTNE = {
    "dist_rijeka_korig", "rel_vis_100_250", "strahler",
    "sm_r100", "vtt_r100",
}

# Pragovi za "ovo su iste dimenzije"
THRESHOLDS = [0.2, 0.3, 0.5]


# ---------------------------------------------------------------------------
#  Association metrics
# ---------------------------------------------------------------------------

def cramers_v(s1, s2):
    """Cramer's V (no continuity correction)."""
    tab = pd.crosstab(s1, s2)
    if tab.size == 0 or tab.shape[0] < 2 or tab.shape[1] < 2:
        return 0.0
    chi2 = stats.chi2_contingency(tab.values, correction=False)[0]
    n = tab.values.sum()
    k = min(tab.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 and n > 0 else 0.0


def correlation_ratio(categories, values):
    """
    Eta = sqrt(SSbetween / SStotal). Categorical predictor → continuous outcome.
    Vraca vrijednost u [0, 1].
    """
    df = pd.DataFrame({"cat": categories, "val": values}).dropna()
    if len(df) < 5:
        return 0.0
    grand_mean = df["val"].mean()
    ss_total   = ((df["val"] - grand_mean) ** 2).sum()
    if ss_total == 0:
        return 0.0
    ss_between = 0.0
    for _, sub in df.groupby("cat", observed=True):
        ss_between += len(sub) * (sub["val"].mean() - grand_mean) ** 2
    eta_sq = ss_between / ss_total
    return float(np.sqrt(max(eta_sq, 0.0)))


def association(x, y, x_cat, y_cat):
    """Vraca jacinu asocijacije u [0,1], bez smjera."""
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 10:
        return np.nan
    x_v, y_v = df["x"], df["y"]
    if not x_cat and not y_cat:
        rho, _ = stats.spearmanr(x_v, y_v)
        return abs(float(rho)) if not np.isnan(rho) else 0.0
    elif x_cat and y_cat:
        return cramers_v(x_v, y_v)
    elif x_cat:
        return correlation_ratio(x_v, y_v)
    else:
        return correlation_ratio(y_v, x_v)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(MASTER)
    sub = df[df["tip_sloja"].isin(["neolitik", "nasumicni_ceste"])].copy()
    print(f"n = {len(sub)} (neolitik + nasumicni_ceste)")
    print(f"Broj varijabli: {len(ALL_VARS)}  ({len(CONT_VARS)} cont + {len(CAT_VARS)} cat)\n")

    # Provjera dostupnosti svih kolona
    missing = [v for v in ALL_VARS if v not in sub.columns]
    if missing:
        print(f"  UPOZORENJE: nedostaju kolone: {missing}")
        return

    # Boolean indikator: kategorijska?
    is_cat = {v: (v in CAT_VARS) for v in ALL_VARS}

    # ---- Izracunaj matricu asocijacija ----
    print("Racunam mixed-type matricu asocijacija...")
    n_vars = len(ALL_VARS)
    A = np.zeros((n_vars, n_vars))
    for i, vi in enumerate(ALL_VARS):
        A[i, i] = 1.0
        for j in range(i + 1, n_vars):
            vj = ALL_VARS[j]
            a = association(sub[vi], sub[vj], is_cat[vi], is_cat[vj])
            if np.isnan(a):
                a = 0.0
            A[i, j] = A[j, i] = a

    A_df = pd.DataFrame(A, index=ALL_VARS, columns=ALL_VARS)
    A_df.to_csv(os.path.join(OUT_DIR, "matrica_korelacija.csv"),
                encoding="utf-8", float_format="%.3f")
    print(f"  Matrica spremljena -> matrica_korelacija.csv\n")

    # ---- Najjace povezani parovi ----
    print("=" * 72)
    print("TOP 20 NAJJACIH PAROVA (jedan smjer)")
    print("=" * 72)
    pairs = []
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            pairs.append((ALL_VARS[i], ALL_VARS[j], A[i, j]))
    pairs.sort(key=lambda x: -x[2])
    for vi, vj, a in pairs[:20]:
        marker_i = " *" if vi in ROBUSTNE else "  "
        marker_j = " *" if vj in ROBUSTNE else "  "
        print(f"  {a:.3f}   {vi:<22s}{marker_i} <-> {vj:<22s}{marker_j}")
    print("  ('*' = robusna varijabla iz univariate analize)")

    # ---- Hijerarhijsko klasteriranje ----
    print()
    print("=" * 72)
    print("HIJERARHIJSKO KLASTERIRANJE")
    print("=" * 72)

    # distance = 1 - association  (oba u [0, 1])
    D = 1.0 - A
    np.fill_diagonal(D, 0.0)
    D_condensed = squareform(D, checks=False)
    Z = linkage(D_condensed, method="average")

    # Klasteri pri vise pragova
    cluster_table = {"varijabla": ALL_VARS}
    for thr in THRESHOLDS:
        cids = fcluster(Z, t=thr, criterion="distance")
        cluster_table[f"klaster_t{thr}"] = cids

    ct = pd.DataFrame(cluster_table)
    ct.to_csv(os.path.join(OUT_DIR, "klasteri_pripadnost.csv"),
              index=False, encoding="utf-8")

    # Ispis klastera za svaki prag
    for thr in THRESHOLDS:
        col = f"klaster_t{thr}"
        groups = ct.groupby(col)["varijabla"].apply(list).to_dict()
        print(f"\n  Prag = {thr}  (distance 1 - assoc):  {len(groups)} klastera")
        for cid, members in sorted(groups.items()):
            if len(members) == 1:
                marker = "*" if members[0] in ROBUSTNE else " "
                print(f"    [{cid:>2}]  {marker} {members[0]}  (samostalna)")
            else:
                print(f"    [{cid:>2}]  ({len(members)} varijabli)")
                for m in members:
                    marker = "*" if m in ROBUSTNE else " "
                    print(f"           {marker} {m}")

    # ---- Dendrogram (PNG) ----
    print()
    print("=" * 72)
    print("DENDROGRAM")
    print("=" * 72)

    fig, ax = plt.subplots(figsize=(10, max(6, n_vars * 0.3)))
    dendrogram(
        Z,
        labels=ALL_VARS,
        orientation="left",
        leaf_font_size=10,
        color_threshold=0.5,
        ax=ax,
    )
    ax.set_xlabel("Distance (1 - association strength)")
    ax.set_title("Hijerarhijsko klasteriranje varijabli (UPGMA, mixed-type)")
    # vertikalna linija na pragovima
    for thr, c in zip(THRESHOLDS, ["red", "orange", "gray"]):
        ax.axvline(thr, linestyle="--", color=c, alpha=0.5,
                   label=f"prag {thr}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "klasteri_dendrogram.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"  Spremljeno -> {out_png}")

    # ---- Heatmapa korelacija ----
    fig, ax = plt.subplots(figsize=(11, 10))
    # Reorder by clustering for readability
    order = leaves_list(Z)
    A_re  = A[np.ix_(order, order)]
    labs  = [ALL_VARS[i] for i in order]
    im = ax.imshow(A_re, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n_vars)); ax.set_yticks(range(n_vars))
    ax.set_xticklabels(labs, rotation=90, fontsize=8)
    ax.set_yticklabels(labs, fontsize=8)
    plt.colorbar(im, ax=ax, label="Association strength")
    ax.set_title("Mixed-type association matrica (reordered po klasteriranju)")
    plt.tight_layout()
    out_hm = os.path.join(OUT_DIR, "heatmapa_korelacija.png")
    plt.savefig(out_hm, dpi=150)
    plt.close()
    print(f"  Heatmapa  -> {out_hm}")

    # ---- Prijedlog reprezentanata (prag 0.3 — srednji) ----
    print()
    print("=" * 72)
    print("PRIJEDLOG REPREZENTANATA  (prag 0.3 — srednje stroga grupacija)")
    print("=" * 72)
    chosen_thr = 0.3
    col = f"klaster_t{chosen_thr}"
    suggestions = []
    for cid, members in ct.groupby(col)["varijabla"].apply(list).items():
        # Heuristika: ako klaster ima robusne, izaberi prvu robusnu; inace prvu
        robust_in = [m for m in members if m in ROBUSTNE]
        if robust_in:
            rep = robust_in[0]   # tu kasnije korisnik moze rucno odabrati
            reason = f"robusna (univariate {robust_in[0]})"
        else:
            rep = members[0]
            reason = "samostalna" if len(members) == 1 else "default (prva u klasteru)"
        suggestions.append({
            "klaster_id":     cid,
            "n_varijabli":    len(members),
            "clanovi":        ", ".join(members),
            "predlozeni":     rep,
            "razlog":         reason,
        })
        print(f"\n  Klaster {cid}: ({len(members)} varijabli) -> {rep}  [{reason}]")
        for m in members:
            mark = " <- predlozeni" if m == rep else \
                   " (robusna)"      if m in ROBUSTNE else ""
            print(f"       {m}{mark}")

    pd.DataFrame(suggestions).to_csv(
        os.path.join(OUT_DIR, "reprezentanti_prijedlozi.csv"),
        index=False, encoding="utf-8")

    print()
    print(f"Reprezentanti spremljeni -> reprezentanti_prijedlozi.csv")
    print()
    print("Sljedeci korak:")
    print(f"  1. Otvori klasteri_dendrogram.png i heatmapa_korelacija.png")
    print(f"  2. Pregledaj reprezentanti_prijedlozi.csv i rucno popravi izbor")
    print(f"     gdje treba (npr. izaberi rel_vis_100_250 umjesto _100_500)")
    print(f"  3. Sa izabranim reprezentantima nastavi na Random Forest")


if __name__ == "__main__":
    main()
