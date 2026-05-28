"""
SHAP Pattern Clustering - identifikacija tipova atribucije
============================================================

Cilj: formalno provjeriti postoje li grupe cluster-a sa slicnim SHAP
profilom. Vizualnom inspekcijom u 02_shap_attribution se cinilo da
postoje dva tipa (Cambisol-driven vs rel_vis-driven). Ova skripta
provjerava taj dojam k-means klasteriranjem u 5-dim SHAP prostoru.

Pipeline:
  1. Ucitaj shap_per_cluster.csv
  2. K-means za k ∈ {2, 3, 4, 5} na standardiziranim SHAP vrijednostima
  3. Silhouette score -> optimalni k
  4. PCA na 2D za vizualizaciju
  5. Centroidi u original SHAP prostoru (interpretabilno)
  6. Spatial mapa: gdje su tipovi geografski

Output:
  - shap_attribution_types.csv      - svaki cluster + tip
  - shap_types_centroids.csv        - mean SHAP per tip
  - shap_silhouette_per_k.png       - opravdanje za izabrani k
  - shap_pca_scatter.png            - 2D PCA, obojeno po tipu
  - shap_types_profiles.png         - bar chart-i mean SHAP per tip
  - shap_types_spatial.png          - geografska distribucija tipova
  - shap_types_summary.txt          - per-tip interpretacija
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT        = r"c:\Users\Martin\Desktop\skripte_za_diplomski"
CLUSTER_DIR = os.path.join(ROOT, "statisticki_testovi", "13_cluster_zones")
SHAP_CSV    = os.path.join(CLUSTER_DIR, "shap_per_cluster.csv")
OUT_DIR     = CLUSTER_DIR

SHAP_COLS = [
    "shap_vtt_r100",
    "shap_rel_vis_100_250",
    "shap_dist_rijeka_korig",
    "shap_strahler",
    "shap_aps_vis",
]
PRETTY = {
    "shap_vtt_r100":          "vtt_r100",
    "shap_rel_vis_100_250":   "rel_vis_100_250",
    "shap_dist_rijeka_korig": "dist_rijeka_korig",
    "shap_strahler":          "strahler",
    "shap_aps_vis":           "aps_vis",
}

K_VALUES = [2, 3, 4, 5]
SEED = 42


def main():
    print("=" * 72)
    print("SHAP PATTERN CLUSTERING")
    print("=" * 72)

    # ----- Ucitaj -----
    print(f"\n[1] Ucitavam: {SHAP_CSV}")
    df = pd.read_csv(SHAP_CSV)
    df = df.dropna(subset=SHAP_COLS).reset_index(drop=True)
    print(f"    n = {len(df)}")
    X = df[SHAP_COLS].values

    # ----- Standardizacija -----
    print(f"\n[2] Standardiziram SHAP vrijednosti za k-means...")
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # ----- Silhouette analiza -----
    print(f"\n[3] Silhouette analiza za k ∈ {K_VALUES}...")
    sil_scores = {}
    models = {}
    for k in K_VALUES:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=20)
        labels = km.fit_predict(X_std)
        sil = float(silhouette_score(X_std, labels))
        sil_scores[k] = sil
        models[k] = km
        print(f"    k={k}: silhouette = {sil:.4f}")

    best_k = max(sil_scores, key=sil_scores.get)
    print(f"\n    Optimalni k = {best_k}  (silhouette = {sil_scores[best_k]:.4f})")

    if sil_scores[best_k] < 0.25:
        print(f"    UPOZORENJE: silhouette < 0.25  - grupacije su slabe")
    elif sil_scores[best_k] < 0.5:
        print(f"    NAPOMENA: silhouette 0.25-0.50  - umjereno jake grupacije")
    else:
        print(f"    DOBRO: silhouette >= 0.50  - jasno definirane grupacije")

    # Silhouette plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ks = list(sil_scores.keys())
    ss = list(sil_scores.values())
    colors_sil = ["tab:green" if k == best_k else "steelblue" for k in ks]
    ax.bar(ks, ss, color=colors_sil)
    for k, s in zip(ks, ss):
        ax.text(k, s + 0.005, f"{s:.3f}", ha="center", fontsize=9)
    ax.set_xlabel("k (broj klastera)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette score po k\n(zeleni = odabrani)")
    ax.set_xticks(ks)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_silhouette_per_k.png"), dpi=140)
    plt.close()
    print(f"    -> shap_silhouette_per_k.png")

    # Best model labels
    df["attr_type"] = models[best_k].labels_

    # ----- PCA na 2D -----
    print(f"\n[4] PCA na 2D za vizualizaciju...")
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_std)
    df["pc1"] = X_pca[:, 0]
    df["pc2"] = X_pca[:, 1]
    expl = pca.explained_variance_ratio_
    print(f"    explained variance:  PC1 = {expl[0]*100:.1f}%,  PC2 = {expl[1]*100:.1f}%  "
          f"(zajedno {sum(expl)*100:.1f}%)")

    # ----- Centroidi -----
    centroids_std = models[best_k].cluster_centers_
    centroids = scaler.inverse_transform(centroids_std)
    pca_centroids = pca.transform(centroids_std)

    centroid_df = pd.DataFrame(centroids, columns=SHAP_COLS)
    centroid_df.insert(0, "attr_type", range(best_k))
    centroid_df.insert(1, "n_clusters",
                       [int((df["attr_type"] == t).sum()) for t in range(best_k)])
    centroid_df.to_csv(os.path.join(OUT_DIR, "shap_types_centroids.csv"),
                       index=False, encoding="utf-8")

    # ----- Save attribution types CSV -----
    cols_save = ["source", "cluster_id", "threshold", "centroid_x", "centroid_y",
                 "area_km2", "prob_max", "prob_mean", "dist_to_known_m",
                 "n_known_inside", "val_vtt_r100", "val_rel_vis",
                 "val_dist_rijeka", "val_strahler", "val_aps_vis",
                 "attr_type", "pc1", "pc2"] + SHAP_COLS
    cols_save = [c for c in cols_save if c in df.columns]
    df[cols_save].to_csv(os.path.join(OUT_DIR, "shap_attribution_types.csv"),
                         index=False, encoding="utf-8")

    # ----- Plot 1: PCA scatter -----
    print(f"\n[5] PCA scatter plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.arange(best_k))
    for t in range(best_k):
        m = df["attr_type"] == t
        ax.scatter(df.loc[m, "pc1"], df.loc[m, "pc2"],
                   c=[colors[t]], label=f"Tip {t} (n={int(m.sum())})",
                   s=15, alpha=0.55, edgecolors="white", linewidths=0.3)
    for t in range(best_k):
        ax.scatter(pca_centroids[t, 0], pca_centroids[t, 1],
                   marker="X", s=300, c=[colors[t]], edgecolors="black",
                   linewidths=2, zorder=10)
    ax.set_xlabel(f"PC1 ({expl[0]*100:.1f}% varijance)")
    ax.set_ylabel(f"PC2 ({expl[1]*100:.1f}% varijance)")
    ax.set_title(f"SHAP atribucija u PCA prostoru (k={best_k})\n"
                 f"X = centroidi tipova")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_pca_scatter.png"), dpi=140)
    plt.close()
    print(f"    -> shap_pca_scatter.png")

    # ----- Plot 2: Centroid bar chart per type -----
    print(f"\n[6] Centroid profili po tipu...")
    fig, axes = plt.subplots(1, best_k, figsize=(4.5 * best_k, 5), sharex=True)
    if best_k == 1:
        axes = [axes]
    pretty_names = [PRETTY[c] for c in SHAP_COLS]
    for t in range(best_k):
        ax = axes[t]
        vals = centroids[t]
        bar_colors = ["#1f77b4" if v >= 0 else "#d62728" for v in vals]
        ax.barh(pretty_names, vals, color=bar_colors)
        ax.axvline(0, color="black", linewidth=0.6)
        n_t = int((df["attr_type"] == t).sum())
        pct = 100 * n_t / len(df)
        ax.set_title(f"Tip {t}  (n={n_t}, {pct:.1f}%)", color=colors[t])
        ax.set_xlabel("Mean SHAP (signed)")
        ax.grid(alpha=0.3, axis="x")
    plt.suptitle(f"Karakteristični SHAP profili po tipu  (k={best_k})", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_types_profiles.png"), dpi=140)
    plt.close()
    print(f"    -> shap_types_profiles.png")

    # ----- Plot 3: Spatial distribution -----
    print(f"\n[7] Prostorna distribucija tipova...")
    fig, ax = plt.subplots(figsize=(13, 9))
    for t in range(best_k):
        m = df["attr_type"] == t
        ax.scatter(df.loc[m, "centroid_x"], df.loc[m, "centroid_y"],
                   c=[colors[t]], label=f"Tip {t} (n={int(m.sum())})",
                   s=20, alpha=0.6, edgecolors="black", linewidths=0.3)
    ax.set_xlabel("X (EPSG:3765)")
    ax.set_ylabel("Y (EPSG:3765)")
    ax.set_title(f"Prostorna distribucija tipova SHAP atribucije (k={best_k})")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_types_spatial.png"), dpi=140)
    plt.close()
    print(f"    -> shap_types_spatial.png")

    # ----- Text summary -----
    print(f"\n[8] Generiram interpretaciju...")
    lines = [
        "SHAP Pattern Clustering Summary",
        "=" * 60,
        "",
        f"n_clustera_total = {len(df)}",
        f"Optimalni k = {best_k}  (silhouette = {sil_scores[best_k]:.4f})",
        "",
        "Silhouette scores per k:",
    ]
    for k, s in sil_scores.items():
        marker = "   <- ODABRAN" if k == best_k else ""
        lines.append(f"  k={k}: {s:.4f}{marker}")

    if sil_scores[best_k] < 0.25:
        lines.append("\n  INTERPRETACIJA: silhouette < 0.25 -> grupacije su slabe,")
        lines.append("  podaci su vise gradient nego diskretni klasteri.")
    elif sil_scores[best_k] < 0.5:
        lines.append("\n  INTERPRETACIJA: silhouette 0.25-0.50 -> umjerene grupacije,")
        lines.append("  postoji razlucivost ali s nesto preklapanja.")
    else:
        lines.append("\n  INTERPRETACIJA: silhouette >= 0.50 -> jasne grupacije,")
        lines.append("  postoji jasna razdvojenost u SHAP prostoru.")
    lines.append("")

    # Per-type description
    for t in range(best_k):
        sub = df[df["attr_type"] == t]
        n_t = len(sub)
        lines.append("=" * 60)
        lines.append(f"TIP {t}    n = {n_t}  ({100*n_t/len(df):.1f}% svih clustera)")
        lines.append("=" * 60)

        lines.append("\nProsjecni SHAP doprinosi:")
        for col in SHAP_COLS:
            v = sub[col].mean()
            sign = "+" if v >= 0 else ""
            lines.append(f"  {PRETTY[col]:<22s}: {sign}{v:.4f}")

        # Dominantni feature
        mean_abs = sub[SHAP_COLS].abs().mean()
        dom = mean_abs.idxmax()
        lines.append(f"\n  DOMINANTNI feature: {PRETTY[dom]}  (|mean|={mean_abs[dom]:.4f})")

        # Karakteristike clustera u ovom tipu
        lines.append(f"\nProsjecne karakteristike clustera u tipu {t}:")
        lines.append(f"  area_km2        : {sub['area_km2'].mean():.3f}")
        lines.append(f"  prob_max        : {sub['prob_max'].mean():.3f}")
        lines.append(f"  dist_to_known_m : {sub['dist_to_known_m'].mean():.0f}")
        lines.append(f"  n_known_inside  : {sub['n_known_inside'].mean():.2f}")

        # Feature vrijednosti unutar clustera (mean +/- std za kontinuirane)
        lines.append(f"\nProsjecne feature vrijednosti unutar clustera tipa {t}:")
        if "val_rel_vis" in sub.columns:
            lines.append(f"  rel_vis_100_250  : "
                         f"{sub['val_rel_vis'].mean():+.2f}  +/-  "
                         f"{sub['val_rel_vis'].std():.2f}   m")
        if "val_dist_rijeka" in sub.columns:
            lines.append(f"  dist_rijeka_korig: "
                         f"{sub['val_dist_rijeka'].mean():.0f}  +/-  "
                         f"{sub['val_dist_rijeka'].std():.0f}   m")
        if "val_aps_vis" in sub.columns:
            lines.append(f"  aps_vis          : "
                         f"{sub['val_aps_vis'].mean():.0f}  +/-  "
                         f"{sub['val_aps_vis'].std():.0f}   m")

        # Strahler distribucija (diskretna ordinalna)
        if "val_strahler" in sub.columns:
            sd = sub["val_strahler"].value_counts().sort_index()
            sd_str = ",  ".join(f"red {int(s)}: {c} ({100*c/n_t:.0f}%)"
                                for s, c in sd.items())
            lines.append(f"  strahler distr.  : {sd_str}")

        # vtt distribucija (kategoricka)
        if "val_vtt_r100" in sub.columns:
            lines.append(f"\nNajcesce vtt_r100 vrijednosti (top 5):")
            for vtt, cnt in sub["val_vtt_r100"].value_counts().head(5).items():
                pct = 100 * cnt / n_t
                lines.append(f"  {str(vtt):<15s}: {cnt:>4d}  ({pct:5.1f}%)")

        # Source distribucija
        if "source" in sub.columns:
            lines.append(f"\nSource distribucija (kojem source-u pripadaju):")
            for src, cnt in sub["source"].value_counts().items():
                pct = 100 * cnt / n_t
                lines.append(f"  {src:<25s}: {cnt:>4d}  ({pct:5.1f}%)")

        lines.append("")

    summary_path = os.path.join(OUT_DIR, "shap_types_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"    -> shap_types_summary.txt")

    # Konzolni ispis
    print("\n" + "\n".join(lines))
    print(f"\nSve spremljeno u: {OUT_DIR}")


if __name__ == "__main__":
    main()
