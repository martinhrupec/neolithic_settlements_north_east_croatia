"""
Spatial Coherence Check za OPTIMALNI_5 model
=============================================

Cilj: prije nego apliciramo model na milijun piksela u heatmapi,
provjeravamo radi li smisleno na trening podacima.

Provjere:
  1. Apliciraj model OOF na svih 548 tocaka, izvuci predict_proba
  2. Distribucija predikcija po grupama (boxplot, histogram)
  3. Spatial scatter neolitik tocaka obojeno po predikciji
  4. Moran's I na predicted probabilities — jesu li predikcije
     prostorno koherentne (susjedi imaju slicne vjerojatnosti)?
  5. Confusion matrix pri razlicitim pragovima (0.3, 0.5, 0.7)
  6. Identifikacija pogresnih klasifikacija — gdje su prostorno?

Output:
  - probabilities_distribution.png
  - spatial_predictions_neolitik.png
  - misclassifications_table.csv
  - spatial_coherence_metrics.txt
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
)


ROOT       = r"c:\Users\Martin\Desktop\skripte_za_diplomski\statisticki_testovi"
MASTER     = os.path.join(ROOT, "master_dataset.csv")
MODEL_IN   = os.path.join(ROOT, "10_random_forest", "model_OPTIMALNI_5.joblib")
COORDS_CSV = os.path.join(ROOT, "01_prostorna_autokorelacija", "neolitik_coords.csv")
OUT_DIR    = os.path.join(ROOT, "11_calibration_validation")

SEED = 42
N_CV = 5
K_NN = 8


def build_knn_weights(coords, k):
    n    = len(coords)
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)
    W = np.zeros((n, n))
    for i in range(n):
        for j in idx[i, 1:]:
            W[i, j] = 1.0
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return W / rs


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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Ucitaj model, podatke, koordinate ----
    model_dict = joblib.load(MODEL_IN)
    pipe       = model_dict["pipeline"]
    features   = model_dict["features"]
    cols       = features["categorical"] + features["numerical"]

    df = pd.read_csv(MASTER)
    sub = df[df["tip_sloja"].isin(["neolitik", "nasumicni_ceste"])].copy()
    sub = sub.dropna(subset=cols).reset_index(drop=True)
    X = sub[cols]
    y = (sub["tip_sloja"] == "neolitik").astype(int).values
    print(f"n_total = {len(sub)}\n")

    # ---- OOF predikcije ----
    cv = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=SEED)
    print("Generiram OOF predikcije...")
    y_prob = cross_val_predict(pipe, X, y, cv=cv,
                                method="predict_proba", n_jobs=-1)[:, 1]
    sub["pred_prob"] = y_prob

    # ---- Distribucija po grupama ----
    neo  = sub[sub["tip_sloja"] == "neolitik"]
    rand = sub[sub["tip_sloja"] == "nasumicni_ceste"]

    print()
    print("=" * 72)
    print("DISTRIBUCIJA PREDIKCIJA PO GRUPAMA")
    print("=" * 72)
    print(f"  Neolitik:        mean={neo['pred_prob'].mean():.3f}  "
          f"median={neo['pred_prob'].median():.3f}  "
          f"sd={neo['pred_prob'].std():.3f}")
    print(f"  Random_ceste:    mean={rand['pred_prob'].mean():.3f}  "
          f"median={rand['pred_prob'].median():.3f}  "
          f"sd={rand['pred_prob'].std():.3f}")

    # Histogram + boxplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.hist(rand["pred_prob"], bins=25, alpha=0.6, color="tab:gray",
             label=f"random_ceste (n={len(rand)})")
    ax1.hist(neo["pred_prob"], bins=25, alpha=0.6, color="tab:red",
             label=f"neolitik (n={len(neo)})")
    ax1.axvline(0.5, color="black", linestyle="--", alpha=0.5, label="prag 0.5")
    ax1.set_xlabel("Predvidena vjerojatnost")
    ax1.set_ylabel("Frekvencija")
    ax1.set_title("Distribucija OOF predikcija")
    ax1.legend()

    ax2.boxplot([rand["pred_prob"], neo["pred_prob"]],
                labels=["random_ceste", "neolitik"])
    ax2.axhline(0.5, color="red", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Predvidena vjerojatnost")
    ax2.set_title("Boxplot OOF predikcija po grupi")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "probabilities_distribution.png"), dpi=140)
    plt.close()
    print(f"\nSpremljeno: probabilities_distribution.png")

    # ---- Confusion matrix pri vise pragova ----
    print()
    print("=" * 72)
    print("PERFORMANSA PRI RAZLICITIM PRAGOVIMA")
    print("=" * 72)
    print(f"{'prag':<6}{'accuracy':>10}{'precision':>11}{'recall':>9}{'F1':>8}"
          f"{'TP':>6}{'FP':>6}{'TN':>6}{'FN':>6}")
    rows = []
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_hat = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
        acc = accuracy_score(y, y_hat)
        prec = precision_score(y, y_hat) if (tp + fp) > 0 else 0
        rec = recall_score(y, y_hat)
        f1 = f1_score(y, y_hat)
        print(f"{thr:<6.1f}{acc:>10.3f}{prec:>11.3f}{rec:>9.3f}{f1:>8.3f}"
              f"{tp:>6d}{fp:>6d}{tn:>6d}{fn:>6d}")
        rows.append({"prag": thr, "accuracy": acc, "precision": prec,
                     "recall": rec, "f1": f1,
                     "TP": tp, "FP": fp, "TN": tn, "FN": fn})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "performance_po_pragu.csv"),
                              index=False, encoding="utf-8")

    auc = roc_auc_score(y, y_prob)
    print(f"\nOOF AUC = {auc:.4f}")

    # ---- Spatial scatter za neolitik ----
    print()
    print("=" * 72)
    print("SPATIAL ANALIZA NEOLITIK PREDIKCIJA")
    print("=" * 72)

    coords = pd.read_csv(COORDS_CSV)
    neo_geo = neo.merge(coords, left_on="fid_raw", right_on="fid", how="inner")
    print(f"  Spojeno s koordinatama: {len(neo_geo)} / {len(neo)} neolitik tocaka")

    fig, ax = plt.subplots(figsize=(11, 8))
    sc = ax.scatter(neo_geo["x"], neo_geo["y"], c=neo_geo["pred_prob"],
                    cmap="RdYlGn", vmin=0, vmax=1, s=35, edgecolors="black",
                    linewidths=0.4)
    plt.colorbar(sc, ax=ax, label="Predicted probability")
    ax.set_xlabel("X (EPSG)")
    ax.set_ylabel("Y (EPSG)")
    ax.set_title(f"Spatial distribution OOF predikcija za neolitik (n={len(neo_geo)})\n"
                 f"Crveno = nisko (model 'nije siguran') | "
                 f"Zeleno = visoko (model dobro pogada)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "spatial_predictions_neolitik.png"), dpi=140)
    plt.close()
    print(f"  Spremljeno: spatial_predictions_neolitik.png")

    # ---- Moran's I na predicted probabilities ----
    xy = neo_geo[["x", "y"]].values
    W  = build_knn_weights(xy, K_NN)
    I, p = morans_i(neo_geo["pred_prob"].values, W, n_perm=999, seed=SEED)
    print(f"\n  Moran's I na pred_prob (k={K_NN}):  I = {I:.4f},  p = {p:.4f}")
    if p < 0.05 and I > 0.10:
        print(f"     -> predikcije su PROSTORNO KOHERENTNE")
        print(f"        susjedne tocke imaju slicne vjerojatnosti (dobro)")
    elif p < 0.05 and I < -0.10:
        print(f"     -> predikcije su RASPRSENE  (neuobicajeno)")
    else:
        print(f"     -> nema znacajne prostorne strukture")
        print(f"        moguce da model ne hvata regionalne razlike")

    # ---- Misklasifikacije ----
    print()
    print("=" * 72)
    print("MISKLASIFIKACIJE (pri pragu 0.5)")
    print("=" * 72)

    sub["pred_class"] = (sub["pred_prob"] >= 0.5).astype(int)
    sub["misclassified"] = (sub["pred_class"] != y).astype(int)
    miscls = sub[sub["misclassified"] == 1].copy()
    print(f"  Ukupno krivo klasificiranih: {len(miscls)} / {len(sub)} "
          f"({100*len(miscls)/len(sub):.1f}%)")
    print(f"  - FN (neolitik klasificiran kao random): "
          f"{((sub['tip_sloja']=='neolitik') & (sub['pred_class']==0)).sum()}")
    print(f"  - FP (random klasificiran kao neolitik): "
          f"{((sub['tip_sloja']=='nasumicni_ceste') & (sub['pred_class']==1)).sum()}")

    miscls[["fid_raw","uid","tip_sloja","pred_prob"] + cols].to_csv(
        os.path.join(OUT_DIR, "misclassifications_table.csv"),
        index=False, encoding="utf-8")

    # Visualiziraj FN spatially
    fn_neo = neo_geo[neo_geo["pred_prob"] < 0.5]
    tp_neo = neo_geo[neo_geo["pred_prob"] >= 0.5]
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(tp_neo["x"], tp_neo["y"], c="green", s=30,
               label=f"TP (neolitik dobro pogoden, n={len(tp_neo)})",
               alpha=0.7, edgecolors="black", linewidths=0.3)
    ax.scatter(fn_neo["x"], fn_neo["y"], c="red", s=50,
               label=f"FN (propustena neolitik nalazista, n={len(fn_neo)})",
               alpha=0.85, edgecolors="black", linewidths=0.5, marker="X")
    ax.set_xlabel("X (EPSG)")
    ax.set_ylabel("Y (EPSG)")
    ax.set_title(f"Pravilno vs propusteno (FN) — prag 0.5")
    ax.set_aspect("equal")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "spatial_misclassifications.png"), dpi=140)
    plt.close()
    print(f"  Spremljeno: spatial_misclassifications.png")

    # ---- Sazetak ----
    with open(os.path.join(OUT_DIR, "spatial_coherence_metrics.txt"),
              "w", encoding="utf-8") as fh:
        fh.write("Spatial Coherence Check — OPTIMALNI_5 model\n")
        fh.write("=" * 60 + "\n\n")
        fh.write(f"n_total = {len(sub)}\n")
        fh.write(f"OOF AUC = {auc:.4f}\n\n")
        fh.write(f"Neolitik   mean pred_prob = {neo['pred_prob'].mean():.4f}\n")
        fh.write(f"Random_C   mean pred_prob = {rand['pred_prob'].mean():.4f}\n")
        fh.write(f"Razlika u srednjim         = "
                 f"{neo['pred_prob'].mean() - rand['pred_prob'].mean():.4f}\n\n")
        fh.write(f"Moran's I na pred_prob   = {I:.4f}  (p = {p:.4f})\n")
        fh.write(f"Misklasifikacije (prag 0.5) = {len(miscls)} / {len(sub)}\n")

    print()
    print("=" * 72)
    print("SAZETAK")
    print("=" * 72)
    print(f"  Model OOF AUC               = {auc:.4f}")
    print(f"  Mean prob (neolitik)         = {neo['pred_prob'].mean():.4f}")
    print(f"  Mean prob (random_ceste)     = {rand['pred_prob'].mean():.4f}")
    print(f"  Razlika                      = "
          f"{neo['pred_prob'].mean() - rand['pred_prob'].mean():.4f}")
    print(f"  Moran's I na pred_prob       = {I:.4f}  (p = {p:.4f})")
    print()
    print("Sve spremljeno u:", OUT_DIR)


if __name__ == "__main__":
    main()
