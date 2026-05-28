"""
Spatial Cross-Validation za OPTIMALNI_5 model
==============================================

Cilj: provjeriti generalizira li model kroz geografiju.
Random K-fold CV moze biti optimistican zbog prostorne autokorelacije
(susjedne tocke zavrse u istom foldu, model "vidi" lokalnu strukturu).

Directional spatial split:
  Fold 1:  trained = sjever     | test = jug
  Fold 2:  trained = jug        | test = sjever
  Fold 3:  trained = zapad      | test = istok
  Fold 4:  trained = istok      | test = zapad

Split temeljen na medianu X i Y koordinata svih 548 tocaka.
Model se retrenira ispocetka za svaki fold (clone, fit).

Output:
  - spatial_cv_rezultati.csv         (per-fold train/test AUC, n)
  - spatial_cv_rocs.png              (4 ROC krivulje + baseline)
  - spatial_cv_splits.png            (vizualizacija 4 splita)
  - spatial_cv_sazetak.txt           (citljivi sazetak + interpretacija)
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score


ROOT          = r"c:\Users\Martin\Desktop\skripte_za_diplomski\statisticki_testovi"
MASTER        = os.path.join(ROOT, "master_dataset.csv")
MODEL_IN      = os.path.join(ROOT, "10_random_forest", "model_OPTIMALNI_5.joblib")
NEO_COORDS    = os.path.join(ROOT, "01_prostorna_autokorelacija", "neolitik_coords.csv")
RAND_COORDS   = os.path.join(ROOT, "12_spatial_cv", "random_ceste_coords.csv")
OUT_DIR       = os.path.join(ROOT, "12_spatial_cv")

SEED = 42
N_CV = 5


def load_data():
    """Ucitaj master + coords, merge i vrati DataFrame sa svim potrebnim stupcima."""
    df = pd.read_csv(MASTER)
    sub = df[df["tip_sloja"].isin(["neolitik", "nasumicni_ceste"])].copy()

    neo  = pd.read_csv(NEO_COORDS)
    neo["tip_sloja"] = "neolitik"
    rand = pd.read_csv(RAND_COORDS)
    rand["tip_sloja"] = "nasumicni_ceste"
    coords = pd.concat([neo, rand], ignore_index=True)

    merged = sub.merge(
        coords[["fid", "x", "y", "tip_sloja"]],
        left_on=["fid_raw", "tip_sloja"],
        right_on=["fid", "tip_sloja"],
        how="inner",
    )
    return merged


def build_splits(merged):
    """Vrati dict {fold_name: train_mask (bool Series)}."""
    mx = merged["x"].median()
    my = merged["y"].median()
    print(f"  median X = {mx:.0f}   median Y = {my:.0f}")
    splits = {
        "trained_N_test_S": merged["y"] >= my,
        "trained_S_test_N": merged["y"] <  my,
        "trained_E_test_W": merged["x"] >= mx,
        "trained_W_test_E": merged["x"] <  mx,
    }
    return splits, mx, my


def evaluate_fold(pipe_template, X, y, train_mask):
    """Treniraj clone(pipe) na trainu, evaluiraj na testu. Vrati metrike."""
    X_tr = X[train_mask]
    y_tr = y[train_mask]
    X_te = X[~train_mask]
    y_te = y[~train_mask]

    pipe = clone(pipe_template)
    pipe.fit(X_tr, y_tr)

    p_tr = pipe.predict_proba(X_tr)[:, 1]
    p_te = pipe.predict_proba(X_te)[:, 1]

    yhat_te = (p_te >= 0.5).astype(int)

    return {
        "n_train":   int(train_mask.sum()),
        "n_test":    int((~train_mask).sum()),
        "n_train_neo":  int(((train_mask) & (y == 1)).sum()),
        "n_test_neo":   int(((~train_mask) & (y == 1)).sum()),
        "train_auc": float(roc_auc_score(y_tr, p_tr)),
        "test_auc":  float(roc_auc_score(y_te, p_te)),
        "test_f1":   float(f1_score(y_te, yhat_te)),
        "test_acc":  float(accuracy_score(y_te, yhat_te)),
        "p_test":    p_te,
        "y_test":    y_te,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Ucitaj sve ----
    print("=" * 72)
    print("SPATIAL CROSS-VALIDATION za OPTIMALNI_5")
    print("=" * 72)

    print("\n[1] Ucitavam podatke + koordinate...")
    merged = load_data()
    print(f"  n_merged = {len(merged)}  (neolitik = {(merged['tip_sloja']=='neolitik').sum()}, "
          f"random_ceste = {(merged['tip_sloja']=='nasumicni_ceste').sum()})")

    if len(merged) < 500:
        print(f"  UPOZORENJE: ocekivao ~548, dobio {len(merged)}. "
              f"Provjeri da je 00_export_random_coords_qgis.py pokrenut.")

    print(f"\n[2] Ucitavam model: {MODEL_IN}")
    model_dict = joblib.load(MODEL_IN)
    pipe_template = model_dict["pipeline"]
    features = model_dict["features"]
    cols = features["categorical"] + features["numerical"]
    print(f"  set_name = {model_dict['set_name']}")
    print(f"  features = {cols}")

    # Drop NaN u feature kolonama
    n_before = len(merged)
    merged = merged.dropna(subset=cols).reset_index(drop=True)
    print(f"  nakon dropna: {len(merged)} (uklonjenо {n_before - len(merged)})")

    X = merged[cols]
    y = (merged["tip_sloja"] == "neolitik").astype(int).values

    # ---- BASELINE: random 5-fold CV (referenca; trebao bi reproducirati ~0.725) ----
    print(f"\n[3] BASELINE random 5-fold CV (za usporedbu)...")
    cv_base = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=SEED)
    p_base = cross_val_predict(pipe_template, X, y, cv=cv_base,
                                method="predict_proba", n_jobs=-1)[:, 1]
    baseline_auc = roc_auc_score(y, p_base)
    fpr_b, tpr_b, _ = roc_curve(y, p_base)
    print(f"  baseline random 5-fold AUC = {baseline_auc:.4f}")

    # ---- Spatial splits ----
    print(f"\n[4] SPATIAL DIRECTIONAL SPLITS")
    splits, mx, my = build_splits(merged)

    results = {}
    rows = []
    for name, train_mask in splits.items():
        print(f"\n  Fold: {name}")
        res = evaluate_fold(pipe_template, X, y, train_mask)
        results[name] = res
        rows.append({
            "fold":          name,
            "n_train":       res["n_train"],
            "n_test":        res["n_test"],
            "n_train_neo":   res["n_train_neo"],
            "n_test_neo":    res["n_test_neo"],
            "train_auc":     round(res["train_auc"], 4),
            "test_auc":      round(res["test_auc"], 4),
            "test_f1":       round(res["test_f1"],  4),
            "test_acc":      round(res["test_acc"], 4),
            "overfit_gap":   round(res["train_auc"] - res["test_auc"], 4),
        })
        print(f"    n_train = {res['n_train']:>3d} (neo {res['n_train_neo']:>3d}) | "
              f"n_test = {res['n_test']:>3d} (neo {res['n_test_neo']:>3d})")
        print(f"    train AUC = {res['train_auc']:.4f}   test AUC = {res['test_auc']:.4f}   "
              f"gap = {res['train_auc']-res['test_auc']:+.4f}")
        print(f"    test F1   = {res['test_f1']:.4f}   test acc = {res['test_acc']:.4f}")

    res_df = pd.DataFrame(rows)
    res_df.to_csv(os.path.join(OUT_DIR, "spatial_cv_rezultati.csv"),
                  index=False, encoding="utf-8")

    # ---- Statistika ----
    spatial_aucs = res_df["test_auc"].values
    mean_sp = float(np.mean(spatial_aucs))
    std_sp  = float(np.std(spatial_aucs, ddof=1))
    print()
    print("=" * 72)
    print("SAZETAK")
    print("=" * 72)
    print(f"  Baseline random 5-fold AUC : {baseline_auc:.4f}")
    print(f"  Spatial mean AUC (4 fold)  : {mean_sp:.4f} +- {std_sp:.4f}")
    print(f"  Spatial min AUC            : {spatial_aucs.min():.4f}  "
          f"({res_df.loc[res_df['test_auc'].idxmin(), 'fold']})")
    print(f"  Spatial max AUC            : {spatial_aucs.max():.4f}  "
          f"({res_df.loc[res_df['test_auc'].idxmax(), 'fold']})")
    print(f"  AUC pad (baseline - spatial): {baseline_auc - mean_sp:+.4f}")

    # ---- ROC krivulje ----
    print(f"\n[5] Crtam ROC krivulje...")
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = {"trained_N_test_S": "tab:blue",
              "trained_S_test_N": "tab:cyan",
              "trained_E_test_W": "tab:red",
              "trained_W_test_E": "tab:orange"}
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res["y_test"], res["p_test"])
        ax.plot(fpr, tpr, color=colors[name], linewidth=1.8,
                label=f"{name}  AUC={res['test_auc']:.3f}")
    ax.plot(fpr_b, tpr_b, color="black", linewidth=2.0, linestyle="--",
            label=f"BASELINE random 5-fold  AUC={baseline_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", alpha=0.6)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Spatial CV vs Random CV ROC krivulje")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "spatial_cv_rocs.png"), dpi=140)
    plt.close()
    print(f"  -> spatial_cv_rocs.png")

    # ---- Split mape ----
    print(f"\n[6] Crtam split mape...")
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for ax, (name, train_mask) in zip(axes.flatten(), splits.items()):
        # train_mask True = train, False = test
        # Boja kruga: red=neolitik, gray=random.  Filled=train, hollow=test.
        for tip, color in [("neolitik", "tab:red"), ("nasumicni_ceste", "tab:gray")]:
            m_type = merged["tip_sloja"] == tip
            tr = train_mask & m_type
            te = (~train_mask) & m_type
            ax.scatter(merged.loc[tr, "x"], merged.loc[tr, "y"],
                       facecolors=color, edgecolors="black", s=22, alpha=0.6,
                       linewidths=0.4, label=f"train {tip} (n={tr.sum()})")
            ax.scatter(merged.loc[te, "x"], merged.loc[te, "y"],
                       facecolors="none", edgecolors=color, s=40, alpha=0.9,
                       linewidths=1.4, marker="o",
                       label=f"test {tip} (n={te.sum()})")
        # podijeljujuca crta
        if "S" in name.split("_")[-1]:
            ax.axhline(my, color="black", linestyle="--", alpha=0.5)
        else:
            ax.axvline(mx, color="black", linestyle="--", alpha=0.5)
        res = results[name]
        ax.set_title(f"{name}\n"
                     f"train AUC = {res['train_auc']:.3f}  |  test AUC = {res['test_auc']:.3f}")
        ax.set_xlabel("X (EPSG)")
        ax.set_ylabel("Y (EPSG)")
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "spatial_cv_splits.png"), dpi=140)
    plt.close()
    print(f"  -> spatial_cv_splits.png")

    # ---- Citljivi sazetak ----
    interpretation_lines = []
    if baseline_auc - mean_sp > 0.10:
        verdict = "ZNACAJAN PAD"
        comment = ("Random K-fold AUC je znacajno napuhan zbog prostorne "
                   "autokorelacije. Pravi out-of-distribution AUC je nizi.")
    elif baseline_auc - mean_sp > 0.05:
        verdict = "UMJEREN PAD"
        comment = ("Model djelomicno hvata regionalne specificnosti. Generalizacija "
                   "na novu geografiju je slabija nego sto random CV sugerira.")
    else:
        verdict = "STABILNO"
        comment = ("Model dobro generalizira kroz geografiju. Random K-fold AUC "
                   "je vjerodostojan i nije napuhan prostornom autokorelacijom.")
    interpretation_lines.append(f"Spatial CV verdikt: {verdict}")
    interpretation_lines.append(f"  {comment}")

    # Asimetricnost: ako neki smjer puno losiji
    if spatial_aucs.max() - spatial_aucs.min() > 0.10:
        worst = res_df.loc[res_df["test_auc"].idxmin()]
        best  = res_df.loc[res_df["test_auc"].idxmax()]
        interpretation_lines.append(
            f"\nAsimetricnost u smjerovima: razlika max-min = "
            f"{spatial_aucs.max() - spatial_aucs.min():.3f}"
        )
        interpretation_lines.append(
            f"  Najlosije: {worst['fold']}  (test AUC = {worst['test_auc']:.3f})"
        )
        interpretation_lines.append(
            f"  Najbolje : {best['fold']}  (test AUC = {best['test_auc']:.3f})"
        )
        interpretation_lines.append(
            "  -> model uci razlicite obrasce u razlicitim podrucjima. "
            "Mozda postoji regionalna ne-homogenost u preferencijama."
        )

    summary = "\n".join([
        "Spatial Cross-Validation - OPTIMALNI_5 model",
        "=" * 60,
        "",
        f"Model:   {model_dict['set_name']}",
        f"Features:   {cols}",
        f"n_total: {len(merged)}  ({(y==1).sum()} neolitik, {(y==0).sum()} random_ceste)",
        "",
        f"Median X koordinata: {mx:.0f}",
        f"Median Y koordinata: {my:.0f}",
        "",
        "Rezultati po foldu:",
        res_df.to_string(index=False),
        "",
        f"Baseline random 5-fold AUC : {baseline_auc:.4f}",
        f"Spatial mean AUC (4 fold)  : {mean_sp:.4f}  (sd {std_sp:.4f})",
        f"AUC pad                    : {baseline_auc - mean_sp:+.4f}",
        "",
        *interpretation_lines,
    ])
    with open(os.path.join(OUT_DIR, "spatial_cv_sazetak.txt"),
              "w", encoding="utf-8") as fh:
        fh.write(summary)
    print(f"\n[7] -> spatial_cv_sazetak.txt")

    print()
    print("=" * 72)
    for line in interpretation_lines:
        print(line)
    print("=" * 72)
    print(f"Sve spremljeno u: {OUT_DIR}")


if __name__ == "__main__":
    main()
