"""
Probability Calibration za OPTIMALNI_5 model
=============================================

Cilj: pretvoriti RF "predict_proba" izlaze u stvarne vjerojatnosti.

Problem: Random Forest probabilities su INTERNE — ako model kaze 0.7,
to ne znaci 70% sigurnosti. Bayesian kalibracija mapira interne
vrijednosti u prave vjerojatnosti.

Metode:
  - Isotonic regression (neparametrijska, dobra za >1000 uzoraka)
  - Platt scaling (sigmoid, dobra za manje uzoraka)

Metrike:
  - Brier score (manje = bolje, savrseni = 0)
  - Log loss (manje = bolje)
  - Calibration curve (kako blizu y=x dijagonala = bolja kalibracija)

Output:
  - calibration_curves.png
  - brier_log_loss_usporedba.csv
  - model_OPTIMALNI_5_calibrated.joblib (najbolja varijanta)
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


ROOT      = r"c:\Users\Martin\Desktop\skripte_za_diplomski\statisticki_testovi"
MASTER    = os.path.join(ROOT, "master_dataset.csv")
MODEL_IN  = os.path.join(ROOT, "10_random_forest", "model_OPTIMALNI_5.joblib")
OUT_DIR   = os.path.join(ROOT, "11_calibration_validation")

SEED = 42
N_CV = 5


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Ucitaj model i podatke ----
    model_dict = joblib.load(MODEL_IN)
    pipe_orig  = model_dict["pipeline"]
    features   = model_dict["features"]
    cols       = features["categorical"] + features["numerical"]

    print(f"Ucitan model: {model_dict['set_name']}")
    print(f"Featuri: {cols}\n")

    df = pd.read_csv(MASTER)
    sub = df[df["tip_sloja"].isin(["neolitik", "nasumicni_ceste"])].copy()
    sub = sub.dropna(subset=cols)
    X = sub[cols]
    y = (sub["tip_sloja"] == "neolitik").astype(int).values
    print(f"n = {len(sub)}  ({y.sum()} neolitik, {(1-y).sum()} random_ceste)\n")

    cv = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=SEED)

    # ---- Predikcije: original (nekalibriran) ----
    print("Generiram out-of-fold predikcije...")
    y_pred_uncal = cross_val_predict(pipe_orig, X, y, cv=cv,
                                      method="predict_proba", n_jobs=-1)[:, 1]
    print(f"  Original RF:      Brier={brier_score_loss(y, y_pred_uncal):.4f}  "
          f"LogLoss={log_loss(y, y_pred_uncal):.4f}  "
          f"AUC={roc_auc_score(y, y_pred_uncal):.4f}")

    # ---- Kalibracija: isotonic ----
    cal_iso = CalibratedClassifierCV(pipe_orig, method="isotonic", cv=cv)
    y_pred_iso = cross_val_predict(cal_iso, X, y, cv=cv,
                                    method="predict_proba", n_jobs=-1)[:, 1]
    print(f"  Isotonic kalib.:  Brier={brier_score_loss(y, y_pred_iso):.4f}  "
          f"LogLoss={log_loss(y, y_pred_iso):.4f}  "
          f"AUC={roc_auc_score(y, y_pred_iso):.4f}")

    # ---- Kalibracija: sigmoid (Platt) ----
    cal_sig = CalibratedClassifierCV(pipe_orig, method="sigmoid", cv=cv)
    y_pred_sig = cross_val_predict(cal_sig, X, y, cv=cv,
                                    method="predict_proba", n_jobs=-1)[:, 1]
    print(f"  Sigmoid kalib.:   Brier={brier_score_loss(y, y_pred_sig):.4f}  "
          f"LogLoss={log_loss(y, y_pred_sig):.4f}  "
          f"AUC={roc_auc_score(y, y_pred_sig):.4f}")

    # ---- Tablica usporedbe ----
    cmp = pd.DataFrame([
        {"metoda": "Original RF",
         "brier": brier_score_loss(y, y_pred_uncal),
         "log_loss": log_loss(y, y_pred_uncal),
         "auc": roc_auc_score(y, y_pred_uncal)},
        {"metoda": "Isotonic",
         "brier": brier_score_loss(y, y_pred_iso),
         "log_loss": log_loss(y, y_pred_iso),
         "auc": roc_auc_score(y, y_pred_iso)},
        {"metoda": "Sigmoid (Platt)",
         "brier": brier_score_loss(y, y_pred_sig),
         "log_loss": log_loss(y, y_pred_sig),
         "auc": roc_auc_score(y, y_pred_sig)},
    ])
    cmp.to_csv(os.path.join(OUT_DIR, "brier_log_loss_usporedba.csv"),
               index=False, encoding="utf-8")

    # ---- Calibration plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    for name, y_pred, color in [
        ("Original RF",  y_pred_uncal, "tab:gray"),
        ("Isotonic",     y_pred_iso,   "tab:blue"),
        ("Sigmoid",      y_pred_sig,   "tab:orange"),
    ]:
        prob_true, prob_pred = calibration_curve(y, y_pred, n_bins=10, strategy="quantile")
        ax1.plot(prob_pred, prob_true, marker="o", color=color, label=name, linewidth=2)
        ax2.hist(y_pred, bins=20, alpha=0.5, color=color, label=name,
                 density=True, histtype="step", linewidth=2)

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Savrseno kalibrirano")
    ax1.set_xlabel("Predvidena vjerojatnost")
    ax1.set_ylabel("Stvarna frekvencija (out-of-fold)")
    ax1.set_title("Calibration curve")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Predvidena vjerojatnost")
    ax2.set_ylabel("Gustoca")
    ax2.set_title("Distribucija predikcija")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "calibration_curves.png"), dpi=140)
    plt.close()
    print(f"\nSpremljeno: calibration_curves.png")

    # ---- Odabir najbolje metode (najnizi Brier) ----
    best_method = cmp.sort_values("brier").iloc[0]
    print(f"\nNajbolja metoda po Brier scoreu: {best_method['metoda']}  "
          f"(Brier={best_method['brier']:.4f})")

    if best_method["metoda"] == "Isotonic":
        cal_iso.fit(X, y)
        best_cal = cal_iso
    elif best_method["metoda"] == "Sigmoid (Platt)":
        cal_sig.fit(X, y)
        best_cal = cal_sig
    else:
        # Original je najbolji - rijetko, ali moguce
        best_cal = pipe_orig

    joblib.dump(
        {
            "pipeline": best_cal,
            "features": features,
            "set_name": "OPTIMALNI_5_CALIBRATED",
            "calibration": best_method["metoda"],
        },
        os.path.join(OUT_DIR, "model_OPTIMALNI_5_calibrated.joblib"),
    )
    print(f"Spremljen kalibrirani model -> model_OPTIMALNI_5_calibrated.joblib")

    print()
    print("=" * 72)
    print("USPOREDBA")
    print("=" * 72)
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(cmp.to_string(index=False))

    print()
    print("Sto trazimo:")
    print("  - Nizi Brier   = bolje (kalibrirane vjerojatnosti)")
    print("  - Nizi LogLoss = bolje (vise sigurnosti u tocnim predikcijama)")
    print("  - AUC bi trebao OSTATI isti  (kalibracija ne mijenja ranking)")
    print()
    print("Calibration curve interpretacija:")
    print("  - Tocke blize y=x dijagonali  = bolje kalibrirano")
    print("  - 'Original RF' obicno odstupa (overconfident ili underconfident)")
    print("  - Isotonic/Sigmoid trebale bi pomaknuti tocke blize dijagonali")


if __name__ == "__main__":
    main()
