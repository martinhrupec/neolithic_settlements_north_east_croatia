"""
Random Forest analiza: predviđanje neolitičke prisutnosti
==========================================================

Cilj: usporediti dva modela:
  - Model A (robusni):  4 varijable iz univariate + thinning sensitivity
  - Model B (širi):     7 varijabli (4 robusne + 3 dopunske dimenzije iz korelacije)

Workflow:
  1. Učitaj master, filtriraj na neolitik + nasumicni_ceste (n = 548)
  2. Definiraj dva feature seta
  3. One-hot encoding kategorijskih varijabli (vtt_r100)
  4. 5-fold stratified CV: accuracy + ROC AUC + F1
  5. Default hiperparametri + 2 alternative (max_depth, n_estimators)
  6. Feature importance (Gini + Permutation) na najboljim modelima
  7. Spremi modele za kasniju heatmap generaciju

Target: y = 1 (neolitik), y = 0 (nasumicni_ceste).
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve,
)


# ---------------------------------------------------------------------------
#  Konfiguracija
# ---------------------------------------------------------------------------

ROOT    = r"c:\Users\Martin\Desktop\skripte_za_diplomski\statisticki_testovi"
MASTER  = os.path.join(ROOT, "master_dataset.csv")
OUT_DIR = os.path.join(ROOT, "10_random_forest")

SEED   = 42
N_CV   = 5

# Feature setovi
FEATURES_ROBUSTNE = {
    "categorical": ["vtt_r100"],
    "numerical":   ["rel_vis_100_250", "dist_rijeka_korig", "strahler"],
}
FEATURES_SIRI = {
    "categorical": ["vtt_r100"],
    "numerical":   ["rel_vis_100_250", "rel_vis_100_500",
                    "dist_rijeka_korig", "strahler",
                    "nagib", "aps_vis"],
}

# Hiperparametri za usporedbu
HP_GRID = [
    {"name": "default",       "n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
    {"name": "deep",          "n_estimators": 500, "max_depth": None, "min_samples_leaf": 1},
    {"name": "regularizirano","n_estimators": 300, "max_depth": 10,   "min_samples_leaf": 5},
]


# ---------------------------------------------------------------------------
#  Helperi
# ---------------------------------------------------------------------------

def build_pipeline(features, n_estimators, max_depth, min_samples_leaf):
    """Pipeline: OHE za kategorijske + RF."""
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                 features["categorical"])],
        remainder="passthrough",
    )
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def cv_evaluate(pipe, X, y):
    cv = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=SEED)
    scoring = {"acc": "accuracy", "auc": "roc_auc", "f1": "f1"}
    cvr = cross_validate(pipe, X, y, cv=cv, scoring=scoring,
                         return_train_score=True, n_jobs=-1)
    return {
        "acc_mean":  cvr["test_acc"].mean(),  "acc_std":  cvr["test_acc"].std(),
        "auc_mean":  cvr["test_auc"].mean(),  "auc_std":  cvr["test_auc"].std(),
        "f1_mean":   cvr["test_f1"].mean(),   "f1_std":   cvr["test_f1"].std(),
        "train_acc": cvr["train_acc"].mean(),
        "train_auc": cvr["train_auc"].mean(),
    }


def get_feature_names(pipe, features):
    ohe = pipe.named_steps["pre"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(features["categorical"]))
    return cat_names + list(features["numerical"])


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(MASTER)
    sub = df[df["tip_sloja"].isin(["neolitik", "nasumicni_ceste"])].copy()
    sub["y"] = (sub["tip_sloja"] == "neolitik").astype(int)

    n_total = len(sub)
    print(f"n_total = {n_total}  ({int(sub['y'].sum())} neolitik, "
          f"{int((1 - sub['y']).sum())} nasumicni_ceste)\n")

    # Izbacujemo redove s NaN u features
    sub_orig = sub.copy()

    results = []
    best_pipes = {}

    for set_name, features in [("ROBUSTNE_4", FEATURES_ROBUSTNE),
                               ("SIRI_7",     FEATURES_SIRI)]:
        all_cols = features["categorical"] + features["numerical"]
        d = sub_orig.dropna(subset=all_cols).copy()
        X = d[all_cols]
        y = d["y"].values
        n_used = len(d)

        print("=" * 72)
        print(f"FEATURE SET: {set_name}")
        print(f"  varijable ({len(all_cols)}): {all_cols}")
        print(f"  n koristen = {n_used}  (nakon dropna)")
        print("=" * 72)

        best = None
        for hp in HP_GRID:
            pipe = build_pipeline(features, hp["n_estimators"],
                                  hp["max_depth"], hp["min_samples_leaf"])
            m = cv_evaluate(pipe, X, y)
            print(f"  [{hp['name']:<16}] n_est={hp['n_estimators']:>3d}  "
                  f"depth={str(hp['max_depth']):<5s}  leaf={hp['min_samples_leaf']}  ||  "
                  f"acc={m['acc_mean']:.3f}±{m['acc_std']:.3f}  "
                  f"AUC={m['auc_mean']:.3f}±{m['auc_std']:.3f}  "
                  f"F1={m['f1_mean']:.3f}±{m['f1_std']:.3f}  "
                  f"(train_acc={m['train_acc']:.3f})")
            row = {"set": set_name, **hp, **m}
            results.append(row)
            if best is None or m["auc_mean"] > best["auc_mean"]:
                best = row
                best_pipe = pipe
        print(f"\n  Najbolji: {best['name']} (AUC = {best['auc_mean']:.3f})")
        # Fitiraj najbolji na cijelom uzorku
        best_pipe.fit(X, y)
        best_pipes[set_name] = (best_pipe, X, y, features)
        print()

    # ---- Spremi tablicu rezultata ----
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUT_DIR, "cv_rezultati.csv"),
                  index=False, encoding="utf-8")

    # ---- Feature importance + dijagnostika za oba modela ----
    for set_name, (pipe, X, y, features) in best_pipes.items():
        print("=" * 72)
        print(f"FEATURE IMPORTANCE: {set_name}")
        print("=" * 72)

        ohe_feat_names = get_feature_names(pipe, features)
        clf = pipe.named_steps["clf"]

        # Gini importance je po POST-OHE features. Permutation importance je
        # po INPUT kolonama. Agregiramo Gini po izvornoj kategoriji da budu
        # usporedivi sa permutation.
        gini_post_ohe = clf.feature_importances_

        X_cols = list(X.columns)
        gini_by_input = {c: 0.0 for c in X_cols}
        for fname, gi in zip(ohe_feat_names, gini_post_ohe):
            matched_cat = None
            for c in features["categorical"]:
                if fname.startswith(c + "_"):
                    matched_cat = c
                    break
            key = matched_cat if matched_cat else fname
            gini_by_input[key] = gini_by_input.get(key, 0.0) + gi

        # Permutation importance po input kolonama
        perm = permutation_importance(pipe, X, y, n_repeats=20,
                                       random_state=SEED, n_jobs=-1,
                                       scoring="roc_auc")
        perm_mean = perm.importances_mean
        perm_std  = perm.importances_std

        fi_df = pd.DataFrame({
            "feature":         X_cols,
            "gini_importance": [gini_by_input[c] for c in X_cols],
            "perm_mean":       perm_mean,
            "perm_std":        perm_std,
        }).sort_values("perm_mean", ascending=False)
        fi_df.to_csv(os.path.join(OUT_DIR, f"importance_{set_name}.csv"),
                     index=False, encoding="utf-8")

        print()
        print(fi_df.to_string(index=False))

        # Bar plot
        fig, ax = plt.subplots(figsize=(8, max(3, len(fi_df) * 0.5)))
        y_pos = np.arange(len(fi_df))
        ax.barh(y_pos, fi_df["perm_mean"], xerr=fi_df["perm_std"],
                color="steelblue", alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(fi_df["feature"])
        ax.invert_yaxis()
        ax.set_xlabel("Permutation importance (drop in AUC)")
        ax.set_title(f"Feature importance: {set_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"importance_{set_name}.png"), dpi=150)
        plt.close()
        print()

    # ---- ROC krivulje + confusion matrice ----
    fig_roc, ax_roc = plt.subplots(figsize=(7, 7))
    for set_name, (pipe, X, y, _) in best_pipes.items():
        cv = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=SEED)
        y_pred = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba",
                                    n_jobs=-1)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_pred)
        auc = roc_auc_score(y, y_pred)
        ax_roc.plot(fpr, tpr, label=f"{set_name}  (AUC = {auc:.3f})", linewidth=2)
        # Confusion matrix pri pragu 0.5
        y_pred_class = (y_pred >= 0.5).astype(int)
        cm = confusion_matrix(y, y_pred_class)
        print(f"\n[{set_name}] Confusion matrix (prag 0.5):")
        print(f"               predikcija: 0      1")
        print(f"  stvarno 0:           {cm[0,0]:>5d}  {cm[0,1]:>5d}")
        print(f"  stvarno 1:           {cm[1,0]:>5d}  {cm[1,1]:>5d}")
        print(f"  Accuracy: {accuracy_score(y, y_pred_class):.3f}")
        print(f"  ROC AUC : {auc:.3f}")
        print(f"  F1      : {f1_score(y, y_pred_class):.3f}")

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4, label="random")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC krivulje (CV out-of-fold predikcije)")
    ax_roc.legend()
    ax_roc.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "roc_krivulje.png"), dpi=150)
    plt.close()

    # ---- Spremi modele za kasniju heatmap generaciju ----
    for set_name, (pipe, X, y, features) in best_pipes.items():
        joblib.dump(
            {"pipeline": pipe, "features": features, "set_name": set_name},
            os.path.join(OUT_DIR, f"model_{set_name}.joblib"),
        )

    # ---- Sazetak ----
    print()
    print("=" * 72)
    print("SAZETAK")
    print("=" * 72)
    print()
    print(res_df[["set","name","n_estimators","max_depth","min_samples_leaf",
                  "acc_mean","auc_mean","f1_mean"]].to_string(index=False))
    print()
    print(f"Sve spremljeno u: {OUT_DIR}")
    print(f"  - cv_rezultati.csv")
    print(f"  - importance_ROBUSTNE_4.csv / .png")
    print(f"  - importance_SIRI_7.csv / .png")
    print(f"  - roc_krivulje.png")
    print(f"  - model_ROBUSTNE_4.joblib / model_SIRI_7.joblib (za heatmap)")


if __name__ == "__main__":
    main()
