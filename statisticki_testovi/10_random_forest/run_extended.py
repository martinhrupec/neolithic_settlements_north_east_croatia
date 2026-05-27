"""
Random Forest — prosireno: 5-var model + grid search + decision rules
======================================================================

Ovo nadograduje run.py s tri stvari:

  1. NOVI MODEL: kompromis 5 varijabli  (drop nagib + rel_vis_100_500
     iz SIRI_7 jer su u permutation importance ~0)
     Final: vtt_r100, rel_vis_100_250, dist_rijeka_korig, strahler, aps_vis

  2. HYPERPARAMETER GRID SEARCH preko sva 3 modela (4, 5, 7 var):
     n_estimators × max_depth × min_samples_leaf × max_features

  3. DECISION RULES — ekstrahira citljive "ako ovo onda ono" pravila iz:
     (a) shallow surrogate stabla (max_depth=4) — interpretabilna pravila
     (b) najinformativnijih grana RF stabala

  4. PARTIAL DEPENDENCE PLOTS — kako se predikcija mijenja s vrijednoscu
     pojedine varijable

Output:
  - grid_search_rezultati.csv
  - decision_rules_text.txt        (citljiva pravila)
  - surrogate_tree.png              (vizualizacija)
  - partial_dependence.png          (kako pojedine varijable mijenjaju y)
  - model_OPTIMALNI_5.joblib        (spremljen kompromisni model)
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
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_val_predict, cross_validate,
)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import roc_auc_score


ROOT    = r"c:\Users\Martin\Desktop\skripte_za_diplomski\statisticki_testovi"
MASTER  = os.path.join(ROOT, "master_dataset.csv")
OUT_DIR = os.path.join(ROOT, "10_random_forest")

SEED = 42
N_CV = 5

# Tri feature seta — sad ukljucujuci 5-var kompromis
FEATURE_SETS = {
    "ROBUSTNE_4": {
        "categorical": ["vtt_r100"],
        "numerical":   ["rel_vis_100_250", "dist_rijeka_korig", "strahler"],
    },
    "OPTIMALNI_5": {
        "categorical": ["vtt_r100"],
        "numerical":   ["rel_vis_100_250", "dist_rijeka_korig", "strahler",
                        "aps_vis"],
    },
    "SIRI_7": {
        "categorical": ["vtt_r100"],
        "numerical":   ["rel_vis_100_250", "rel_vis_100_500",
                        "dist_rijeka_korig", "strahler",
                        "nagib", "aps_vis"],
    },
}


# ---------------------------------------------------------------------------
#  Helperi
# ---------------------------------------------------------------------------

def build_preprocessor(features):
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                 features["categorical"])],
        remainder="passthrough",
    )


def build_pipeline(features, **rf_kw):
    pre = build_preprocessor(features)
    clf = RandomForestClassifier(
        class_weight="balanced", n_jobs=-1, random_state=SEED, **rf_kw,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def get_feature_names_post_ohe(pipe, features):
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
    print(f"n_total = {len(sub)}  (274 vs 274)\n")

    # ===============================================================
    # 1) GRID SEARCH preko sva 3 modela
    # ===============================================================
    print("=" * 72)
    print("1) GRID SEARCH za sva 3 feature seta")
    print("=" * 72)

    param_grid = {
        "clf__n_estimators":      [200, 500],
        "clf__max_depth":         [None, 8, 15],
        "clf__min_samples_leaf":  [1, 5, 10],
        "clf__max_features":      ["sqrt", "log2"],
    }
    cv = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=SEED)

    grid_rows  = []
    best_pipes = {}

    for set_name, features in FEATURE_SETS.items():
        cols = features["categorical"] + features["numerical"]
        d    = sub.dropna(subset=cols).copy()
        X, y = d[cols], d["y"].values

        pipe = build_pipeline(features, n_estimators=200)
        gs   = GridSearchCV(pipe, param_grid=param_grid, cv=cv,
                            scoring="roc_auc", n_jobs=-1,
                            return_train_score=True)
        print(f"\n  [{set_name}] fitting grid...")
        gs.fit(X, y)

        best_params = gs.best_params_
        best_auc    = gs.best_score_
        # ucitaj train_auc za izabrane param-e
        idx = gs.best_index_
        train_auc = gs.cv_results_["mean_train_score"][idx]
        std_auc   = gs.cv_results_["std_test_score"][idx]

        print(f"    Najbolja AUC: {best_auc:.4f} ± {std_auc:.4f}  "
              f"(train AUC: {train_auc:.4f})")
        print(f"    Parametri: {best_params}")

        grid_rows.append({
            "set": set_name, "n_features": len(cols),
            "best_test_auc": best_auc, "std_test_auc": std_auc,
            "best_train_auc": train_auc,
            "overfit_gap": train_auc - best_auc,
            **{k.replace("clf__", ""): v for k, v in best_params.items()},
        })
        best_pipes[set_name] = (gs.best_estimator_, X, y, features)

    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(os.path.join(OUT_DIR, "grid_search_rezultati.csv"),
                   index=False, encoding="utf-8")

    print()
    print("USPOREDBA NAJBOLJIH MODELA:")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(grid_df.to_string(index=False))

    # ===============================================================
    # 2) DECISION RULES — surrogate stablo na predikcije RF-a
    # ===============================================================
    # Ideja: RF s 500 stabala je crna kutija. Treniramo JEDNO plitko stablo
    # da imitira RF predikcije — to daje citljive "ako-onda" rules s
    # razumnom tocnoscu.
    print()
    print("=" * 72)
    print("2) DECISION RULES (surrogate stablo) za OPTIMALNI_5")
    print("=" * 72)

    pipe, X, y, features = best_pipes["OPTIMALNI_5"]
    feat_names_ohe = get_feature_names_post_ohe(pipe, features)

    # Preprocesiraj X kroz OHE (da surrogate stablo radi na istim featurima)
    pre = pipe.named_steps["pre"]
    X_ohe = pre.transform(X)

    # RF predikcije (vjerojatnosti) kao "soft target"
    y_rf_prob = pipe.predict_proba(X)[:, 1]
    y_rf_class = (y_rf_prob >= 0.5).astype(int)

    # Surrogate plitko stablo
    surrogate = DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=15, random_state=SEED,
    )
    surrogate.fit(X_ohe, y_rf_class)
    fidelity = surrogate.score(X_ohe, y_rf_class)
    print(f"  Fidelity surrogate stabla (slaganje s RF-om): {fidelity:.3f}")
    print(f"  (1.0 = perfect; >0.85 znaci da pravila dobro reprezentiraju RF)")
    print()

    # Ispisi pravila
    rules_text = export_text(surrogate, feature_names=feat_names_ohe)
    print("PRAVILA (max_depth=4):")
    print(rules_text)

    # Spremi kao TXT
    with open(os.path.join(OUT_DIR, "decision_rules_text.txt"),
              "w", encoding="utf-8") as fh:
        fh.write(f"Surrogate decision tree za OPTIMALNI_5 model\n")
        fh.write(f"Fidelity (slaganje s RF-om): {fidelity:.3f}\n")
        fh.write(f"Maksimalna dubina: 4\n\n")
        fh.write(rules_text)

    # Vizualiziraj stablo
    fig, ax = plt.subplots(figsize=(16, 10))
    plot_tree(
        surrogate, feature_names=feat_names_ohe,
        class_names=["random_ceste", "neolitik"],
        filled=True, rounded=True, fontsize=8, ax=ax,
    )
    ax.set_title("Surrogate decision tree (interpretira RF predikcije)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "surrogate_tree.png"), dpi=120)
    plt.close()

    # ===============================================================
    # 3) PARTIAL DEPENDENCE PLOTS
    # ===============================================================
    # Kako se predicted prob mijenja kako vrijednost jedne varijable raste?
    print()
    print("=" * 72)
    print("3) PARTIAL DEPENDENCE PLOTS za OPTIMALNI_5")
    print("=" * 72)

    pdp_features = [
        "rel_vis_100_250", "dist_rijeka_korig", "strahler", "aps_vis",
        "vtt_r100",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, feat in enumerate(pdp_features):
        try:
            PartialDependenceDisplay.from_estimator(
                pipe, X, [feat], ax=axes[i],
                categorical_features=features["categorical"] if feat in features["categorical"] else None,
            )
            axes[i].set_title(f"PDP: {feat}")
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Skipped {feat}\n{e}",
                         ha="center", va="center", transform=axes[i].transAxes)
    # sakrij visak
    for j in range(len(pdp_features), len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "partial_dependence.png"), dpi=130)
    plt.close()
    print(f"  Spremljeno: partial_dependence.png")

    # ===============================================================
    # 4) Feature importance final
    # ===============================================================
    print()
    print("=" * 72)
    print("4) PERMUTATION IMPORTANCE za OPTIMALNI_5  (final model)")
    print("=" * 72)
    perm = permutation_importance(pipe, X, y, n_repeats=30,
                                   random_state=SEED, n_jobs=-1,
                                   scoring="roc_auc")
    fi_df = pd.DataFrame({
        "feature":   list(X.columns),
        "perm_mean": perm.importances_mean,
        "perm_std":  perm.importances_std,
    }).sort_values("perm_mean", ascending=False)
    fi_df.to_csv(os.path.join(OUT_DIR, "importance_OPTIMALNI_5.csv"),
                 index=False, encoding="utf-8")
    print(fi_df.to_string(index=False))

    # ===============================================================
    # 5) Spremi finalne modele
    # ===============================================================
    print()
    for set_name, (pipe, X, y, features) in best_pipes.items():
        joblib.dump(
            {"pipeline": pipe, "features": features, "set_name": set_name},
            os.path.join(OUT_DIR, f"model_{set_name}.joblib"),
        )
        print(f"  Spremljen: model_{set_name}.joblib")

    # ===============================================================
    # 6) Sazetak
    # ===============================================================
    print()
    print("=" * 72)
    print("SAZETAK")
    print("=" * 72)
    print()
    print(grid_df[["set","n_features","best_test_auc","std_test_auc",
                    "overfit_gap","n_estimators","max_depth",
                    "min_samples_leaf"]].to_string(index=False))


if __name__ == "__main__":
    main()
