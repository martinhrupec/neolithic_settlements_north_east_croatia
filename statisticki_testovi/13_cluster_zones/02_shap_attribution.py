"""
SHAP feature attribution za selektirane cluster zone
=====================================================

Cilj: za odredene cluster polygone identificirati koji feature je
najvise pridonio predikciji modela. Daje interpretaciju "zasto je
ova zona prepoznata kao neolitik".

Selekcija clustera:
  thr 0.70:  top 10 po score_size  +  top 10 po score_discovery
  thr 0.75:  top 10 po score_size  +  top 10 po score_discovery
  thr 0.85:  top 50 po score_size
  thr 0.90:  SVI clusteri (~854)

Po clusteru:
  1. Maskira sve 5 input rastera s polygonom (rasterio.mask)
  2. Izracuna reprezentativne feature vrijednosti:
       - vtt_r100   = MODUS unutar polygona
       - strahler   = MODUS
       - rel_vis_100_250, dist_rijeka_korig, aps_vis = MEAN
  3. Transformira kroz OneHotEncoder iz spremljenog modela
  4. SHAP TreeExplainer -> per-feature doprinos predikciji
  5. Agregira OHE doprinose nazad na vtt_r100 (zbroj svih kategorija)

Output:
  - shap_per_cluster.csv   - jedan record per cluster sa vrijednostima i SHAP-om
  - shap_aggregate.png     - prosjecni doprinos po feature-u
  - shap_per_source.png    - per-source bar chart-i
  - shap_summary.txt       - top 3 doprinosa za svaki cluster, formatirano

Dependencije:  pip install shap
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import shape
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap


# ============================================================
#  POSTAVKE
# ============================================================

ROOT          = r"c:\Users\Martin\Desktop\skripte_za_diplomski"
CLUSTER_DIR   = os.path.join(ROOT, "statisticki_testovi", "13_cluster_zones")
RASTERS_DIR   = os.path.join(ROOT, "12_heatmap_qgis", "rasters")
META_PATH     = os.path.join(RASTERS_DIR, "metadata.json")
MODEL_PATH    = os.path.join(ROOT, "statisticki_testovi",
                              "10_random_forest", "model_OPTIMALNI_5.joblib")
OUT_DIR       = CLUSTER_DIR

# Selekcija: samo svi cluster-i na pragu 0.85.
# Cista metodologija - bez cherry-picking po score-ovima, jedan prag = jedan source.
# Discovery zone i core zone ce se PRIRODNO razlikovati u k-means rezultatu (03)
# ako imaju razlicite SHAP profile.
SELECTION = {
    "85": [("all", None)],
}


# ============================================================
#  HELPERS
# ============================================================

def safe_mean(arr, nodata):
    """Mean ignorirajuci nodata i NaN."""
    a = np.asarray(arr, dtype=float)
    valid = np.isfinite(a)
    if nodata is not None:
        valid &= (a != nodata)
    if not valid.any():
        return np.nan
    return float(a[valid].mean())


def safe_mode(arr, exclude_zero=True):
    """Mode (najcesca vrijednost) iz integer arraya, ignorirajuci 0."""
    a = np.asarray(arr).ravel()
    if exclude_zero:
        a = a[a > 0]
    if len(a) == 0:
        return None
    counts = Counter(a.tolist())
    return counts.most_common(1)[0][0]


def get_shap_positive_class(shap_vals_raw, n_features):
    """
    SHAP RF binary classification moze vratiti:
      - list [shap_class0, shap_class1] (older API)
      - array (n_samples, n_features) (positive class)
      - array (n_samples, n_features, n_classes)
    Ovo izvlaci redovi 0 za pozitivnu klasu.
    """
    if isinstance(shap_vals_raw, list):
        # older API
        return np.asarray(shap_vals_raw[1])[0]
    arr = np.asarray(shap_vals_raw)
    if arr.ndim == 3:
        return arr[0, :, 1]
    return arr[0]


# ============================================================
#  GLAVNA LOGIKA
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 72)
    print("SHAP FEATURE ATTRIBUTION za cluster zone")
    print("=" * 72)

    # ----- Ucitaj model -----
    print(f"\n[1] Ucitavam model: {MODEL_PATH}")
    model_dict = joblib.load(MODEL_PATH)
    pipe       = model_dict["pipeline"]
    features   = model_dict["features"]
    cat_cols   = features["categorical"]
    num_cols   = features["numerical"]
    all_cols   = cat_cols + num_cols
    print(f"    features = {all_cols}")

    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    ohe = pre.named_transformers_["cat"]
    cat_names_ohe = list(ohe.get_feature_names_out(cat_cols))
    all_feature_names_ohe = cat_names_ohe + num_cols
    print(f"    post-OHE features = {all_feature_names_ohe}")

    # SHAP TreeExplainer
    print(f"\n[2] Inicijaliziram SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(clf)

    # ----- Ucitaj metadata -----
    print(f"\n[3] Ucitavam metadata: {META_PATH}")
    with open(META_PATH, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    soil_types = {int(k): v for k, v in meta["soil_types"].items()}
    soil_types[0] = "Nepoznato"

    raster_paths = {
        "rel_vis":   os.path.join(RASTERS_DIR, meta["files"]["rel_vis_100_250"]),
        "vtt":       os.path.join(RASTERS_DIR, meta["files"]["vtt_r100"]),
        "dist":      os.path.join(RASTERS_DIR, meta["files"]["dist_rijeka_korig"]),
        "strah":     os.path.join(RASTERS_DIR, meta["files"]["strahler"]),
        "apsv":      meta["files"]["aps_vis"],
    }

    # ----- Selektiraj clustere -----
    print(f"\n[4] Selektiram clustere prema SELECTION...")
    all_selected = []
    for thr_str, criteria_list in SELECTION.items():
        geojson_path = os.path.join(CLUSTER_DIR, f"clusters_thr_{thr_str}.geojson")
        if not os.path.exists(geojson_path):
            print(f"    UPOZORENJE: {geojson_path} ne postoji, preskacem.")
            continue
        with open(geojson_path, "r", encoding="utf-8") as fh:
            fc = json.load(fh)

        for criterion, n_top in criteria_list:
            if criterion == "all":
                selected = fc["features"]
                source = f"thr{thr_str}_all"
            else:
                sorted_feats = sorted(
                    fc["features"],
                    key=lambda f: f["properties"].get(criterion, 0),
                    reverse=True,
                )
                selected = sorted_feats[:n_top]
                source = f"thr{thr_str}_{criterion}"

            for f in selected:
                all_selected.append({"feature": f, "source": source})
            print(f"    {source}: {len(selected)} clustera")

    print(f"    UKUPNO: {len(all_selected)} cluster entries")

    # ----- Compute SHAP per cluster -----
    print(f"\n[5] Maskiram rastere i racunam SHAP...")
    t0 = time.time()

    # Otvori sve rastere jednom
    rds = {k: rasterio.open(v) for k, v in raster_paths.items()}

    records = []
    n_total = len(all_selected)
    for i, entry in enumerate(all_selected):
        feat   = entry["feature"]
        source = entry["source"]
        geom   = shape(feat["geometry"])
        props  = feat["properties"]

        try:
            # Mask + statistika za svaki raster
            rel_arr, _   = rio_mask(rds["rel_vis"], [geom], crop=True,
                                    filled=False)
            vtt_arr, _   = rio_mask(rds["vtt"],     [geom], crop=True,
                                    filled=False)
            dist_arr, _  = rio_mask(rds["dist"],    [geom], crop=True,
                                    filled=False)
            strah_arr, _ = rio_mask(rds["strah"],   [geom], crop=True,
                                    filled=False)
            apsv_arr, _  = rio_mask(rds["apsv"],    [geom], crop=True,
                                    filled=False)

            rel_vals   = rel_arr[0].compressed()
            vtt_vals   = vtt_arr[0].compressed()
            dist_vals  = dist_arr[0].compressed()
            strah_vals = strah_arr[0].compressed()
            apsv_vals  = apsv_arr[0].compressed()

            rel_mean   = safe_mean(rel_vals,   rds["rel_vis"].nodata)
            dist_mean  = safe_mean(dist_vals,  rds["dist"].nodata)
            apsv_mean  = safe_mean(apsv_vals,  rds["apsv"].nodata)
            vtt_mode   = safe_mode(vtt_vals.astype(int),   exclude_zero=True)
            strah_mode = safe_mode(strah_vals.astype(int), exclude_zero=True)

            if any(v is None or np.isnan(v) for v in
                   [rel_mean, dist_mean, apsv_mean]):
                continue
            if vtt_mode is None or strah_mode is None:
                continue
        except Exception as e:
            continue

        vtt_str = soil_types.get(int(vtt_mode), "Nepoznato")

        X_row = pd.DataFrame([{
            "vtt_r100":          vtt_str,
            "rel_vis_100_250":   float(rel_mean),
            "dist_rijeka_korig": float(dist_mean),
            "strahler":          int(strah_mode),
            "aps_vis":           float(apsv_mean),
        }])[all_cols]

        # Predict probability
        pred_prob = float(pipe.predict_proba(X_row)[0, 1])

        # Transform kroz preprocessor i racunaj SHAP
        X_trans = pre.transform(X_row)
        shap_raw = explainer.shap_values(X_trans)
        shap_pos = get_shap_positive_class(shap_raw, len(all_feature_names_ohe))

        # Agregiraj OHE doprinose nazad na original feature
        feat_shap = {c: 0.0 for c in all_cols}
        for name, val in zip(all_feature_names_ohe, shap_pos):
            matched = None
            for cat_orig in cat_cols:
                if name.startswith(cat_orig + "_"):
                    matched = cat_orig
                    break
            key = matched if matched else name
            feat_shap[key] = feat_shap.get(key, 0.0) + float(val)

        rec = {
            "source":              source,
            "cluster_id":          props["cluster_id"],
            "threshold":           props["threshold"],
            "area_km2":            props["area_km2"],
            "prob_max":            props["prob_max"],
            "prob_mean":           props["prob_mean"],
            "centroid_x":          props["centroid_x"],
            "centroid_y":          props["centroid_y"],
            "dist_to_known_m":     props["dist_to_known_m"],
            "n_known_inside":      props["n_known_inside"],
            "score_size":          props.get("score_size"),
            "score_discovery":     props.get("score_discovery"),
            # Reprezentativne feature vrijednosti
            "val_vtt_r100":        vtt_str,
            "val_rel_vis":         round(float(rel_mean), 3),
            "val_dist_rijeka":     round(float(dist_mean), 1),
            "val_strahler":        int(strah_mode),
            "val_aps_vis":         round(float(apsv_mean), 1),
            # Pred prob za mean feature vector
            "pred_prob_mean_feat": round(pred_prob, 4),
            # SHAP per original feature
            "shap_vtt_r100":          round(feat_shap.get("vtt_r100", 0), 5),
            "shap_rel_vis_100_250":   round(feat_shap.get("rel_vis_100_250", 0), 5),
            "shap_dist_rijeka_korig": round(feat_shap.get("dist_rijeka_korig", 0), 5),
            "shap_strahler":          round(feat_shap.get("strahler", 0), 5),
            "shap_aps_vis":           round(feat_shap.get("aps_vis", 0), 5),
        }
        records.append(rec)

        if (i + 1) % 100 == 0 or i == n_total - 1:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_total - i - 1)
            print(f"    {i+1}/{n_total}  elapsed {elapsed:.0f}s  ETA {eta:.0f}s")

    # Zatvori rastere
    for ds in rds.values():
        ds.close()

    if not records:
        print("\nGreska: nema record-a generiranih (svi clusteri preskaceni).")
        return

    # ----- Spremi CSV -----
    df = pd.DataFrame(records)
    csv_path = os.path.join(OUT_DIR, "shap_per_cluster.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n[6] -> {csv_path}  ({len(df)} record-a)")

    # ----- Aggregate plot -----
    shap_cols = ["shap_vtt_r100", "shap_rel_vis_100_250",
                 "shap_dist_rijeka_korig", "shap_strahler", "shap_aps_vis"]
    pretty = {
        "shap_vtt_r100":          "vtt_r100 (tip tla)",
        "shap_rel_vis_100_250":   "rel_vis_100_250",
        "shap_dist_rijeka_korig": "dist_rijeka_korig",
        "shap_strahler":          "strahler",
        "shap_aps_vis":           "aps_vis",
    }

    shap_mean = df[shap_cols].mean()
    shap_mean_abs = df[shap_cols].abs().mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#1f77b4" if v > 0 else "#d62728" for v in shap_mean.values]
    ax1.barh([pretty[c] for c in shap_cols], shap_mean.values, color=colors)
    ax1.axvline(0, color="black", linewidth=0.6)
    ax1.set_xlabel("Mean SHAP value (signed)  - + povecava p neolitik, - smanjuje")
    ax1.set_title(f"Prosjecni doprinos featurea (n={len(df)} clustera)")
    ax1.grid(alpha=0.3, axis="x")

    ax2.barh([pretty[c] for c in shap_cols], shap_mean_abs.values, color="darkred")
    ax2.set_xlabel("Mean |SHAP value|")
    ax2.set_title("Prosjecna magnituda doprinosa (vaznost)")
    ax2.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_aggregate.png"), dpi=140)
    plt.close()
    print(f"  -> shap_aggregate.png")

    # ----- Per-source plot (samo ako ima vise od 1 source-a) -----
    sources = df["source"].unique()
    n_src = len(sources)
    if n_src > 1:
        ncols = 2
        nrows = (n_src + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4 * nrows))
        axes = np.atleast_1d(axes).flatten()
        for ax, src in zip(axes, sources):
            sub = df[df["source"] == src]
            if len(sub) == 0:
                continue
            s_mean = sub[shap_cols].mean()
            colors = ["#1f77b4" if v > 0 else "#d62728" for v in s_mean.values]
            ax.barh([pretty[c] for c in shap_cols], s_mean.values, color=colors)
            ax.axvline(0, color="black", linewidth=0.6)
            ax.set_title(f"{src}  (n={len(sub)})")
            ax.grid(alpha=0.3, axis="x")
        for ax in axes[len(sources):]:
            ax.set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "shap_per_source.png"), dpi=140)
        plt.close()
        print(f"  -> shap_per_source.png")
    else:
        print(f"  per_source plot preskocen (samo 1 source: {sources[0]})")

    # ----- Per-cluster text summary -----
    summary_lines = ["SHAP per-cluster summary",
                     "=" * 60,
                     ""]
    for src in sources:
        sub = df[df["source"] == src].copy()
        if len(sub) == 0:
            continue
        summary_lines.append(f"\n{'='*60}")
        summary_lines.append(f"SOURCE: {src}    n_clustera = {len(sub)}")
        summary_lines.append(f"{'='*60}")

        # Sort by score (decay order) — koristim score iz source-a
        if "discovery" in src:
            sort_col = "score_discovery"
        elif "size" in src:
            sort_col = "score_size"
        else:
            sort_col = "prob_max"
        sub = sub.sort_values(sort_col, ascending=False)

        # Prvi 20 (ili svi ako manje)
        for _, row in sub.head(20).iterrows():
            shap_pairs = [(pretty[c], row[c]) for c in shap_cols]
            shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            top3 = shap_pairs[:3]
            contrib = ", ".join(f"{n}={v:+.3f}" for n, v in top3)
            summary_lines.append(
                f"  cid={int(row['cluster_id']):>4d}  "
                f"p={row['prob_max']:.2f}  "
                f"area={row['area_km2']:.3f} km2  "
                f"dist_known={row['dist_to_known_m']:.0f} m  "
                f"({row['n_known_inside']} inside)  "
                f"vtt={row['val_vtt_r100']}"
            )
            summary_lines.append(f"    top3 doprinosi: {contrib}")

    sum_path = os.path.join(OUT_DIR, "shap_summary.txt")
    with open(sum_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(summary_lines))
    print(f"  -> shap_summary.txt")

    # ----- Konzolni sazetak -----
    print()
    print("=" * 72)
    print("AGREGATNI DOPRINOSI PO FEATURE-U (mean SHAP, sva selekcija)")
    print("=" * 72)
    for c in shap_cols:
        print(f"  {pretty[c]:<32s}: mean={df[c].mean():+.4f}  "
              f"|mean|={df[c].abs().mean():.4f}")

    print()
    print(f"Sve spremljeno u: {OUT_DIR}")


if __name__ == "__main__":
    main()
