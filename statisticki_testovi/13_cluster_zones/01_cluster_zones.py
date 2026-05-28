"""
Cluster zones identifikacija u probability rasteru
====================================================

Cilj: identificirati "hot-spot" zone modela na 4 razine pragova predikcije.
Za svaki prag, generira polygon layer (GeoJSON) gdje je svaki record povezana
zona pixela koja zadovoljava uvjet prob >= prag.

Pragovi:  0.70, 0.75, 0.85, 0.90  (4 GeoJSON layera)

Output GeoJSON-i se ucitavaju u QGIS samo drag-and-drop (no fiona dependency).

Atributi po polygonu:
  - cluster_id           jedinstveni ID unutar threshold-a
  - threshold            koji prag
  - area_km2             povrsina u km2
  - n_pixels             broj piksela
  - prob_max             najvisa vjerojatnost u clusteru
  - prob_mean            prosjecna vjerojatnost
  - prob_p75             75. percentil
  - centroid_x, centroid_y    koordinate centroida
  - dist_to_known_m      udaljenost centroida do najblizeg poznatog neolitik lokaliteta
  - nearest_known_fid    fid tog najblizeg lokaliteta
  - contains_known       1 ako polygon sadrzi barem jedan poznati neolitik, inace 0
  - n_known_inside       koliko poznatih neolitika je unutar polygona
  - score_size           rang po velicini (= area_km2 * prob_mean)
  - score_discovery      rang po discovery potential (= prob_max * log(1+dist))

Output:
  - clusters_thr_70.geojson, clusters_thr_75.geojson, clusters_thr_85.geojson, clusters_thr_90.geojson
  - clusters_summary.csv     sve clustere objedinjeno
  - clusters_top10.txt       citljiv summary top 10 per threshold po oba scoring-a
"""

import os
import json
import numpy as np
import pandas as pd
import rasterio
from rasterio import features as rfeatures
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import shape, mapping, Point
from shapely.ops import unary_union


PROB_RASTER = (r"c:\Users\Martin\Desktop\skripte_za_diplomski"
               r"\12_heatmap_qgis\probability_neolitik.tif")
NEO_COORDS  = (r"c:\Users\Martin\Desktop\skripte_za_diplomski"
               r"\statisticki_testovi\01_prostorna_autokorelacija\neolitik_coords.csv")
OUT_DIR     = (r"c:\Users\Martin\Desktop\skripte_za_diplomski"
               r"\statisticki_testovi\13_cluster_zones")

THRESHOLDS    = [0.70, 0.75, 0.85, 0.90]

# Dinamicki min area: pri niskom pragu (puno suma) filtriramo agresivno,
# pri visokom pragu (model siguran) hvatamo i male clustere.
MIN_AREA_KM2_PER_THRESHOLD = {
    0.70: 0.05,    # 5 ha  - filtrira sum
    0.75: 0.025,   # 2.5 ha
    0.85: 0.01,    # 1 ha  - single tell-scale
    0.90: 0.0025,  # 0.25 ha - i pojedinacni piksel ako prob >= 0.90
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 72)
    print("CLUSTER ZONES - identifikacija hot-spot zona")
    print("=" * 72)

    # ----- Load probability raster -----
    print(f"\n[1] Ucitavam probability raster: {PROB_RASTER}")
    with rasterio.open(PROB_RASTER) as ds:
        prob = ds.read(1)
        transform = ds.transform
        crs = ds.crs
        nodata = ds.nodata
        res_x = transform.a
        res_y = -transform.e
    print(f"    shape = {prob.shape}, res = {res_x:.1f} x {res_y:.1f} m, nodata = {nodata}")

    pixel_area_km2 = (res_x * res_y) / 1_000_000.0
    print(f"    pixel area = {pixel_area_km2:.6f} km2")
    print(f"    min area po pragu:")
    for thr in THRESHOLDS:
        min_area = MIN_AREA_KM2_PER_THRESHOLD[thr]
        n_px = max(1, int(np.ceil(min_area / pixel_area_km2)))
        print(f"      thr {thr:.2f}: {min_area:.4f} km2 = {n_px} px")

    if nodata is not None:
        valid = (prob != nodata) & np.isfinite(prob)
    else:
        valid = np.isfinite(prob)
    print(f"    valid piksela: {int(valid.sum()):,} / {valid.size:,}")

    # ----- Load known neolitik coords -----
    print(f"\n[2] Ucitavam poznate neolitik koordinate: {NEO_COORDS}")
    neo = pd.read_csv(NEO_COORDS)
    print(f"    n = {len(neo)}")
    neo_xy   = neo[["x", "y"]].values
    neo_fids = neo["fid"].values
    neo_tree = cKDTree(neo_xy)
    neo_points = [Point(x, y) for x, y in neo_xy]

    # CRS za GeoJSON
    crs_name = crs.to_string() if crs else "EPSG:3765"   # fallback na HTRS96

    summary_records = []

    for thr in THRESHOLDS:
        print(f"\n[3] Threshold = {thr}")

        # Threshold-specific min area
        min_area_km2 = MIN_AREA_KM2_PER_THRESHOLD[thr]
        min_pixels = max(1, int(np.ceil(min_area_km2 / pixel_area_km2)))

        binary = (prob >= thr) & valid

        # Connected components labeling
        labels, n_init = ndimage.label(binary)
        if n_init == 0:
            print(f"    nema piksela >= {thr}, preskacem.")
            continue

        # Size filter
        sizes = ndimage.sum(binary, labels, range(1, n_init + 1)).astype(int)
        keep_ids_orig = np.where(sizes >= min_pixels)[0] + 1
        print(f"    inicijalno {n_init} cluster-a -> "
              f"{len(keep_ids_orig)} nakon size filtera "
              f"(>= {min_pixels} px = {min_area_km2:.4f} km2)")

        if len(keep_ids_orig) == 0:
            continue

        # Output GeoJSON
        geojson_path = os.path.join(OUT_DIR, f"clusters_thr_{int(thr*100):02d}.geojson")

        # ===== Optimizacija: polygonize JEDNOM preko cijelog labels array =====
        print("    polygonizing (single pass)...")
        # Brzi mask: lookup array umjesto np.isin
        keep_lookup = np.zeros(int(labels.max()) + 1, dtype=bool)
        keep_lookup[keep_ids_orig] = True
        keep_mask = keep_lookup[labels]
        geom_by_label = {}
        for geom_dict, lab in rfeatures.shapes(
                labels.astype(np.int32), mask=keep_mask, transform=transform):
            lab_i = int(lab)
            g = shape(geom_dict)
            if lab_i in geom_by_label:
                prev = geom_by_label[lab_i]
                if isinstance(prev, list):
                    prev.append(g)
                else:
                    geom_by_label[lab_i] = [prev, g]
            else:
                geom_by_label[lab_i] = g

        # Bounding box za svaki cluster (za sub-array statistike)
        slices = ndimage.find_objects(labels)

        print(f"    iterating {len(keep_ids_orig)} clusters...")
        records = []
        for new_cid, old_cid in enumerate(keep_ids_orig, start=1):
            if new_cid % 500 == 0:
                print(f"      {new_cid}/{len(keep_ids_orig)}...")
            g = geom_by_label.get(int(old_cid))
            if g is None:
                continue
            merged = unary_union(g) if isinstance(g, list) else g

            # Sub-array statistika (brzo, na bounding boxu cluster-a)
            sl = slices[old_cid - 1]
            sub_labels = labels[sl]
            sub_mask = (sub_labels == old_cid)
            sub_probs = prob[sl][sub_mask]

            n_pix = int(sub_mask.sum())
            area_km2 = n_pix * pixel_area_km2
            prob_max = float(sub_probs.max())
            prob_mean = float(sub_probs.mean())
            prob_p75 = float(np.percentile(sub_probs, 75))

            cx, cy = merged.centroid.x, merged.centroid.y

            dist, idx = neo_tree.query([cx, cy], k=1)
            nearest_fid = int(neo_fids[idx])

            # n_known_inside: vektoriziraj — koje su tocke unutar polygon bounds
            minx, miny, maxx, maxy = merged.bounds
            in_box = (
                (neo_xy[:, 0] >= minx) & (neo_xy[:, 0] <= maxx) &
                (neo_xy[:, 1] >= miny) & (neo_xy[:, 1] <= maxy)
            )
            n_inside = 0
            for k in np.where(in_box)[0]:
                if merged.contains(neo_points[k]):
                    n_inside += 1
            contains_known = 1 if n_inside > 0 else 0

            # Scores
            score_size = area_km2 * prob_mean
            score_discovery = prob_max * float(np.log1p(dist))

            props = {
                "cluster_id":        new_cid,
                "threshold":         float(thr),
                "area_km2":          round(area_km2, 4),
                "n_pixels":          n_pix,
                "prob_max":          round(prob_max, 4),
                "prob_mean":         round(prob_mean, 4),
                "prob_p75":          round(prob_p75, 4),
                "centroid_x":        round(cx, 2),
                "centroid_y":        round(cy, 2),
                "dist_to_known_m":   round(float(dist), 2),
                "nearest_known_fid": nearest_fid,
                "contains_known":    contains_known,
                "n_known_inside":    n_inside,
                "score_size":        round(score_size, 6),
                "score_discovery":   round(score_discovery, 6),
            }
            records.append({"geometry": merged, "properties": props})

        # Write GeoJSON (pure Python, no fiona)
        if records:
            features = []
            for rec in records:
                features.append({
                    "type":       "Feature",
                    "geometry":   mapping(rec["geometry"]),
                    "properties": rec["properties"],
                })
            feature_collection = {
                "type":     "FeatureCollection",
                "name":     f"clusters_thr_{int(thr*100):02d}",
                "crs":      {"type": "name",
                             "properties": {"name": crs_name}},
                "features": features,
            }
            with open(geojson_path, "w", encoding="utf-8") as fh:
                json.dump(feature_collection, fh, ensure_ascii=False)
            print(f"    -> {geojson_path}  ({len(records)} polygona)")

        for rec in records:
            summary_records.append(rec["properties"])

    # ----- Save summary CSV -----
    if not summary_records:
        print("\nNema clustera ni na jednom pragu. Provjeri probability raster.")
        return

    summary_df = pd.DataFrame(summary_records)
    summary_path = os.path.join(OUT_DIR, "clusters_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"\n[4] Summary CSV -> {summary_path}  ({len(summary_df)} record)")

    # ----- Top 10 per threshold per metric -----
    print("\n[5] Top 10 per threshold:")
    top_text_lines = []

    cols_top = ["cluster_id", "area_km2", "prob_max", "prob_mean",
                "centroid_x", "centroid_y", "dist_to_known_m",
                "n_known_inside"]

    for thr in THRESHOLDS:
        sub = summary_df[summary_df["threshold"] == thr]
        if len(sub) == 0:
            continue
        top_text_lines.append(f"\n{'='*60}")
        top_text_lines.append(f"THRESHOLD = {thr}    n_clustera = {len(sub)}")
        top_text_lines.append(f"  ukupna povrsina svih clustera = {sub['area_km2'].sum():.2f} km2")
        top_text_lines.append(f"  prosjek prob_max  = {sub['prob_max'].mean():.3f}")
        top_text_lines.append(f"  prosjek dist_to_known = {sub['dist_to_known_m'].mean():.0f} m")
        top_text_lines.append(f"  cluster-a koji sadrze poznati neolitik = "
                              f"{(sub['contains_known']==1).sum()} / {len(sub)}")
        top_text_lines.append(f"{'='*60}")

        top_size = sub.sort_values("score_size", ascending=False).head(10)
        top_text_lines.append("\nTop 10 po SIZE  (najveci hot-spot-ovi, area x mean_prob):")
        top_text_lines.append(top_size[cols_top].to_string(index=False))

        top_disc = sub.sort_values("score_discovery", ascending=False).head(10)
        top_text_lines.append("\nTop 10 po DISCOVERY  (visoka p i daleko od poznatih):")
        top_text_lines.append(top_disc[cols_top].to_string(index=False))

    top_path = os.path.join(OUT_DIR, "clusters_top10.txt")
    with open(top_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(top_text_lines))
    print(f"  -> {top_path}")

    # Konzolni summary
    print("\n" + "=" * 72)
    print("SAZETAK PO PRAGOVIMA")
    print("=" * 72)
    agg = summary_df.groupby("threshold").agg(
        n_clustera=("cluster_id", "count"),
        total_area_km2=("area_km2", "sum"),
        mean_prob_max=("prob_max", "mean"),
        mean_dist_to_known=("dist_to_known_m", "mean"),
        n_contains_known=("contains_known", "sum"),
    ).round(2)
    print(agg.to_string())

    print(f"\nSve spremljeno u: {OUT_DIR}")


if __name__ == "__main__":
    main()
