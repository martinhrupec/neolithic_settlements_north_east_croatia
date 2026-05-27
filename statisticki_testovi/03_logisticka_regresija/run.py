"""
Scenarij 3 (dopunska): LOGISTIČKA REGRESIJA — parcijalni učinak vtt
=====================================================================

Pitanje: Je li signal tipa tla (vtt) u Scenariju 2 nezavisan od
nadmorske visine (aps_vis) i udaljenosti od rijeke (dist_rijeka_korig),
ili je samo proxy za te varijable?

Odgovor: 4 logistička modela (neolitik=1 vs nasumicni_ceste=0):

  Null    : samo intercept
  Terrain : intercept + aps_vis_z + dist_rijeka_korig_z
  Soil    : intercept + vtt_dummies (ref=Cambisols)
  Full    : intercept + vtt_dummies + aps_vis_z + dist_rijeka_korig_z

Usporedbe:
  LR(Full vs Terrain) → je li vtt znacajan nakon kontrole terena?
  LR(Full vs Soil)    → je li teren znacajan nakon kontrole vtt?

Implementacija: vlastiti L-BFGS logit (scipy.optimize.minimize),
SE iz Fisher information matrice, Wald z-test, McFadden / Nagelkerke R2.

Output: rezultati_koeficijenti.csv + rezultati_modeli.csv
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, chi2 as chi2_dist

ROOT    = r"c:\Users\Martin\Desktop\skripte_za_diplomski\statisticki_testovi"
MASTER  = os.path.join(ROOT, "master_dataset.csv")
OUT_DIR = os.path.join(ROOT, "03_logisticka_regresija")

os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
#  Logistička regresija — L-BFGS
# ---------------------------------------------------------------------------

def logit_fit(X, y):
    """
    Maksimalna vjerodostojnost za logistički model.
    X : design matrix (n x k), uključuje stupac intercepta (1-ice)
    y : binarna zavisna (0/1)
    Vraća (beta, se, z, p_values, log_likelihood)
    """
    n, k = X.shape
    y = np.asarray(y, dtype=float)

    def neg_ll_grad(beta):
        eta = np.clip(X @ beta, -700, 700)
        p   = 1.0 / (1.0 + np.exp(-eta))
        ll  = np.sum(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))
        g   = X.T @ (y - p)
        return -ll, -g

    res = minimize(neg_ll_grad, np.zeros(k), jac=True, method="L-BFGS-B",
                   options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-9})

    beta = res.x
    ll   = -res.fun

    # Fisher information  → kovarijancijska matrica
    eta = np.clip(X @ beta, -700, 700)
    p   = 1.0 / (1.0 + np.exp(-eta))
    w   = p * (1.0 - p)
    info = X.T @ (w[:, None] * X)

    try:
        cov = np.linalg.inv(info)
        se  = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)

    z  = beta / se
    pv = 2.0 * norm.sf(np.abs(z))
    return beta, se, z, pv, ll


def pseudo_r2(ll_model, ll_null, n):
    """McFadden i Nagelkerke pseudo-R²."""
    mcf = 1.0 - ll_model / ll_null
    cox = 1.0 - np.exp(2.0 * (ll_null - ll_model) / n)
    mx  = 1.0 - np.exp(2.0 * ll_null / n)
    nag = cox / mx if mx > 0 else np.nan
    return float(mcf), float(nag)


def lr_test(ll_full, ll_restricted, df):
    """Likelihood ratio test: LR = 2*(LL_full - LL_restr) ~ chi2(df)."""
    stat = 2.0 * (ll_full - ll_restricted)
    p    = chi2_dist.sf(stat, df)
    return float(stat), float(p)


# ---------------------------------------------------------------------------
#  Glavni program
# ---------------------------------------------------------------------------

def main():
    df = pd.read_csv(MASTER)

    # --- Filtriraj na relevantne grupe ---
    sub = df[df["tip_sloja"].isin(["neolitik", "nasumicni_ceste"])].copy()
    sub["y"] = (sub["tip_sloja"] == "neolitik").astype(float)
    n = len(sub)
    print(f"n = {n}  ({int(sub['y'].sum())} neolitik,  "
          f"{int((1-sub['y']).sum())} nasumicni_ceste)\n")

    # --- Z-standardizacija kontinuiranih prediktora ---
    for col in ["aps_vis", "dist_rijeka_korig"]:
        mu, sd = sub[col].mean(), sub[col].std()
        sub[f"{col}_z"] = (sub[col] - mu) / sd
        print(f"  {col:25s}: mean={mu:.1f}  sd={sd:.1f}")
    print()

    # --- VTT dummies (referentna kategorija = Cambisols) ---
    vtt_col = "vtt_r100"   # primarni radijus
    ref_cat = "Cambisols"
    cats = [c for c in sorted(sub[vtt_col].dropna().unique()) if c != ref_cat]
    for c in cats:
        sub[f"vtt_{c}"] = (sub[vtt_col] == c).astype(float)

    vtt_dummy_cols = [f"vtt_{c}" for c in cats]

    # ---- Design matrice ----
    intercept = np.ones(n)

    X_null    = np.column_stack([intercept])
    X_terrain = np.column_stack([intercept, sub["aps_vis_z"], sub["dist_rijeka_korig_z"]])
    X_soil    = np.column_stack([intercept] + [sub[c] for c in vtt_dummy_cols])
    X_full    = np.column_stack([intercept] + [sub[c] for c in vtt_dummy_cols]
                                + [sub["aps_vis_z"], sub["dist_rijeka_korig_z"]])

    y = sub["y"].values

    # ---- Fitiranje ----
    print("Fitiranje modela...")
    models = {}
    for name, X in [("Null", X_null), ("Terrain", X_terrain),
                    ("Soil", X_soil), ("Full", X_full)]:
        beta, se, z, pv, ll = logit_fit(X, y)
        models[name] = {"beta": beta, "se": se, "z": z, "pv": pv, "ll": ll}
        print(f"  {name:10s}: LL = {ll:.4f}")

    print()
    ll_null = models["Null"]["ll"]

    # ---- Tablica koeficijenata za Full model ----
    var_names_full = (
        ["intercept"] + cats + ["aps_vis_z", "dist_rijeka_korig_z"]
    )
    m = models["Full"]
    coef_rows = []
    for i, vname in enumerate(var_names_full):
        b, s, z, p = m["beta"][i], m["se"][i], m["z"][i], m["pv"][i]
        OR = np.exp(b)
        sig = ("***" if p < 0.001 else
               "**"  if p < 0.01  else
               "*"   if p < 0.05  else
               "."   if p < 0.10  else "")
        coef_rows.append({
            "model":   "Full",
            "varijabla": vname,
            "beta":    round(b,  4),
            "OR":      round(OR, 4),
            "SE":      round(s,  4),
            "z":       round(z,  3),
            "p_value": round(p,  6),
            "sig":     sig,
        })
    coef_df = pd.DataFrame(coef_rows)

    # --- Također Soil model koeficijenti (za usporedbu) ---
    var_names_soil = ["intercept"] + cats
    m2 = models["Soil"]
    for i, vname in enumerate(var_names_soil):
        b, s, z, p = m2["beta"][i], m2["se"][i], m2["z"][i], m2["pv"][i]
        OR = np.exp(b)
        sig = ("***" if p < 0.001 else
               "**"  if p < 0.01  else
               "*"   if p < 0.05  else
               "."   if p < 0.10  else "")
        coef_rows.append({
            "model":   "Soil",
            "varijabla": vname,
            "beta":    round(b,  4),
            "OR":      round(OR, 4),
            "SE":      round(s,  4),
            "z":       round(z,  3),
            "p_value": round(p,  6),
            "sig":     sig,
        })
    coef_df = pd.DataFrame(coef_rows)

    # ---- LR testovi ----
    lr_vtt_after_terrain, p_vtt   = lr_test(models["Full"]["ll"],
                                             models["Terrain"]["ll"],
                                             df=len(vtt_dummy_cols))
    lr_terrain_after_vtt, p_terr  = lr_test(models["Full"]["ll"],
                                             models["Soil"]["ll"],
                                             df=2)

    # ---- Pseudo-R² ----
    model_rows = []
    for name in ["Null", "Terrain", "Soil", "Full"]:
        ll = models[name]["ll"]
        mcf, nag = pseudo_r2(ll, ll_null, n)
        lr_vs_null  = 2 * (ll - ll_null)
        df_vs_null  = {"Null": 0, "Terrain": 2,
                       "Soil": len(vtt_dummy_cols),
                       "Full": len(vtt_dummy_cols) + 2}[name]
        p_vs_null   = chi2_dist.sf(lr_vs_null, df_vs_null) if df_vs_null > 0 else np.nan
        model_rows.append({
            "model":       name,
            "n_params":    {"Null":1,"Terrain":3,"Soil":1+len(vtt_dummy_cols),
                            "Full":1+len(vtt_dummy_cols)+2}[name],
            "log_lik":     round(ll,  4),
            "McFadden_R2": round(mcf, 4),
            "Nagelkerke_R2": round(nag, 4),
            "LR_vs_null":  round(lr_vs_null, 3),
            "df_vs_null":  df_vs_null,
            "p_vs_null":   round(p_vs_null, 6) if not np.isnan(p_vs_null) else np.nan,
        })
    model_df = pd.DataFrame(model_rows)

    # ---- Ispis ----
    print("=" * 65)
    print("USPOREDBA MODELA")
    print("=" * 65)
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.width", 120):
        print(model_df.to_string(index=False))

    print()
    print("=" * 65)
    print("LIKELIHOOD RATIO TESTOVI")
    print("=" * 65)
    print(f"  LR(Full vs Terrain)  ->  je li vtt znacajan | aps_vis + dist_rijeka:")
    print(f"      chi2({len(vtt_dummy_cols)}) = {lr_vtt_after_terrain:.3f},  "
          f"p = {p_vtt:.4g}")
    print()
    print(f"  LR(Full vs Soil)     ->  je li teren znacajan | vtt dummies:")
    print(f"      chi2(2) = {lr_terrain_after_vtt:.3f},  "
          f"p = {p_terr:.4g}")

    print()
    print("=" * 65)
    print(f"KOEFICIJENTI — Full model  (ref. kategorija vtt = {ref_cat})")
    print("=" * 65)
    full_df = coef_df[coef_df.model == "Full"].drop(columns="model")
    print(full_df.to_string(index=False))
    print("\n  OR > 1 = vise u neolitiku vs ref (Cambisols)")
    print("  OR < 1 = manje u neolitiku vs ref (Cambisols)")

    print()
    print("=" * 65)
    print(f"KOEFICIJENTI — Soil model  (ref. kategorija vtt = {ref_cat})")
    print("=" * 65)
    soil_df = coef_df[coef_df.model == "Soil"].drop(columns="model")
    print(soil_df.to_string(index=False))

    # ---- Tumačenje ----
    print()
    print("=" * 65)
    print("INTERPRETACIJA")
    print("=" * 65)

    if p_vtt < 0.05:
        print(f"  [DA]  vtt OSTAJE znacajan nakon kontrole terena (p={p_vtt:.4g})")
        print("        -> signal tla je NEZAVISAN od nadmorske visine i rijeke")
    else:
        print(f"  [NE]  vtt NESTAJE nakon kontrole terena (p={p_vtt:.4g})")
        print("        -> vtt je vjerojatno samo proxy za aps_vis / dist_rijeka")

    if p_terr < 0.05:
        print(f"  [DA]  aps_vis + dist_rijeka OSTAJU znacajni | vtt (p={p_terr:.4g})")
        print("        -> teren objasnjava varijancu nezavisno od tipa tla")
    else:
        print(f"  [NE]  aps_vis + dist_rijeka NESTAJU | vtt (p={p_terr:.4g})")
        print("        -> tlo posreduje terenski signal")

    # ---- Robusnost: ponoviti za r250 ----
    print()
    print("=" * 65)
    print("ROBUSNOST: vtt_r250  (isti model, siri radijus)")
    print("=" * 65)

    vtt_col2 = "vtt_r250"
    cats2    = [c for c in sorted(sub[vtt_col2].dropna().unique()) if c != ref_cat]
    for c in cats2:
        sub[f"vtt250_{c}"] = (sub[vtt_col2] == c).astype(float)

    X_full2 = np.column_stack(
        [intercept] + [sub[f"vtt250_{c}"] for c in cats2]
        + [sub["aps_vis_z"], sub["dist_rijeka_korig_z"]]
    )
    X_soil2 = np.column_stack([intercept] + [sub[f"vtt250_{c}"] for c in cats2])
    X_terr2 = X_terrain

    _, _, _, _, ll_full2  = logit_fit(X_full2, y)
    _, _, _, _, ll_soil2  = logit_fit(X_soil2, y)
    _, _, _, _, ll_terr2  = logit_fit(X_terr2, y)

    lr2, p2 = lr_test(ll_full2, ll_terr2, df=len(cats2))
    print(f"  LR(Full_r250 vs Terrain): chi2({len(cats2)}) = {lr2:.3f},  p = {p2:.4g}")
    if p2 < 0.05:
        print("  -> vtt_r250 takoder znacajan | teren  OK  (konzistentno)")
    else:
        print("  -> vtt_r250 NIJE znacajan | teren  UPOZORENJE")

    # ---- Spremi CSV ----
    coef_df.to_csv(os.path.join(OUT_DIR, "rezultati_koeficijenti.csv"),
                   index=False, encoding="utf-8")
    model_df.to_csv(os.path.join(OUT_DIR, "rezultati_modeli.csv"),
                    index=False, encoding="utf-8")
    print(f"\nSpremi u: {OUT_DIR}")


if __name__ == "__main__":
    main()
