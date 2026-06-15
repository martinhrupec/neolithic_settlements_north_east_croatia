"""
Provjera Drennanovih uvjeta za chi-square goodness-of-fit (scenarij 1).
=====================================================================

Drennan (1996: 197-198) postavlja dva uvjeta za valjanost asimptotske
chi-square aproksimacije:
  (a) nijedna ocekivana frekvencija ne smije biti < 1
  (b) najvise 20 % celija smije imati ocekivanu frekvenciju < 5

Skript za svaku kategorijsku GoF varijablu (aspect_cat4/ew/sn, vtt_rN, sm_rN)
izracuna ocekivane frekvencije TOCNO kao test_categorical u run.py, pa
provjerava oba uvjeta u dvije verzije:
  - BEZ GRUPIRANJA: sve izvorne kategorije (kako bi izgledalo bez prilagodbe)
  - A PRIORI:       nakon run.grupiraj_rijetke (tla <1% povrsine -> 'rijetka_tla'),
                    sto je tocno ono sto scenarij 1 sada racuna za vtt

Output: provjera_drennan.csv + ispis u konzolu.
"""

import os
import sys

import numpy as np
import pandas as pd

# reuse putanja i helpera iz run.py (main() je pod __main__ guardom, ne pokrece se)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run


# ---------------------------------------------------------------------------
#  Pomocne funkcije
# ---------------------------------------------------------------------------

def build_obs_exp(values, bg_props):
    """Izgradi opazene i ocekivane frekvencije identicno kao test_categorical."""
    obs_counts = pd.Series(values).value_counts()
    cats       = [c for c in bg_props if bg_props[c] > 0]
    observed   = np.array([obs_counts.get(c, 0) for c in cats], dtype=float)
    n          = observed.sum()
    expected   = np.array([bg_props[c] * n for c in cats], dtype=float)
    return observed, expected


def drennan_check(expected):
    """Vrati metrike i prolaz/pad za oba Drennanova uvjeta."""
    exp     = np.asarray(expected, dtype=float)
    k       = len(exp)
    n_lt1   = int((exp < 1).sum())
    n_lt5   = int((exp < 5).sum())
    pct_lt5 = 100.0 * n_lt5 / k if k else float("nan")
    return {
        "k":          k,
        "min_exp":    float(exp.min()) if k else float("nan"),
        "n_exp_lt5":  n_lt5,
        "pct_lt5":    pct_lt5,
        "uvjet_a_ok": (n_lt1 == 0),       # (a) nijedna exp < 1
        "uvjet_b_ok": (pct_lt5 <= 20.0),  # (b) <= 20 % celija s exp < 5
    }


# ---------------------------------------------------------------------------
#  Glavna logika
# ---------------------------------------------------------------------------

def main():
    df  = pd.read_csv(run.MASTER)
    neo = df[df.tip_sloja == "neolitik"]

    # --- sastavi listu (varijabla, vrijednosti, bg_proporcije) ---
    variables = []

    variables.append(("aspect_cat4", neo["aspect_cat4"].dropna(),
                      run.aspect_bg_proportions(run.bin_aspect_cat4, ["NE", "SE", "SW", "NW"])))
    variables.append(("aspect_ew", neo["aspect_ew"].dropna(),
                      run.aspect_bg_proportions(run.bin_aspect_ew, ["E", "W"])))
    variables.append(("aspect_sn", neo["aspect_sn"].dropna(),
                      run.aspect_bg_proportions(run.bin_aspect_sn, ["N", "S"])))

    bg_vtt    = pd.read_csv(os.path.join(run.BG_DIR, "background_vtt.csv"), encoding="utf-8-sig")
    vtt_props = dict(zip(bg_vtt["tip_tla"], bg_vtt["n_piksela"] / bg_vtt["n_piksela"].sum()))
    for r in [100, 250, 500, 1000]:
        col  = f"vtt_r{r}"
        vals = neo[col].dropna()
        vals = vals[vals.isin(vtt_props.keys())]
        variables.append((col, vals, vtt_props))

    bg_sm    = pd.read_csv(os.path.join(run.BG_DIR, "background_sm.csv"), encoding="utf-8-sig")
    sm_props = dict(zip(bg_sm["kategorija"], bg_sm["n_piksela"] / bg_sm["n_piksela"].sum()))
    for r in [100, 250, 500, 1000]:
        col = f"sm_r{r}"
        variables.append((col, neo[col].dropna(), sm_props))

    # --- provjeri svaku varijablu: bez grupiranja vs a priori grupiranje ---
    rows = []
    for name, vals, props in variables:
        # bez grupiranja (sve izvorne kategorije)
        _, exp_raw = build_obs_exp(vals, props)
        bez = drennan_check(exp_raw)
        # a priori grupiranje (run.grupiraj_rijetke) - mijenja samo vtt, ostalo no-op
        vals_g, props_g = run.grupiraj_rijetke(vals, props)
        _, exp_grp = build_obs_exp(vals_g, props_g)
        grp = drennan_check(exp_grp)
        rows.append({
            "varijabla":      name,
            "k_bez":          bez["k"],
            "min_exp_bez":    round(bez["min_exp"], 3),
            "lt5_bez":        bez["n_exp_lt5"],
            "pct_lt5_bez":    round(bez["pct_lt5"], 1),
            "a_ok_bez":       bez["uvjet_a_ok"],
            "b_ok_bez":       bez["uvjet_b_ok"],
            "k_grp":          grp["k"],
            "min_exp_grp":    round(grp["min_exp"], 3),
            "lt5_grp":        grp["n_exp_lt5"],
            "pct_lt5_grp":    round(grp["pct_lt5"], 1),
            "a_ok_grp":       grp["uvjet_a_ok"],
            "b_ok_grp":       grp["uvjet_b_ok"],
        })

    out = pd.DataFrame(rows)
    out_csv = os.path.join(run.OUT_DIR, "provjera_drennan.csv")
    out.to_csv(out_csv, index=False, encoding="utf-8")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print("=" * 78)
    print("DRENNANOVI UVJETI ZA CHI-SQUARE GoF (scenarij 1)")
    print("  (a) nijedna ocekivana frekvencija < 1")
    print("  (b) najvise 20% celija s ocekivanom frekvencijom < 5")
    print("=" * 78)
    print("\nBEZ GRUPIRANJA (sve izvorne kategorije):")
    print(out[["varijabla", "k_bez", "min_exp_bez", "lt5_bez",
               "pct_lt5_bez", "a_ok_bez", "b_ok_bez"]].to_string(index=False))

    print("\nA PRIORI GRUPIRANJE (tla <1% -> 'rijetka_tla'; kako scenarij 1 sada racuna):")
    print(out[["varijabla", "k_grp", "min_exp_grp", "lt5_grp",
               "pct_lt5_grp", "a_ok_grp", "b_ok_grp"]].to_string(index=False))

    bez_ok = bool((out["a_ok_bez"] & out["b_ok_bez"]).all())
    grp_ok = bool((out["a_ok_grp"] & out["b_ok_grp"]).all())
    print("\n" + "-" * 78)
    print("SAZETAK (svi testovi):")
    print(f"  BEZ GRUPIRANJA: oba uvjeta {'OK' if bez_ok else 'PADAJU'}")
    print(f"  A PRIORI GRUP.: oba uvjeta {'OK' if grp_ok else 'PADAJU'}")
    if not bez_ok:
        loših = out.loc[~(out["a_ok_bez"] & out["b_ok_bez"]), "varijabla"].tolist()
        print(f"  -> bez grupiranja krse: {', '.join(loših)}")
    if not grp_ok:
        loših = out.loc[~(out["a_ok_grp"] & out["b_ok_grp"]), "varijabla"].tolist()
        print(f"  -> i nakon grupiranja krse: {', '.join(loših)}")
    print("-" * 78)
    print(f"\nCSV spremljen: {out_csv}")


if __name__ == "__main__":
    main()
