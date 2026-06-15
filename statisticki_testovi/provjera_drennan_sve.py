"""
KONSOLIDIRANA PROVJERA DRENNANOVIH UVJETA ZA SVE HI-KVADRAT TESTOVE
==================================================================
Scenariji 01 (GoF), 02, 04, 06.

Za svaki hi-kvadrat izracun provjerava oba Drennanova (1996: 197-198) uvjeta:
  (a) nijedna ocekivana frekvencija < 1
  (b) najvise 20 % celija s ocekivanom frekvencijom < 5

Skript UVOZI prave run.py module svakog scenarija i koristi njihove helpere,
pa odrazava TOCNO ono sto scenariji stvarno racunaju (a priori grupiranje tla
u 01/02 i spajanje Strahlerovih redova >= 4 u svim scenarijima).

Izlaz - dva CSV-a, oba na hrvatskom (utf-8, ; razdjelnik, decimalni zarez):
  provjera_drennan_zadovoljeni.csv    - testovi koji zadovoljavaju oba uvjeta
  provjera_drennan_nezadovoljeni.csv  - testovi koji ne zadovoljavaju (mali poduzorci)
"""

import os
import importlib.util

import numpy as np
import pandas as pd
from scipy import stats

ROOT = r"c:\Users\Martin\Desktop\skripte_za_diplomski\statisticki_testovi"
MASTER = os.path.join(ROOT, "master_dataset.csv")
OUT_OK  = os.path.join(ROOT, "provjera_drennan_zadovoljeni.csv")
OUT_NOK = os.path.join(ROOT, "provjera_drennan_nezadovoljeni.csv")


def load_run(folder, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, folder, "run.py"))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


r01 = load_run("01_background_vs_neolitik",        "r01")
r02 = load_run("02_random_ceste_bias_vs_neolitik", "r02")
r04 = load_run("04_rano_vs_kasno",                 "r04")

STRAHLER_CAP = r04.STRAHLER_CAP  # jedinstveno pravilo (1,2,3,4+) u svim scenarijima


# ---------------------------------------------------------------------------
#  Drennan provjera + ocekivane frekvencije
# ---------------------------------------------------------------------------

def drennan(expected):
    exp   = np.asarray(expected, dtype=float).ravel()
    ncell = exp.size
    n_lt1 = int((exp < 1).sum())
    n_lt5 = int((exp < 5).sum())
    pct   = 100.0 * n_lt5 / ncell if ncell else float("nan")
    return ncell, float(exp.min()), n_lt1, n_lt5, pct, (n_lt1 == 0), (pct <= 20.0)


def make_row(scenarij, varijabla, prilagodba, kategorije, expected, p):
    ncell, mn, n1, n5, pct, a, b = drennan(expected)
    return {
        "scenarij":    scenarij,
        "varijabla":   varijabla,
        "prilagodba":  prilagodba,
        "kategorije":  list(kategorije),
        "broj_celija": ncell,
        "min_exp":     mn,
        "n_lt1":       n1,
        "n_lt5":       n5,
        "pct_lt5":     pct,
        "a":           a,
        "b":           b,
        "oba":         a and b,
        "p":           float(p),
    }


def gof_expected(values, props):
    obs      = pd.Series(values).value_counts()
    cats     = [c for c in props if props[c] > 0]
    observed = np.array([obs.get(c, 0) for c in cats], dtype=float)
    n        = observed.sum()
    expected = np.array([props[c] * n for c in cats], dtype=float)
    chi2, p  = stats.chisquare(f_obs=observed, f_exp=expected)
    return cats, expected, p


def cont_expected(a_vals, b_vals):
    a = pd.Series(a_vals).dropna()
    b = pd.Series(b_vals).dropna()
    cats  = sorted(set(a.unique()) | set(b.unique()), key=lambda x: str(x))
    table = np.array([
        [int((a == c).sum()) for c in cats],
        [int((b == c).sum()) for c in cats],
    ], dtype=float)
    keep  = table.sum(axis=0) > 0
    table = table[:, keep]
    cats  = [c for c, k in zip(cats, keep) if k]
    chi2, p, _, exp = stats.chi2_contingency(table)
    return cats, exp, p


# ---------------------------------------------------------------------------
#  Hrvatski citljivi nazivi za prikaz
# ---------------------------------------------------------------------------

SCEN_HR = {
    "01_background":    "Usporedba I: pozadina / neolitik",
    "02_random_ceste":  "Usporedba II: nasumične ceste / neolitik",
    "04_rano_vs_kasno": "Usporedba III: ranoneolitička / kasnoneolitička",
    "06_jedno_vs_kont": "Usporedba IV: jednofazna / kontinuirana",
}


def var_hr(v):
    fiksni = {
        "aspect_cat4": "Ekspozicija (4 smjera)",
        "aspect_ew":   "Ekspozicija (istok / zapad)",
        "aspect_sn":   "Ekspozicija (sjever / jug)",
        "strahler":    "Strahlerov red toka",
    }
    if v in fiksni:
        return fiksni[v]
    if v.startswith("vtt_r"):
        return f"Tip tla (radijus {v[5:]} m)"
    if v.startswith("sm_r"):
        return f"Vlažnost tla (radijus {v[4:]} m)"
    return v


def prilagodba_hr(p):
    if p == "-":
        return "bez prilagodbe"
    if p.startswith("vtt"):
        return "tlo: 4 dominantna tipa + skupina „rijetka tla”"
    if p.startswith("strahler"):
        return "Strahler: redovi ≥ 4 spojeni u jednu klasu"
    return p


def fmt(x, dec=2):
    return f"{x:.{dec}f}".replace(".", ",")


def p_hr(p):
    if p < 0.001:
        return "< 0,001"
    return fmt(p, 3)


def da_ne(x):
    return "DA" if x else "NE"


def presentacija(r):
    kategorije = " | ".join(str(c).replace("rijetka_tla", "rijetka tla") for c in r["kategorije"])
    return {
        "Usporedba":                                          SCEN_HR[r["scenarij"]],
        "Varijabla":                                          var_hr(r["varijabla"]),
        "Prilagodba":                                         prilagodba_hr(r["prilagodba"]),
        "Kategorije (ćelije)":                                kategorije,
        "Broj ćelija":                                        r["broj_celija"],
        "Najmanja očekivana frekvencija":                     fmt(r["min_exp"], 2),
        "Ćelije s očekivanom f. < 1":                         r["n_lt1"],
        "Ćelije s očekivanom f. < 5":                         r["n_lt5"],
        "Postotak ćelija s očekivanom f. < 5":                fmt(r["pct_lt5"], 1),
        "Uvjet (a): nijedna očekivana f. < 1":                da_ne(r["a"]),
        "Uvjet (b): najviše 20 % ćelija s očekivanom f. < 5": da_ne(r["b"]),
        "Oba uvjeta zadovoljena":                             da_ne(r["oba"]),
        "p-vrijednost":                                       p_hr(r["p"]),
        "Statistički značajno (p < 0,05)":                    da_ne(r["p"] < 0.05),
    }


# ---------------------------------------------------------------------------
#  Glavna logika
# ---------------------------------------------------------------------------

def main():
    df  = pd.read_csv(MASTER)
    neo = df[df.tip_sloja == "neolitik"]
    rows = []

    # =================== SCENARIJ 01 (GoF) ===================
    SC = "01_background"
    for var, binner, cats in [("aspect_cat4", r01.bin_aspect_cat4, ["NE", "SE", "SW", "NW"]),
                              ("aspect_ew",   r01.bin_aspect_ew,   ["E", "W"]),
                              ("aspect_sn",   r01.bin_aspect_sn,   ["N", "S"])]:
        props = r01.aspect_bg_proportions(binner, cats)
        c, e, p = gof_expected(neo[var].dropna(), props)
        rows.append(make_row(SC, var, "-", c, e, p))

    bg_vtt    = pd.read_csv(os.path.join(r01.BG_DIR, "background_vtt.csv"), encoding="utf-8-sig")
    vtt_props = dict(zip(bg_vtt["tip_tla"], bg_vtt["n_piksela"] / bg_vtt["n_piksela"].sum()))
    for r in [100, 250, 500, 1000]:
        vals = neo[f"vtt_r{r}"].dropna()
        vals = vals[vals.isin(vtt_props.keys())]
        vg, pg = r01.grupiraj_rijetke(vals, vtt_props)
        c, e, p = gof_expected(vg, pg)
        rows.append(make_row(SC, f"vtt_r{r}", "vtt", c, e, p))

    bg_sm    = pd.read_csv(os.path.join(r01.BG_DIR, "background_sm.csv"), encoding="utf-8-sig")
    sm_props = dict(zip(bg_sm["kategorija"], bg_sm["n_piksela"] / bg_sm["n_piksela"].sum()))
    for r in [100, 250, 500, 1000]:
        c, e, p = gof_expected(neo[f"sm_r{r}"].dropna(), sm_props)
        rows.append(make_row(SC, f"sm_r{r}", "-", c, e, p))

    # strahler GoF (redovi >= cap spojeni)
    bg_str   = pd.read_csv(os.path.join(r01.BG_DIR, "background_strahler.csv"), encoding="utf-8-sig")
    neo_str  = neo["strahler"].dropna().astype(int).clip(upper=STRAHLER_CAP)
    bg_red   = bg_str["strahler"].astype(int).clip(upper=STRAHLER_CAP)
    obs      = neo_str.value_counts()
    s_cats   = sorted(bg_red.unique())
    observed = np.array([obs.get(c, 0) for c in s_cats], dtype=float)
    n        = observed.sum()
    total_km = bg_str["duljina_km"].sum()
    expected = np.array([bg_str.loc[bg_red == c, "duljina_km"].sum() / total_km * n
                         for c in s_cats], dtype=float)
    mask     = expected > 0
    s_labels = [c for c, m in zip(s_cats, mask) if m]
    chi2, p  = stats.chisquare(f_obs=observed[mask], f_exp=expected[mask])
    rows.append(make_row(SC, "strahler", "strahler", s_labels, expected[mask], p))

    # =================== SCENARIJ 02 / 04 / 06 (2-uzorkovni) ===================
    vtt_glavna = r02.vtt_glavna_kategorije()
    scenariji = [
        ("02_random_ceste", neo, df[df.tip_sloja == "nasumicni_ceste"],
         {"group_vtt": True, "cap_strahler": STRAHLER_CAP}),
        ("04_rano_vs_kasno", neo[neo.samo_rano == True], neo[neo.samo_kasno == True],
         {"cap_strahler": STRAHLER_CAP}),
        ("06_jedno_vs_kont", neo[(neo.samo_rano == True) | (neo.samo_kasno == True)],
         neo[neo.kontinuirano == True], {"cap_strahler": STRAHLER_CAP}),
    ]
    kat_vars = (["aspect_cat4", "aspect_ew", "aspect_sn"]
                + [f"vtt_r{r}" for r in [100, 250, 500, 1000]]
                + [f"sm_r{r}"  for r in [100, 250, 500, 1000]]
                + ["strahler"])

    for label, dfA, dfB, rules in scenariji:
        for var in kat_vars:
            a, b = dfA[var], dfB[var]
            prilagodba = "-"
            if var.startswith("vtt_") and rules.get("group_vtt"):
                a = r02.grupiraj_vtt(pd.Series(a).dropna(), vtt_glavna)
                b = r02.grupiraj_vtt(pd.Series(b).dropna(), vtt_glavna)
                prilagodba = "vtt"
            if var == "strahler" and rules.get("cap_strahler"):
                cap = rules["cap_strahler"]
                a = pd.Series(a).dropna().astype(int).clip(upper=cap)
                b = pd.Series(b).dropna().astype(int).clip(upper=cap)
                prilagodba = "strahler"
            c, e, p = cont_expected(a, b)
            rows.append(make_row(label, var, prilagodba, c, e, p))

    # =================== izlaz: dva hrvatska CSV-a ===================
    pres = pd.DataFrame([presentacija(r) for r in rows])
    prolaze    = pres[pres["Oba uvjeta zadovoljena"] == "DA"].reset_index(drop=True)
    ne_prolaze = pres[pres["Oba uvjeta zadovoljena"] == "NE"].reset_index(drop=True)

    prolaze.to_csv(OUT_OK,   index=False, encoding="utf-8-sig", sep=";")
    ne_prolaze.to_csv(OUT_NOK, index=False, encoding="utf-8-sig", sep=";")

    print("=" * 70)
    print("PROVJERA DRENNANOVIH UVJETA - SVI HI-KVADRAT TESTOVI")
    print("=" * 70)
    print(f"Ukupno hi-kvadrat testova:        {len(pres)}")
    print(f"  zadovoljavaju oba uvjeta:       {len(prolaze)}  -> {os.path.basename(OUT_OK)}")
    print(f"  ne zadovoljavaju oba uvjeta:    {len(ne_prolaze)}  -> {os.path.basename(OUT_NOK)}")
    znac_nok = [r for r in rows if not r["oba"] and r["p"] < 0.05]
    print(f"  od nezadovoljenih, ZNACAJNIH:   {len(znac_nok)}")
    if znac_nok:
        print("  PAZNJA: postoje znacajni nalazi koji ne zadovoljavaju (vidi CSV).")
    else:
        print("  -> svi ZNACAJNI hi-kvadrat nalazi zadovoljavaju oba uvjeta.")
    if not ne_prolaze.empty:
        print("\nNe zadovoljavaju (ASCII ispis; puni hrvatski nazivi su u CSV-u):")
        for r in rows:
            if not r["oba"]:
                print(f"  {r['scenarij']:16s} {r['varijabla']:12s}  "
                      f"pct_celija<5={r['pct_lt5']:5.1f}  p={r['p']:.3f}  "
                      f"znacajno={'DA' if r['p'] < 0.05 else 'NE'}")


if __name__ == "__main__":
    main()
