"""
00_build_master.py
Sastavlja master_dataset.csv iz svih izvora u deskriptivna/*/csv_output/.

Tri sloja u jednom CSV-u (svaki red = jedna tocka):
  - neolitik          (274 sites; uid = "n{fid}")
  - nasumicni_ceste   (274 random ceste-biased; uid = "cn{fid}")
  - potpuno_nasumicni (274 umjetno generirani; uid = "pn{fid}")

Membership flags (samo neolitik moze imati True u jednoj od tri):
  - samo_rano, samo_kasno, kontinuirano

Usput kopira background_*.csv u statisticki_testovi/background/.
"""

import os
import pandas as pd


ROOT = r"c:\Users\Martin\Desktop\skripte_za_diplomski"
DESK = os.path.join(ROOT, "deskriptivna")
OUT  = os.path.join(ROOT, "statisticki_testovi")

LAYER_FILES = {
    "neolitik":          "neolitik_svi_odredeni",
    "nasumicni_ceste":   "random_ceste_biased",
    "potpuno_nasumicni": "nasumicni_lokaliteti_umjetno_generirani",
}

UID_PREFIX = {
    "neolitik":          "n",
    "nasumicni_ceste":   "cn",
    "potpuno_nasumicni": "pn",
}

REL_VIS_COMBOS = ["100_250", "100_500", "100_1000", "200_500", "200_1000", "500_1000"]
VTT_RADII      = [100, 250, 500, 1000]
GUSTOCA_RADII  = [1000, 2000]

# Klasifikacija tla -> suho/mocvarno
MOCVARNI_TIPOVI = {"Gleysols", "Fluvisols", "Vertisols"}


def _read(path, value_col, new_col, fid_col="fid"):
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df[[fid_col, value_col]].rename(columns={value_col: new_col})


def aspect_cat4(deg):
    if pd.isna(deg) or deg < 0:
        return pd.NA
    d = deg % 360
    if d < 90:  return "NE"
    if d < 180: return "SE"
    if d < 270: return "SW"
    return "NW"


def aspect_ew(deg):
    if pd.isna(deg) or deg < 0:
        return pd.NA
    d = deg % 360
    return "E" if d < 180 else "W"


def aspect_sn(deg):
    if pd.isna(deg) or deg < 0:
        return pd.NA
    d = deg % 360
    return "S" if 90 <= d < 270 else "N"


def vtt_to_sm(tip):
    if pd.isna(tip):
        return pd.NA
    return "Mocvarno" if tip in MOCVARNI_TIPOVI else "Suho"


def build_layer(tip_sloja):
    base   = LAYER_FILES[tip_sloja]
    prefix = UID_PREFIX[tip_sloja]

    df = _read(os.path.join(DESK, "apsolutna_visina_", "csv_output", f"{base}.csv"),
               "elev", "aps_vis")

    for combo in REL_VIS_COMBOS:
        rv = _read(
            os.path.join(DESK, "relativna_visina_", "csv_output", f"{base}_rv_{combo}.csv"),
            f"rel_{combo}m",
            f"rel_vis_{combo}",
        )
        df = df.merge(rv, on="fid", how="outer")

    asp = pd.read_csv(
        os.path.join(DESK, "aspect_", "csv_output", f"{base}.csv"),
        encoding="utf-8-sig",
    )
    # nagib: 250 (raw) = ravno, manje = strmije -> inverziju radimo: 250 - raw,
    # tako da 0 = ravno, vece = strmije
    asp["nagib"] = 250 - asp["nagib"]
    df = df.merge(asp[["fid", "aspect", "nagib"]], on="fid", how="outer")
    df["aspect_cat4"] = df["aspect"].apply(aspect_cat4)
    df["aspect_ew"]   = df["aspect"].apply(aspect_ew)
    df["aspect_sn"]   = df["aspect"].apply(aspect_sn)

    cfrag = _read(os.path.join(DESK, "coarse_fragments_", "csv_output", f"{base}.csv"),
                  "c_frag", "coarse_fragments")
    # SoilGrids: cm^3/dm^3 -> vol%  (kao u res_coarse_fragments.py)
    cfrag["coarse_fragments"] = cfrag["coarse_fragments"] / 10
    df = df.merge(cfrag, on="fid", how="outer")

    for r in VTT_RADII:
        vtt = _read(
            os.path.join(DESK, "vecinski_tip_tla_", "csv_output", f"{base}_vtt_r{r}.csv"),
            "vecinski_tip_tla",
            f"vtt_r{r}",
        )
        df = df.merge(vtt, on="fid", how="outer")
        df[f"sm_r{r}"] = df[f"vtt_r{r}"].apply(vtt_to_sm)

    df = df.merge(
        _read(os.path.join(DESK, "dist_rijeka_", "csv_output", f"{base}.csv"),
              "dist_rijeka", "dist_rijeka"),
        on="fid", how="outer",
    )
    df = df.merge(
        _read(os.path.join(DESK, "dist_rijeka_korig_", "csv_output", f"{base}.csv"),
              "dist_rijeka_korig", "dist_rijeka_korig"),
        on="fid", how="outer",
    )

    for r in GUSTOCA_RADII:
        df = df.merge(
            _read(
                os.path.join(DESK, "gustoca_rijeka_", "csv_output", f"{base}_gr_{r}.csv"),
                "gustoca_km_km2",
                f"gustoca_rijeka_{r}",
            ),
            on="fid", how="outer",
        )

    df = df.merge(
        _read(os.path.join(DESK, "strahler_", "csv_output", f"{base}.csv"),
              "strahler", "strahler"),
        on="fid", how="outer",
    )
    df = df.merge(
        _read(os.path.join(DESK, "tri_", "csv_output", f"{base}.csv"),
              "tri", "tri"),
        on="fid", how="outer",
    )

    df["tip_sloja"] = tip_sloja
    if tip_sloja == "neolitik":
        # tri faze su DISJUNKTNI podskupovi neolitik_svi_odredeni (83+137+54=274)
        src = os.path.join(DESK, "dist_rijeka_", "csv_output")
        rani = set(pd.read_csv(os.path.join(src, "samo_rani.csv"))["fid"])
        kasn = set(pd.read_csv(os.path.join(src, "samo_srednji_kasni.csv"))["fid"])
        kont = set(pd.read_csv(os.path.join(src, "kontinuirana_naselja.csv"))["fid"])
        df["samo_rano"]    = df["fid"].isin(rani)
        df["samo_kasno"]   = df["fid"].isin(kasn)
        df["kontinuirano"] = df["fid"].isin(kont)
    else:
        df["samo_rano"]    = False
        df["samo_kasno"]   = False
        df["kontinuirano"] = False

    df["uid"]     = prefix + df["fid"].astype(int).astype(str)
    df["fid_raw"] = df["fid"].astype(int)

    return df[[
        "uid", "fid_raw", "tip_sloja",
        "samo_rano", "samo_kasno", "kontinuirano",
        "aps_vis",
        "rel_vis_100_250", "rel_vis_100_500", "rel_vis_100_1000",
        "rel_vis_200_500", "rel_vis_200_1000", "rel_vis_500_1000",
        "aspect", "aspect_cat4", "aspect_ew", "aspect_sn", "nagib",
        "coarse_fragments",
        "vtt_r100", "vtt_r250", "vtt_r500", "vtt_r1000",
        "sm_r100",  "sm_r250",  "sm_r500",  "sm_r1000",
        "dist_rijeka", "dist_rijeka_korig",
        "gustoca_rijeka_1000", "gustoca_rijeka_2000",
        "strahler", "tri",
    ]]


def centralize_backgrounds():
    bg_dir = os.path.join(OUT, "background")
    os.makedirs(bg_dir, exist_ok=True)
    sources = [
        ("apsolutna_visina_",  "background_elev.csv"),
        ("aspect_",            "background_aspect.csv"),
        ("coarse_fragments_",  "background_cfrag.csv"),
        ("gustoca_rijeka_",    "background_gustoca.csv"),
        ("strahler_",          "background_strahler.csv"),
        ("tri_",               "background_tri.csv"),
        ("vecinski_tip_tla_",  "background_sm.csv"),
        ("vecinski_tip_tla_",  "background_vtt.csv"),
    ]
    for folder, fname in sources:
        src = os.path.join(DESK, folder, "csv_output", fname)
        if not os.path.exists(src):
            print(f"  PRESKACEM (ne postoji): {fname}")
            continue
        with open(src, "rb") as fi, open(os.path.join(bg_dir, fname), "wb") as fo:
            fo.write(fi.read())
        print(f"  kopirano: {fname}")


def main():
    os.makedirs(OUT, exist_ok=True)

    print("=== centralizacija background fajlova ===")
    centralize_backgrounds()

    print("\n=== sastavljanje master_dataset ===")
    frames = []
    for tip in LAYER_FILES:
        print(f"  ucitavam: {tip}")
        frames.append(build_layer(tip))
    master = pd.concat(frames, ignore_index=True)

    out_path = os.path.join(OUT, "master_dataset.csv")
    master.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nGOTOVO! {len(master)} redaka  ->  {out_path}")

    print("\npregled tip_sloja:")
    print(master["tip_sloja"].value_counts().to_string())

    n = master[master.tip_sloja == "neolitik"]
    print(f"\nneolitik membership (od ukupno {len(n)}):")
    print(f"  samo_rano:    {int(n.samo_rano.sum())}")
    print(f"  samo_kasno:   {int(n.samo_kasno.sum())}")
    print(f"  kontinuirano: {int(n.kontinuirano.sum())}")
    overlap = n[n[["samo_rano", "samo_kasno", "kontinuirano"]].sum(axis=1) > 1]
    if len(overlap):
        print(f"  UPOZORENJE: {len(overlap)} naselja ima vise od jedne faze postavljene")


if __name__ == "__main__":
    main()
