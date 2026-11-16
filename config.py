"""Paths and the map list shared by every script.

The measured maps are not stored in this repository (see data/README.md). Point to
the folder that holds them in one of two ways:

    export SERS_DATA=/path/to/data          # environment variable, or
    ln -s /path/to/data data/raw            # a link inside this repository

`data/annotations/maps.csv` lists every map used, relative to that folder.
"""
import csv
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
MAP_LIST = os.path.join(ROOT, "data", "annotations", "maps.csv")
MODEL = os.path.join(ROOT, "results", "model", "pls_composition_261110.dlm")
RESULTS = os.path.join(ROOT, "results")
FEATURES = os.path.join(ROOT, "data", "features")    # pixel table of the fitting maps

ANALYTES = ("DQ", "TBZ", "THI")


def data_root():
    for p in (os.environ.get("SERS_DATA"), os.path.join(ROOT, "data", "raw")):
        if p and os.path.isdir(p):
            return p
    raise SystemExit("The measured maps were not found. Set SERS_DATA or link data/raw "
                     "(see data/README.md).")


def _rows():
    with open(MAP_LIST, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def maps(split):
    """Maps of one split as (path, label, pixel_seed), in the order of data/annotations/maps.csv.

    split  "fit"       the 95 maps both stages are fitted on
           "test"      the 20 mixture maps kept out of fitting
           "lettering" the written-ink map (12 µM each, 1:1:1)
    label  {analyte: µM} for a mixture, an analyte name for a pure-substance map,
           None for a bare-ink map (the form `algorithm.training_table` takes)
    pixel_seed  the seed of the representative-pixel choice (fitting maps only, else None)"""
    root = data_root()
    out = []
    for r in _rows():
        if r["split"] != split:
            continue
        if r["label"] == "mixture":
            label = {a: float(r[f"{a}_uM"]) for a in ANALYTES}
        elif r["label"] in ANALYTES:
            label = r["label"]
        else:
            label = None
        seed = int(r["pixel_seed"]) if r.get("pixel_seed") else None
        out.append((os.path.join(root, r["path"]), label, seed))
    return out


def templates():
    """Class → map paths for the gate's class-mean spectra (DQ, TBZ, THI, BLK, INK, LEAF)."""
    root, out = data_root(), {}
    for r in _rows():
        if r["split"] == "template":
            out.setdefault(r["label"], []).append(os.path.join(root, r["path"]))
    return out


def fit_table(progress=print):
    """The pixel table of the 95 fitting maps (data/features/fit_table.npz).

    Rebuilt from the measured maps when the stored table does not match the map list."""
    import algorithm as A
    cache = os.path.join(FEATURES, "fit_table.npz")
    if os.path.isfile(cache):
        t = dict(np.load(cache, allow_pickle=True))
        t["paths"], t["analytes"] = list(t["paths"]), list(t["analytes"])
        listed = [r["path"] for r in _rows() if r["split"] == "fit"]
        if [os.path.basename(p) for p in t["paths"]] == [os.path.basename(p) for p in listed]:
            return t
    fit_maps = maps("fit")
    t = A.training_table(fit_maps, progress=progress)
    os.makedirs(FEATURES, exist_ok=True)
    np.savez(cache, **{k: np.asarray(v, object) if k in ("paths", "group") else np.asarray(v)
                       for k, v in t.items()})
    return t
