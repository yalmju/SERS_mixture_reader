"""Paths and the map list shared by every script.

The measured maps are not stored in this repository (see data/README.md). Point to
the folder that holds them in one of two ways:

    export SERS_DATA=/path/to/data          # environment variable, or
    ln -s /path/to/data data/raw            # a link inside this repository

`data/annotations/maps.csv` lists every map used, relative to that folder.
"""
import csv
import os
import hashlib
import json
import tempfile
import sys

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


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def annotation_hash():
    rows = [r for r in _rows() if r["split"] == "fit"]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def raw_root():
    return next((p for p in (os.environ.get("SERS_DATA"), os.path.join(ROOT, "data", "raw"))
                 if p and os.path.isdir(p)), None)


def runtime_versions():
    import scipy, sklearn
    return {"python": sys.version.split()[0], "platform": sys.platform,
            "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}


def cache_context():
    import algorithm as A
    return {"annotation_sha256": annotation_hash(), "algorithm_sha256": file_hash(A.__file__),
            "settings": A.pipeline_settings(),
            "versions": runtime_versions()}


def validate_table(t):
    import algorithm as A
    required = {"wn", "X", "Y", "map", "paths", "uM", "group", "analytes"}
    if not required <= t.keys():
        raise ValueError("Training table is missing required arrays")
    A.validate_spectra(t["wn"], t["X"])
    n, m = len(t["X"]), len(t["paths"])
    if not m or list(t["analytes"]) != list(ANALYTES):
        raise ValueError("Training table has no maps or an incompatible analyte order")
    if (t["Y"].shape != (n, len(ANALYTES) + 1) or t["map"].shape != (n,)
            or t["uM"].shape != (m, len(ANALYTES)) or len(t["group"]) != m):
        raise ValueError("Training table array dimensions do not agree")
    owner = t["map"]
    if not np.issubdtype(owner.dtype, np.integer) or (owner < 0).any() or (owner >= m).any() or len(np.unique(owner)) != m:
        raise ValueError("Invalid training table map indices")
    if not np.isfinite(t["Y"]).all() or (t["Y"] < 0).any() or not np.allclose(t["Y"].sum(axis=1), 1):
        raise ValueError("Invalid training target shares")
    listed = [r["path"].replace("\\", "/") for r in _rows() if r["split"] == "fit"]
    if [str(p).replace("\\", "/") for p in t["paths"]] != listed:
        raise ValueError("Training table paths do not match annotation relative paths")
    return t


def legacy_axis():
    """Frozen axis extracted from the exact published feature table."""
    published = os.path.join(FEATURES, "published_table.json")
    axis = os.path.join(FEATURES, "published_axis.json")
    with open(published, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if file_hash(axis) != manifest["axis_sha256"]:
        raise ValueError("Published legacy axis checksum mismatch")
    with open(axis, encoding="utf-8") as fh:
        return np.asarray(json.load(fh), float)


def load_reader_model(path=MODEL):
    import algorithm as A
    if not str(path).lower().endswith(".npz"):
        with open(os.path.join(FEATURES, "published_table.json"), encoding="utf-8") as fh:
            published = json.load(fh)
        if file_hash(path) != published["model_sha256"]:
            raise ValueError("Unknown legacy pickle: convert it with an explicitly verified reference axis")
    return A.load_model(path, reference_axis=None if str(path).lower().endswith(".npz") else legacy_axis())


def fit_table(progress=print, allow_legacy=False):
    """The pixel table of the 95 fitting maps (data/features/fit_table.npz).

    Rebuilt from the measured maps when the stored table does not match the map list."""
    import algorithm as A
    cache = os.path.join(FEATURES, "fit_table.npz")
    manifest_path = os.path.join(FEATURES, "fit_table.manifest.json")
    context = cache_context()
    root = raw_root()
    sources = None
    if root is not None:
        sources = {r["path"]: file_hash(os.path.join(root, r["path"]))
                   for r in _rows() if r["split"] == "fit"}
    valid, legacy = False, False
    if os.path.isfile(cache):
        digest = file_hash(cache)
        if os.path.isfile(manifest_path):
            with open(manifest_path, encoding="utf-8") as fh:
                manifest = json.load(fh)
            valid = (manifest.get("schema_version") == 1 and manifest.get("context") == context
                     and manifest.get("cache_sha256") == digest
                     and (sources is None or sources == manifest.get("sources")))
        if not valid and allow_legacy and root is None:
            with open(os.path.join(FEATURES, "published_table.json"), encoding="utf-8") as fh:
                published = json.load(fh)
            legacy = digest == published["cache_sha256"] and context["annotation_sha256"] == published["annotation_sha256"]
        if valid or legacy:
            # The published legacy table contains object arrays; verify its hash before loading.
            with np.load(cache, allow_pickle=legacy) as data:
                t = {k: data[k] for k in data.files}
            t["paths"], t["analytes"] = list(t["paths"]), list(t["analytes"])
            validate_table(t)
            if progress:
                progress("Using verified published legacy features (not rebuilt with current preprocessing)" if legacy
                         else "Using verified feature cache" + ("; raw source files unavailable for rechecking" if root is None else ""))
            return t
    if root is None:
        raise SystemExit("Feature cache is missing or incompatible. Set SERS_DATA to rebuild, or use "
                         "--allow-legacy-cache for the exact published table and unchanged annotations.")
    fit_maps = maps("fit")
    t = A.training_table(fit_maps, progress=progress)
    os.makedirs(FEATURES, exist_ok=True)
    t["paths"] = [os.path.relpath(p, root).replace("\\", "/") for p in t["paths"]]
    t["group"] = np.array([str(g).replace(root, "<data>") for g in t["group"]], dtype=str)
    validate_table(t)
    fd, temporary = tempfile.mkstemp(dir=FEATURES, suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez_compressed(fh, **{k: np.asarray(v, dtype=str) if k in ("paths", "group", "analytes") else np.asarray(v)
                                       for k, v in t.items()})
        metadata = {"schema_version": 1, "context": context, "sources": sources,
                    "cache_sha256": file_hash(temporary)}
        os.replace(temporary, cache)
        with open(manifest_path + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, sort_keys=True)
        os.replace(manifest_path + ".tmp", manifest_path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return t
