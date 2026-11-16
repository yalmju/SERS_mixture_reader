"""Reading one Raman map of the ink: composition and apparent concentration.

The whole algorithm in one file; only numpy, scipy and scikit-learn.

    one map
      │  cut to 500–2500 cm⁻¹ · bridge detector plateaus
      │
      ├─ gate             every pixel:  non-negative least squares against the class-mean spectra
      │                   → the pixels that carry analyte; ① and ② read these
      │
      │  remove the ALS baseline · clip at 0
      │
      ├─ ① composition    every pixel:  ln(1 + spectrum) → PLS (8 latent variables)
      │                   → shares of DQ, TBZ, THI and bare ink, clipped at 0, summing to 1
      │                   → mean over the pixels → DQ : TBZ : THI
      │
      └─ ② concentration  the map:  ln(1 + mean spectrum) − ln(1 + ink band at 2137 cm⁻¹),
                          500–2000 cm⁻¹ → PLS (8 latent variables) → log10 C of each component
                          → apparent µM, fitted on maps with every component at 3–24 µM

Use
    model = load_model("results/model/pls_composition_261110.dlm")     # the saved weights
    templates = class_templates({"DQ": [...], "TBZ": [...], "THI": [...], "BLK": [...], ...})
    result = read_map(*load_map("map.csv"), model, templates)
    result["composition"], result["uM"]

Fitting from maps is `training_table` → `fit`; `held_out` scores a stage with whole
conditions left out. `train.py` refits both stages from the maps and compares the weights with the saved
model; `evaluate.py` reads every held-out map from its file.
"""
import csv
import hashlib
import os
import pickle
import json
import tempfile

import numpy as np
from scipy.optimize import nnls
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve
from sklearn.cluster import MiniBatchKMeans
from sklearn.cross_decomposition import PLSRegression

ANALYTES = ("DQ", "TBZ", "THI")
BLANK = "BLK"                      # bare ink: the fourth output of the composition stage
RANGE_CM = (500.0, 2500.0)         # spectral range both stages see
CONC_RANGE_CM = (500.0, 2000.0)    # the concentration stage regresses on the fingerprint part
INK_CM = 2137.0                    # ink band used as the internal reference
BAND_HALF_WIDTH_CM = 10.0          # a band height is the maximum within ± this
N_LATENT = 8                       # PLS latent variables, both stages
WINDOW_UM = (3.0, 24.0)            # the concentration stage is fitted on maps inside this
PIXELS_PER_MAP = 100               # pixels a map contributes to a fit
MAX_CLIPPED = 0.01                 # a pixel with more clipped channels than this is left out of fits
GATE_MIN_SHARE = 0.15              # gate: least-squares analyte share a pixel needs
MIN_ANALYTE_SHARE = 0.5            # ②: pixels whose PLS analyte share is below this are left out
UM_LIMITS = (1e-3, 5e3)            # apparent µM is reported inside these bounds


# ─────────────────────────────────────────────────────────────── map file ──

def validate_axis(wn):
    wn = np.asarray(wn, float)
    if wn.ndim != 1 or len(wn) < 3 or not np.isfinite(wn).all() or not (np.diff(wn) > 0).all():
        raise ValueError("Wavenumbers must be finite, strictly increasing and contain at least 3 channels")
    return wn


def require_axis(actual, expected, context="spectrum"):
    actual, expected = validate_axis(actual), validate_axis(expected)
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0, atol=1e-6):
        raise ValueError(f"{context}: wavenumber axis differs from the reference; resample explicitly")


def validate_spectra(wn, cube):
    wn = validate_axis(wn)
    cube = np.asarray(cube, float)
    if cube.ndim != 2 or not len(cube) or cube.shape[1] != len(wn) or not np.isfinite(cube).all():
        raise ValueError("Spectra must be a nonempty finite pixel-by-channel array matching the axis")
    return wn, cube


def load_map(path, return_coords=False):
    """Map CSV → (wavenumbers, spectra (n_pixels, n_wavenumbers)).

    Row 3 holds the wavenumber axis from the third column on; each later row is
    X, Y, intensities…"""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 4:
        raise ValueError(f"{path}: expected 3 header rows and at least one pixel")
    def trimmed(row):
        row = list(row)
        while row and not row[-1].strip():
            row.pop()
        return row
    try:
        wn = validate_axis([float(v) for v in trimmed(rows[2])[2:]])
    except ValueError as exc:
        raise ValueError(f"{path}: row 3: invalid wavenumber axis: {exc}") from exc
    data, xy = [], []
    for number, row in enumerate(rows[3:], 4):
        row = trimmed(row)
        if not row:
            continue
        if len(row) != len(wn) + 2 or any(not v.strip() for v in row):
            raise ValueError(f"{path}: row {number}: expected X, Y and {len(wn)} intensities without missing cells")
        try:
            values = np.asarray(row, float)
            if not np.isfinite(values).all():
                raise ValueError("non-finite value")
        except ValueError as exc:
            raise ValueError(f"{path}: row {number}: {exc}") from exc
        xy.append(values[:2]); data.append(values[2:])
    wn, cube = validate_spectra(wn, data)
    return (wn, cube, np.asarray(xy)) if return_coords else (wn, cube)


def _between(wn, limits):
    wn = np.asarray(wn, float)
    return (wn >= limits[0]) & (wn <= limits[1])


# ────────────────────────────────────────────────────────── preprocessing ──

_PENALTY = {}


def als_baseline(y, lam=1e5, p=0.01, n_iter=10):
    """Asymmetric least squares baseline (Eilers & Boelens); subtract it from ``y``."""
    n = len(y)
    if (n, lam) not in _PENALTY:
        d = diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(n, n - 2), dtype=float)
        _PENALTY[n, lam] = csc_matrix(lam * (d @ d.T))
    w, z = np.ones(n), y
    for _ in range(n_iter):
        z = spsolve(csc_matrix(diags(w, 0, dtype=float) + _PENALTY[n, lam]), w * y)
        w = np.where(y > z, p, 1 - p)
    return z


def desaturate(spectra, min_run=3, top_frac=0.98):
    """Bridge detector-clipped stretches with a straight line between their shoulders.

    Two signatures: a flat top at the spectrum's own maximum, and a strong block that
    zig-zags from channel to channel (what a plateau looks like after an earlier
    baseline removal). Returns (repaired spectra, fraction of clipped channels per spectrum)."""
    X = np.array(spectra, float, order="C")            # row-contiguous: the pixel choice below sums rows
    n, length = X.shape
    clipped_fraction = np.zeros(n)

    def bridge(y, a, b, top):
        lo = y[a - 1] if a > 0 else y[b + 1] if b + 1 < length else top
        hi = y[b + 1] if b + 1 < length else lo
        y[a:b + 1] = np.linspace(lo, hi, b - a + 1)

    for i in range(n):
        y = X[i]
        top = float(y.max())
        if top <= 0 or top < 2.5 * max(float(np.median(y)), 1e-12):
            continue                                   # no peak towers over the spectrum body
        flat = np.flatnonzero(y >= top * top_frac)
        if len(flat) < min_run:
            continue
        clipped = np.zeros(length, bool)
        for run in np.split(flat, np.where(np.diff(flat) > 3)[0] + 1):
            a, b = int(run[0]), int(run[-1])           # dips of 1–2 channels belong to the plateau
            if b - a + 1 < min_run:
                continue                               # a sharp real peak
            bridge(y, a, b, top)
            clipped[a:b + 1] = True
        top = float(y.max())
        strong = np.flatnonzero(y > 0.5 * top)
        if len(strong) >= min_run + 2:
            for run in np.split(strong, np.where(np.diff(strong) > 1)[0] + 1):
                if len(run) < min_run + 2:
                    continue
                d = np.diff(y[run[0]:run[-1] + 1])
                big = np.abs(d) > 0.12 * top
                if int(((d[1:] * d[:-1] < 0) & big[1:] & big[:-1]).sum()) < 3:
                    continue                           # real bands do not alternate like this
                a, b = int(run[0]), int(run[-1])
                bridge(y, a, b, top)
                clipped[a:b + 1] = True
        clipped_fraction[i] = clipped.mean()
    return X, clipped_fraction


def prepare(wn, cube):
    """Every pixel of a map, as the stages read it: cut to 500–2500 cm⁻¹, plateaus
    bridged, ALS baseline removed, clipped at 0. Returns (wavenumbers, spectra)."""
    wn, cube = validate_spectra(wn, cube)
    keep = _between(wn, RANGE_CM)
    if keep.sum() < 3:
        raise ValueError("At least 3 channels within the preprocessing range are required")
    S = desaturate(np.asarray(cube, float)[:, keep])[0]
    return wn[keep], np.clip(np.stack([y - als_baseline(y) for y in S]), 0, None)


def map_seed(map_id, seed=0):
    """The same map always gets the same representative pixels."""
    key = os.path.normcase(os.path.normpath(str(map_id or "map"))).encode("utf-8", "replace")
    return int(seed) ^ int.from_bytes(hashlib.blake2s(key, digest_size=4).digest(), "little")


def representative_pixels(cube, n_px, seed=0):
    """Indices of ``n_px`` measured pixels that span the map: one per k-means cluster of
    spectral shape (32 bins, unit length) and log total intensity. A map with no more
    than ``n_px`` pixels keeps them all."""
    cube = np.ascontiguousarray(cube, float)
    n = len(cube)
    k = min(max(1, int(n_px)), n)
    if k >= n:
        return np.arange(n)
    bins = np.array_split(np.arange(cube.shape[1]), min(32, cube.shape[1]))
    shape = np.column_stack([cube[:, b].mean(axis=1) for b in bins])
    shape /= np.linalg.norm(shape, axis=1, keepdims=True) + 1e-12
    total = np.log1p(np.clip(cube.sum(axis=1), 0, None))
    z = (total - np.median(total)) / (np.std(total) + 1e-12)
    feat = np.column_stack([shape, 0.35 * z])          # keep this order of operations: the choice
                                                       # of pixels changes with the last digit
    km = MiniBatchKMeans(n_clusters=k, random_state=int(seed), n_init=3, batch_size=min(512, n),
                         max_iter=100, reassignment_ratio=0.0)
    labels = km.fit_predict(feat)
    chosen = []
    for j in range(k):                                 # the pixel nearest each cluster centre
        idx = np.where(labels == j)[0]
        if len(idx):
            chosen.append(int(idx[np.argmin(((feat[idx] - km.cluster_centers_[j]) ** 2).sum(axis=1))]))
    chosen = list(dict.fromkeys(chosen))
    while len(chosen) < k:                             # empty clusters: add the farthest pixel left
        rest = np.setdiff1d(np.arange(n), chosen)
        if not len(rest):
            break
        d2 = ((feat[rest, None, :] - feat[chosen][None, :, :]) ** 2).sum(axis=2).min(axis=1)
        chosen.append(int(rest[np.argmax(d2)]))
    return np.asarray(sorted(chosen, key=lambda i: (-total[i], i)), int)


def training_spectra(wn, cube, map_id, n_px=PIXELS_PER_MAP, seed=0, pixel_seed=None):
    """The pixels a map contributes to a fit: as `prepare`, but pixels with more than 1 %
    clipped channels are left out and at most ``n_px`` representative pixels are kept.

    ``pixel_seed`` fixes the choice of representative pixels; without it the seed is
    derived from ``map_id`` (`map_seed`). data/annotations/maps.csv stores the seed of every
    fitting map so that the choice does not depend on where the maps are kept."""
    wn, cube = validate_spectra(wn, cube)
    keep = _between(wn, RANGE_CM)
    if keep.sum() < 3:
        raise ValueError(f"{map_id}: too few channels in the preprocessing range")
    S, clipped = desaturate(np.asarray(cube, float)[:, keep])
    ok = clipped <= MAX_CLIPPED
    if not ok.any():
        raise ValueError(f"{map_id}: no training pixels remain after saturation filtering")
    S = S[ok]
    s = map_seed(map_id, seed) if pixel_seed is None else int(pixel_seed)
    S = S[representative_pixels(S, n_px, s)[:n_px]]
    return wn[keep], np.clip(np.stack([y - als_baseline(y) for y in S]), 0, None)


# ─────────────────────────────────────────────────────────────────── gate ──

def class_templates(maps_by_class):
    """{class: [map paths]} → {class: mean raw spectrum over every pixel of its maps}.
    The classes are the pure analytes and the backgrounds (bare ink, …)."""
    reference, spectra = None, {}
    for name, paths in maps_by_class.items():
        arrays = []
        for path in paths:
            wn, cube = load_map(path)
            if reference is None:
                reference = wn
            require_axis(wn, reference, str(path))
            arrays.append(cube)
        if not arrays:
            raise ValueError(f"No template maps for {name}")
        spectra[name] = np.vstack(arrays).mean(axis=0)
    if reference is None:
        raise ValueError("No gate templates supplied")
    return {"wn": reference, "spectra": spectra}


def _unit(X):
    X = np.clip(np.atleast_2d(np.asarray(X, float)), 0, None)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def gate(wn, cube, templates, analytes=ANALYTES, min_share=GATE_MIN_SHARE):
    """Which pixels carry analyte. Each raw spectrum (500–2500 cm⁻¹, plateaus bridged,
    unit length, no baseline removal) is fitted by non-negative least squares with the
    class-mean spectra; a pixel passes when the analytes hold at least ``min_share`` of
    the fitted amounts. ``templates`` are on the same wavenumber axis as the map."""
    wn, cube = validate_spectra(wn, cube)
    require_axis(wn, templates["wn"], "gate templates")
    templates = templates["spectra"]
    if any(a not in templates for a in analytes):
        raise ValueError("Gate templates must include every analyte")
    keep = _between(wn, RANGE_CM)
    X = _unit(desaturate(np.asarray(cube, float)[:, keep])[0])
    names = list(templates)
    T = _unit(np.array([templates[n] for n in names], float)[:, keep])
    A = np.array([nnls(T.T, x)[0] for x in X])
    total = A.sum(axis=1)
    analyte = A[:, [n in analytes for n in names]].sum(axis=1)
    return np.divide(analyte, total, out=np.zeros_like(total), where=total > 0) >= min_share


# ──────────────────────────────────────────────────────────── PLS weights ──

def _weights(pls, n_features):
    """A fitted PLS regression as the linear map it is: prediction = x @ B + b0."""
    b0 = np.asarray(pls.predict(np.zeros((1, n_features))), float).reshape(-1)
    B = np.asarray(pls.predict(np.eye(n_features)), float).reshape(n_features, -1) - b0
    return B, b0


# ──────────────────────────────────────────────────────── ① composition ──

def fit_composition(X, Y, n_latent=N_LATENT):
    """PLS regression of the shares ``Y`` (rows sum to 1; analytes, then bare ink) on
    ln(1 + spectrum) of the prepared pixel spectra ``X``. Returns (B, b0)."""
    F = np.log1p(np.clip(X, 0, None)).astype(np.float32)
    if F.ndim != 2 or len(F) < 2 or not F.shape[1] or not np.isfinite(F).all() or not np.isfinite(Y).all():
        raise ValueError("Composition fit requires at least 2 finite samples and nonempty features")
    k = max(1, min(int(n_latent), len(F) - 1, F.shape[1]))
    pls = PLSRegression(n_components=k).fit(F, np.asarray(Y, np.float32))
    return _weights(pls, F.shape[1])


def pixel_shares(weights, X):
    """Prepared pixel spectra → shares per pixel, clipped at 0 and summing to 1."""
    B, b0 = weights
    p = np.clip(np.log1p(np.clip(np.atleast_2d(X), 0, None)) @ B + b0, 0, None)
    return p / (p.sum(axis=1, keepdims=True) + 1e-12)


def map_composition(shares, n_analytes=len(ANALYTES)):
    """Shares of the pixels of one map → the analytes' composition, summing to 1."""
    p = np.clip(np.atleast_2d(shares).mean(axis=0)[:n_analytes], 0, None)
    if not np.isfinite(p).all() or p.sum() <= 1e-9:
        raise ValueError("No valid analyte composition after prediction")
    return p / p.sum()


def pixel_composition(shares, n_analytes=len(ANALYTES)):
    """Shares of each pixel → that pixel's composition of the analytes, summing to 1
    (what a composition image shows pixel by pixel)."""
    p = np.clip(np.atleast_2d(shares)[:, :n_analytes], 0, None)
    return p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)


# ─────────────────────────────────────────────────────── ② concentration ──

def band_height(wn, X, centre, half_width=BAND_HALF_WIDTH_CM):
    """Height of a band per spectrum: the maximum within ± ``half_width`` of ``centre``."""
    wn = np.asarray(wn, float)
    m = np.abs(wn - centre) <= half_width
    if not m.any():
        raise ValueError(f"Required reference band {centre:g} +/- {half_width:g} cm-1 is missing")
    return np.clip(np.atleast_2d(X), 0, None)[:, m].max(axis=1)


def ink_normalised_spectrum(wn, X):
    """Prepared pixel spectra of one map → the one row the concentration stage reads:
    ln(1 + mean spectrum) − ln(1 + mean height of the ink band)."""
    X = np.clip(np.atleast_2d(X), 0, None)
    return np.log1p(X.mean(axis=0)) - np.log1p(band_height(wn, X, INK_CM).mean())


def fit_concentration(wn, F, C_uM, n_latent=N_LATENT):
    """PLS regression of log10 µM (one column per analyte, all > 0) on the map rows ``F``,
    using 500–2000 cm⁻¹ only. Returns (B, b0); B is zero outside that range, so the
    weights apply to the whole row."""
    F = np.asarray(F, float)
    keep = _between(wn, CONC_RANGE_CM)
    C_uM = np.asarray(C_uM, float)
    if F.ndim != 2 or len(F) < 2 or not keep.any() or not np.isfinite(F).all() or not np.isfinite(C_uM).all() or (C_uM <= 0).any():
        raise ValueError("Concentration fit requires at least 2 finite maps and positive concentration labels")
    k = int(min(n_latent, len(F) - 1, keep.sum()))
    pls = PLSRegression(n_components=k, scale=False).fit(F[:, keep], np.log10(np.asarray(C_uM, float)))
    Bk, b0 = _weights(pls, int(keep.sum()))
    B = np.zeros((F.shape[1], len(b0)))
    B[keep] = Bk
    return B, b0


def concentration(weights, F):
    """Map rows → apparent µM of each analyte."""
    B, b0 = weights
    log_um = np.atleast_2d(F) @ B + b0
    if not np.isfinite(log_um).all():
        raise ValueError("Non-finite concentration prediction")
    return 10.0 ** np.clip(log_um, *np.log10(UM_LIMITS))


# ───────────────────────────────────────────────────────── the whole read ──

def pipeline_settings():
    return {"range_cm": list(RANGE_CM), "conc_range_cm": list(CONC_RANGE_CM),
            "ink_cm": INK_CM, "band_half_width_cm": BAND_HALF_WIDTH_CM,
            "max_clipped": MAX_CLIPPED, "gate_min_share": GATE_MIN_SHARE,
            "min_analyte_share": MIN_ANALYTE_SHARE, "um_limits": list(UM_LIMITS),
            "preprocessing_version": 2}


def validate_model(model):
    wn = validate_axis(model["wn"])
    names, outputs = list(model["analytes"]), list(model["outputs"])
    if not names or len(set(outputs)) != len(outputs) or outputs != names + [BLANK]:
        raise ValueError("Model output order must be analytes followed by BLK")
    for stage, width in (("composition", len(outputs)), ("concentration", len(names))):
        B, b = (np.asarray(v, float) for v in model[stage])
        if B.shape != (len(wn), width) or b.shape != (width,) or not np.isfinite(B).all() or not np.isfinite(b).all():
            raise ValueError(f"Invalid {stage} model dimensions or values")
    templates = model.get("templates")
    if templates is not None:
        twn = validate_axis(templates["wn"])
        require_axis(twn[_between(twn, RANGE_CM)], wn, "model templates")
        if any(a not in templates["spectra"] for a in names):
            raise ValueError("Missing analyte gate template")
        for s in templates["spectra"].values():
            validate_spectra(twn, np.atleast_2d(s))
    return model


def save_model(path, model, templates=None, provenance=None):
    """Store portable linear weights, axis, settings and optional raw gate templates."""
    model = dict(model)
    if templates is not None:
        model["templates"] = templates
    validate_model(model)
    arrays = {"schema_version": np.array(1), "wn": model["wn"],
              "analytes": np.array(model["analytes"], dtype=str),
              "outputs": np.array(model["outputs"], dtype=str),
              "settings_json": np.array(json.dumps(pipeline_settings(), sort_keys=True)),
              "provenance_json": np.array(json.dumps(provenance or model.get("provenance", {}), sort_keys=True))}
    for stage in ("composition", "concentration"):
        arrays[f"{stage}_coef"], arrays[f"{stage}_intercept"] = model[stage]
    t = model.get("templates")
    if t is not None:
        arrays.update(template_wn=t["wn"], template_names=np.array(list(t["spectra"]), dtype=str),
                      template_spectra=np.array(list(t["spectra"].values()), dtype=float))
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=parent, suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def load_model(path, reference_axis=None):
    """Portable NPZ, or a trusted legacy pickle with an explicitly verified axis."""
    if str(path).lower().endswith(".npz"):
        with np.load(path, allow_pickle=False) as data:
            if "schema_version" not in data or int(data["schema_version"]) != 1:
                raise ValueError("Unsupported model schema; regenerate the model with train.py")
            if json.loads(str(data["settings_json"])) != pipeline_settings():
                raise ValueError("Model preprocessing settings differ from this algorithm")
            model = {"wn": data["wn"], "analytes": data["analytes"].tolist(),
                     "outputs": data["outputs"].tolist(),
                     "provenance": json.loads(str(data["provenance_json"]))}
            for stage in ("composition", "concentration"):
                model[stage] = (data[f"{stage}_coef"], data[f"{stage}_intercept"])
            if "template_wn" in data:
                model["templates"] = {"wn": data["template_wn"],
                                      "spectra": dict(zip(data["template_names"].tolist(), data["template_spectra"]))}
        return validate_model(model)
    if reference_axis is None:
        raise ValueError("Legacy pickle has no axis; supply its verified reference_axis or convert to a model bundle")
    with open(path, "rb") as fh:
        m = pickle.load(fh)
    conc = np.asarray(m["uM"]["coef"], float)          # last row is the intercept
    return validate_model({"wn": np.asarray(reference_axis, float),
            "analytes": list(m["uM"]["subs"]), "outputs": list(m["subs"]),
            "composition": _weights(m["sk"], int(m["n_feat"])),
            "concentration": (conc[:-1], conc[-1])})


def read_map(wn, cube, model, templates=None):
    """One map → composition and apparent µM.

    ``templates`` are axis-tagged raw class means (see `class_templates`). Bundled
    templates are used by default; without either, this low-level API skips the gate.
    The prediction CLI requires gate templates."""
    ax, X = prepare(wn, cube)
    templates = templates if templates is not None else model.get("templates")
    hit = (gate(wn, cube, templates, model["analytes"]) if templates is not None
           else np.ones(len(X), bool))
    return read_pixels(ax, X, hit, model)


def read_pixels(ax, X, hit, model):
    """The prepared pixels of one map and their gate → composition and apparent µM."""
    names = model["analytes"]
    require_axis(ax, model["wn"], "prediction")
    _, X = validate_spectra(ax, X)
    hit = np.asarray(hit, bool)
    if hit.shape != (len(X),):
        raise ValueError("Gate length differs from pixel count")
    shares = pixel_shares(model["composition"], X)
    if not hit.any():
        return {"composition": None, "uM": None, "pixel_shares": shares, "gate": hit}
    comp = map_composition(shares[hit], len(names))
    use = hit & (shares[:, :len(names)].sum(axis=1) >= MIN_ANALYTE_SHARE)
    if not use.any():
        use = hit
    um = concentration(model["concentration"], ink_normalised_spectrum(ax, X[use]))[0]
    return {"composition": dict(zip(names, comp)), "uM": dict(zip(names, um)),
            "pixel_shares": shares, "gate": hit}


# ───────────────────────────────────────────────────────────────── fitting ──

def training_table(samples, analytes=ANALYTES, progress=None):
    """Maps → the pixel table both stages are fitted on.

    ``samples`` is a list of (path, label) or (path, label, pixel_seed): label is a dict
    {analyte: µM} for a mixture map, an analyte name for a pure-substance map, None for a
    bare-ink map; pixel_seed as in `training_spectra`.

    Returns a dict with
      wn     wavenumbers (500–2500 cm⁻¹)
      X      prepared pixel spectra
      Y      shares per pixel: analytes, then bare ink
      map    index into ``paths`` per pixel
      paths  the maps
      uM     µM per map (nan where the map has no concentration label)
      group  per map, what is held out together: mixtures of one composition share a
             group, every pure or bare-ink map is its own"""
    analytes = list(analytes)
    X, Y, owner, paths, uM, group, ax = [], [], [], [], [], [], None
    for k, (path, label, *pixel_seed) in enumerate(samples):
        if progress:
            progress(f"map {k + 1}/{len(samples)}")
        current_ax, S = training_spectra(*load_map(path), map_id=path,
                                 pixel_seed=pixel_seed[0] if pixel_seed else None)
        if ax is not None:
            require_axis(current_ax, ax, str(path))
        ax = current_ax
        c = np.full(len(analytes), np.nan)
        if isinstance(label, dict):
            c = np.array([float(label.get(a, 0.0)) for a in analytes])
            if not np.isfinite(c).all() or (c < 0).any() or c.sum() <= 0:
                raise ValueError(f"{path}: mixture labels must be finite, nonnegative and have positive sum")
            y = np.r_[c / c.sum(), 0.0]
            g = "r:" + ",".join(f"{v:.6f}" for v in y[:-1])
        else:
            y = np.array([float(a == label) for a in analytes] + [float(label is None)])
            g = path
        X.append(S); Y.append(np.tile(y, (len(S), 1))); owner += [k] * len(S)
        paths.append(path); uM.append(c); group.append(g)
    if not X:
        raise ValueError("No training maps supplied")
    return {"wn": ax, "X": np.vstack(X), "Y": np.vstack(Y), "map": np.asarray(owner),
            "paths": paths, "uM": np.array(uM), "group": np.array(group, object),
            "analytes": analytes}


def map_rows(table):
    """The concentration stage's input row for every map of a training table."""
    return np.array([ink_normalised_spectrum(table["wn"], table["X"][table["map"] == i])
                     for i in range(len(table["paths"]))])


def in_window(uM, window=WINDOW_UM):
    """Maps whose every component lies inside the fitting window."""
    uM = np.asarray(uM, float)
    with np.errstate(invalid="ignore"):
        return ((uM >= window[0]) & (uM <= window[1])).all(axis=1)


def fit(table):
    """Both stages from a training table → a model `read_map` can use."""
    F, ok = map_rows(table), in_window(table["uM"])
    return {"wn": np.asarray(table["wn"]), "analytes": list(table["analytes"]), "outputs": list(table["analytes"]) + [BLANK],
            "composition": fit_composition(table["X"], table["Y"]),
            "concentration": fit_concentration(table["wn"], F[ok], table["uM"][ok])}


def held_out(fit_stage, predict, X, Y, groups):
    """Leave one group out at a time: refit on the rest, predict the group. One prediction per row."""
    groups = np.asarray(groups, object)
    P = np.zeros((len(X), np.shape(Y)[1]))
    for g in dict.fromkeys(groups.tolist()):
        out = groups == g
        P[out] = predict(fit_stage(X[~out], Y[~out]), X[out])
    return P


# ───────────────────────────────────────────────────────────────── scoring ──

def composition_error(read, applied):
    """½ Σ|read − applied| in percentage points: the share placed on the wrong component."""
    return 50.0 * np.abs(np.asarray(read, float) - np.asarray(applied, float)).sum(axis=-1)


def fold_error(read, applied):
    """max(read / applied, applied / read) per value; 2 means off by a factor of two."""
    read, applied = np.asarray(read, float), np.asarray(applied, float)
    return np.maximum(read / applied, applied / read)
