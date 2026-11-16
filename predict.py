"""Step 3 — read Raman maps with the saved model: composition and apparent µM of each map.

    python predict.py                      # the test maps and the written-ink map of data/annotations/maps.csv
    python predict.py map1.csv map2.csv    # any map files

Each map is read the way `algorithm.read_map` reads it: prepare → gate → composition →
concentration. Writes to results/predict/:
  maps.csv            one row per map: pixels passing the gate, composition (%), apparent µM
  <map>_pixels.csv    one row per pixel: x, y, gate, PLS shares of DQ, TBZ, THI and bare ink (BLK)
"""
import csv
import os
import sys
import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone

import numpy as np

import algorithm as A
import config

OUT = os.path.join(config.RESULTS, "predict")
COMP = config.ANALYTES


def load_coords(path, n_wn):
    """x, y of the pixels `algorithm.load_map` keeps, in the same order."""
    wn, _cube, xy = A.load_map(path, return_coords=True)
    if len(wn) != n_wn:
        raise ValueError("Coordinate/spectral channel mismatch")
    return xy


def default_maps():
    return [p for p, _lab, _s in config.maps("test")] + [p for p, _lab, _s in config.maps("lettering")]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read Raman maps with a portable or legacy PLS model")
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--model", default=config.MODEL)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    paths = args.paths or default_maps()
    if not paths:
        parser.error("No input maps selected")
    model = config.load_reader_model(args.model)
    templates = model.get("templates")
    if templates is None:
        if config.raw_root() is None:
            parser.error("Model has no gate templates. Set SERS_DATA and run bundle_model.py once, then use --model bundle.npz")
        templates = A.class_templates(config.templates())
    parent = os.path.join(config.RESULTS, "predict_runs")
    if args.output_dir:
        out = os.path.abspath(args.output_dir)
        os.makedirs(out, exist_ok=False)
    else:
        os.makedirs(parent, exist_ok=True)
        out = tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_"), dir=parent)
    outputs = list(model["outputs"])
    summary, inputs, failures = [], [], 0
    seen = set()
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        identity = os.path.normcase(os.path.abspath(path))
        if identity in seen:
            continue
        seen.add(identity)
        file_id = name + "_" + hashlib.sha256(identity.encode()).hexdigest()[:12]
        row = {"map": name, "input": os.path.abspath(path), "status": "failed", "error": "",
               "pixel_file": "", "gate_px": "", "n_px": ""}
        row.update({f"{s}_{unit}": "" for unit in ("pct", "uM") for s in COMP})
        record = {"path": os.path.abspath(path)}
        try:
            record["sha256"] = config.file_hash(path)
            wn, cube, xy = A.load_map(path, return_coords=True)
            r = A.read_map(wn, cube, model, templates)
            if not (len(xy) == len(r["gate"]) == len(r["pixel_shares"])):
                raise ValueError("Coordinate, gate and prediction counts differ")
            pixel_file = f"{file_id}_pixels.csv"
            with open(os.path.join(out, pixel_file + ".tmp"), "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["x", "y", "gate"] + outputs)
                for (x, y), g, s in zip(xy, r["gate"], r["pixel_shares"]):
                    w.writerow([f"{x:g}", f"{y:g}", int(g)] + [f"{v:.4f}" for v in s])
            os.replace(os.path.join(out, pixel_file + ".tmp"), os.path.join(out, pixel_file))
            row.update(status="completed" if r["composition"] is not None else "no_analyte",
                       pixel_file=pixel_file, gate_px=int(r["gate"].sum()), n_px=len(r["gate"]))
            for s in COMP:
                row[f"{s}_pct"] = "" if r["composition"] is None else round(100 * float(r["composition"][s]), 2)
                row[f"{s}_uM"] = "" if r["uM"] is None else round(float(r["uM"][s]), 3)
        except (OSError, ValueError, RuntimeError) as exc:
            failures += 1
            row["error"] = str(exc)
        inputs.append(record)
        summary.append(row)
        print(f"{name}: {row['status']} {row['error']}", flush=True)
    with open(os.path.join(out, "maps.csv.tmp"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    os.replace(os.path.join(out, "maps.csv.tmp"), os.path.join(out, "maps.csv"))
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"model": os.path.abspath(args.model), "model_sha256": config.file_hash(args.model),
                   "inputs": inputs, "settings": A.pipeline_settings(),
                   "runtime": config.runtime_versions(), "algorithm_sha256": config.file_hash(A.__file__),
                   "template_sha256": hashlib.sha256(np.asarray(templates["wn"]).tobytes()
                       + np.asarray(list(templates["spectra"].values())).tobytes()
                       + json.dumps(list(templates["spectra"])).encode()).hexdigest(),
                   "model_provenance": model.get("provenance", {}),
                   "failures": failures}, fh, indent=2)
    print("saved:", os.path.abspath(out))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
