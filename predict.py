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

import numpy as np

import algorithm as A
import config

OUT = os.path.join(config.RESULTS, "predict")
COMP = config.ANALYTES


def load_coords(path, n_wn):
    """x, y of the pixels `algorithm.load_map` keeps, in the same order."""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    xy = []
    for row in rows[3:]:
        cells = [v for v in row if str(v).strip() != ""]
        if len(cells) >= n_wn + 2:
            xy.append([float(cells[0]), float(cells[1])])
    return np.asarray(xy, float)


def default_maps():
    return [p for p, _lab, _s in config.maps("test")] + [p for p, _lab, _s in config.maps("lettering")]


def main(paths):
    model = A.load_model(config.MODEL)
    templates = A.class_templates(config.templates())
    os.makedirs(OUT, exist_ok=True)
    outputs = list(model["outputs"])
    summary = []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        wn, cube = A.load_map(path)
        r = A.read_map(wn, cube, model, templates)
        xy = load_coords(path, len(wn))
        with open(os.path.join(OUT, f"{name}_pixels.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["x", "y", "gate"] + outputs)
            for (x, y), g, s in zip(xy, r["gate"], r["pixel_shares"]):
                w.writerow([f"{x:g}", f"{y:g}", int(g)] + [f"{v:.4f}" for v in s])
        row = {"map": name, "gate_px": int(r["gate"].sum()), "n_px": len(r["gate"])}
        for s in COMP:
            row[f"{s}_pct"] = "" if r["composition"] is None else round(100 * float(r["composition"][s]), 2)
        for s in COMP:
            row[f"{s}_uM"] = "" if r["uM"] is None else round(float(r["uM"][s]), 3)
        summary.append(row)
        comp = "no pixel passed the gate" if r["composition"] is None else \
            "DQ : TBZ : THI = " + " : ".join(f"{row[f'{s}_pct']:.1f}" for s in COMP) + \
            " %  ·  µM " + " / ".join(f"{row[f'{s}_uM']:.2f}" for s in COMP)
        print(f"{name}  gate {row['gate_px']}/{row['n_px']}  {comp}", flush=True)
    with open(os.path.join(OUT, "maps.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print("saved:", os.path.relpath(OUT, config.ROOT))


if __name__ == "__main__":
    main(sys.argv[1:] or default_maps())
