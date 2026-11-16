"""Step 2 — read every measured mixture map held out, from its file, and score the reads.

    python evaluate.py

Each map is read the way any map is read (`algorithm`: prepare → gate → composition →
concentration):
  fitting maps  weights refitted without that map's composition (every map of the same
                ratio left out) and, for the concentration stage, without its condition
  test maps     the saved model
Apparent µM is reported only for maps whose every component is at 3–24 µM.

Writes to results/evaluate/:
  heldout_reads.csv  one row per mixture map (102): split, applied µM, composition read (%),
                     apparent µM read (maps inside 3–24 µM only), pixels passing the gate
  scores.txt         composition error and fold error of the reads
"""
import csv
import os
import argparse
import json

import numpy as np

import algorithm as A
import config

OUT = os.path.join(config.RESULTS, "evaluate")
GRID = {3.0, 6.0, 12.0, 24.0}
COMP = config.ANALYTES


def held_out_reads(table, saved, templates):
    A.require_axis(table["wn"], saved["wn"], "evaluation feature table")
    cols = [saved["analytes"].index(s) for s in COMP]
    rows, window = A.map_rows(table), A.in_window(table["uM"])
    cond = np.array([",".join(f"{v:g}" for v in t) for t in table["uM"]], object)
    group_px = table["group"][table["map"]]
    w_comp, w_conc = {}, {}

    def without(i):
        g = table["group"][i]
        if g not in w_comp:
            w_comp[g] = A.fit_composition(table["X"][group_px != g], table["Y"][group_px != g])
        conc = saved["concentration"]             # maps outside the window never entered the fit
        if window[i]:
            if cond[i] not in w_conc:
                keep = window & (cond != cond[i])
                w_conc[cond[i]] = A.fit_concentration(table["wn"], rows[keep], table["uM"][keep])
            conc = w_conc[cond[i]]
        return dict(saved, composition=w_comp[g], concentration=conc)

    jobs = [(p, lab, i) for i, (p, lab, _s) in enumerate(config.maps("fit")) if isinstance(lab, dict)]
    jobs += [(p, lab, None) for p, lab, _s in config.maps("test")]
    out = {}
    for k, (path, label, i) in enumerate(jobs):
        wn, cube = A.load_map(path)
        ax, X = A.prepare(wn, cube)
        hit = A.gate(wn, cube, templates, saved["analytes"])
        r = A.read_pixels(ax, X, hit, saved if i is None else without(i))
        if r["composition"] is None:
            raise SystemExit(f"no pixel passed the gate: {path}")
        true = np.array([label[s] for s in COMP])
        name = os.path.splitext(os.path.relpath(path, config.data_root()))[0].replace("\\", "/")
        out[name] = {"share": np.array([r["composition"][s] for s in saved["analytes"]])[cols],
                     "uM": np.array([r["uM"][s] for s in saved["analytes"]])[cols]
                     if A.in_window(true[None])[0] else None,
                     "gate": int(hit.sum()), "n_px": len(hit), "split": "test" if i is None else "loo",
                     "true": true}
        if k % 10 == 0:
            print(f"[{k + 1}/{len(jobs)}] {name}  gate {int(hit.sum())}/{len(hit)}"
                  f"  composition {np.round(out[name]['share'] * 100).astype(int)}", flush=True)
    return out


def write_reads(reads):
    path = os.path.join(OUT, "heldout_reads.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["map", "split"] + [f"{c}_true_uM" for c in COMP] + [f"{c}_pct" for c in COMP]
                   + [f"{c}_uM" for c in COMP] + ["gate_px", "n_px"])
        for name in sorted(reads, key=lambda n: (tuple(reads[n]["true"]), n)):
            g = reads[name]
            um = [""] * 3 if g["uM"] is None else [round(float(v), 3) for v in g["uM"]]
            w.writerow([name, g["split"]] + [f"{v:g}" for v in g["true"]]
                       + [round(100 * float(v), 2) for v in g["share"]] + um + [g["gate"], g["n_px"]])
    print("saved:", os.path.abspath(path))


def score(reads):
    names = sorted(reads)
    if not names:
        raise ValueError("No evaluated maps to score")
    T = np.array([reads[n]["true"] for n in names])
    grid = np.array([set(t) <= GRID for t in T])
    test = np.array([reads[n]["split"] == "test" for n in names])
    lines = []
    err = A.composition_error([reads[n]["share"] for n in names], T / T.sum(axis=1, keepdims=True))
    mean = lambda a: f"{np.mean(a):.2f}" if len(a) else "N/A"
    n_grid = int(grid.sum())
    lines.append(f"composition, {n_grid} conditions: {mean(err[grid])} %p · fitting grid {mean(err[grid & ~test])}"
          f" (n={int((grid & ~test).sum())}) · test grid {mean(err[grid & test])} (n={int((grid & test).sum())})"
          f" · outside the grid {mean(err[~grid])} (n={int((~grid).sum())})")
    if not n_grid:
        lines.append("concentration: N/A (n=0 in the grid)")
        path = os.path.join(OUT, "scores.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines))
        return
    um = np.array([reads[n]["uM"] for n in np.array(names)[grid]])
    fold = A.fold_error(um, T[grid])
    lines.append(f"concentration, {n_grid} conditions: within 2-fold {100 * (fold <= 2).mean():.1f} % ({int((fold <= 2).sum())}/{fold.size})"
          f" · within 1.5-fold {100 * (fold <= 1.5).mean():.1f} % · median {np.median(fold):.2f}×"
          f" · all three within 2-fold {int((fold <= 2).all(axis=1).sum())}/{n_grid}")
    path = os.path.join(OUT, "scores.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("saved:", os.path.abspath(path))


def main(argv=None):
    global OUT
    parser = argparse.ArgumentParser(description="Evaluate held-out maps")
    parser.add_argument("--model", default=config.MODEL)
    parser.add_argument("--allow-legacy-cache", action="store_true")
    parser.add_argument("--output-dir", default=os.path.join(config.RESULTS, "evaluate_runs"))
    args = parser.parse_args(argv)
    saved = config.load_reader_model(args.model)
    templates = saved.get("templates") or A.class_templates(config.templates())
    OUT = args.output_dir
    os.makedirs(OUT, exist_ok=False)
    reads = held_out_reads(config.fit_table(progress=lambda s: print("  ", s, flush=True), allow_legacy=args.allow_legacy_cache), saved, templates)
    write_reads(reads)
    score(reads)
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"model_sha256": config.file_hash(args.model), "runtime": config.runtime_versions(),
                   "algorithm_sha256": config.file_hash(A.__file__),
                   "annotation_sha256": config.file_hash(config.MAP_LIST),
                   "sources": {p: config.file_hash(p) for p, _label, _seed in config.maps("fit") + config.maps("test")},
                   "settings": A.pipeline_settings()}, fh, indent=2)


if __name__ == "__main__":
    main()
