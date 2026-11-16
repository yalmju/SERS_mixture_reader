"""Fit portable model weights and optionally compare against the published model.

    python train.py --allow-legacy-cache --compare

Current caches require matching source/context manifests. The exact published historical
features can be used offline with explicit opt-in. The default output is refit_portable.npz;
raw templates are included when SERS_DATA is available. The published model is unchanged.
"""
import os
import argparse
import json

import numpy as np

import algorithm as A
import config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit portable PLS model weights")
    parser.add_argument("--output", default=os.path.join(config.RESULTS, "model", "refit_portable.npz"))
    parser.add_argument("--allow-legacy-cache", action="store_true", help="Use the exact published features without claiming they were rebuilt")
    parser.add_argument("--compare", action="store_true", help="Compare against the original published model")
    args = parser.parse_args(argv)
    table = config.fit_table(progress=lambda s: print("  ", s, flush=True), allow_legacy=args.allow_legacy_cache)
    print(f"fitting table: {len(table['paths'])} maps · {len(table['X'])} spectra · {len(table['wn'])} points")
    model = A.fit(table)
    out = args.output
    templates = A.class_templates(config.templates()) if config.raw_root() else None
    provenance = {"feature_sha256": config.file_hash(os.path.join(config.FEATURES, "fit_table.npz")),
                  "cache_context": config.cache_context(), "templates_included": templates is not None}
    with open(os.path.join(config.FEATURES, "published_table.json"), encoding="utf-8") as fh:
        published = json.load(fh)
    provenance["feature_origin"] = ("published legacy preprocessed features; not rebuilt"
                                    if provenance["feature_sha256"] == published["cache_sha256"]
                                    else "rebuilt with recorded preprocessing and raw source checksums")
    A.save_model(out, model, templates=templates, provenance=provenance)
    print("saved:", os.path.abspath(out))
    if templates is None:
        print("Weights saved without gate templates. Run bundle_model.py with SERS_DATA before standalone prediction.")

    if not args.compare:
        return
    saved = config.load_reader_model()
    for stage in ("composition", "concentration"):
        d = max(float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
                for a, b in zip(model[stage], saved[stage]))
        print(f"{stage} weights: largest difference from the saved model {d:.2e}")
    shares = A.pixel_shares(model["composition"], table["X"])
    shares0 = A.pixel_shares(saved["composition"], table["X"])
    print(f"pixel shares of the fitting pixels: largest difference {np.max(np.abs(shares - shares0)):.2e}")


if __name__ == "__main__":
    main()
