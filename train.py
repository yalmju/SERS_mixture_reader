"""Step 1 — fit both stages on the 95 fitting maps and compare with the saved model.

    python train.py

Reads the pixel table of the fitting maps (data/features/fit_table.npz; rebuilt from the maps
listed in data/annotations/maps.csv, split "fit", when it does not match), fits the composition
and concentration stages (`algorithm.fit`), writes the weights to results/model/refit.npz and
prints how far they are from the saved model (results/model/pls_composition_261110.dlm). The
saved model is never changed.
"""
import os

import numpy as np

import algorithm as A
import config


def main():
    table = config.fit_table(progress=lambda s: print("  ", s, flush=True))
    print(f"fitting table: {len(table['paths'])} maps · {len(table['X'])} spectra · {len(table['wn'])} points")
    model = A.fit(table)
    out = os.path.join(config.RESULTS, "model", "refit.npz")
    np.savez(out, composition_coef=model["composition"][0], composition_intercept=model["composition"][1],
             concentration_coef=model["concentration"][0], concentration_intercept=model["concentration"][1],
             analytes=np.array(model["analytes"]), outputs=np.array(model["outputs"]))
    print("saved:", os.path.relpath(out, config.ROOT))

    saved = A.load_model(config.MODEL)
    for stage in ("composition", "concentration"):
        d = max(float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
                for a, b in zip(model[stage], saved[stage]))
        print(f"{stage} weights: largest difference from the saved model {d:.2e}")
    shares = A.pixel_shares(model["composition"], table["X"])
    shares0 = A.pixel_shares(saved["composition"], table["X"])
    print(f"pixel shares of the fitting pixels: largest difference {np.max(np.abs(shares - shares0)):.2e}")


if __name__ == "__main__":
    main()
