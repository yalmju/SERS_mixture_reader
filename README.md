# Reading pesticide mixtures on a surface from one SERS map

This repository contains the code and results for reading one Raman map of a SERS ink applied to a
surface. For three pesticides, diquat (DQ), thiabendazole (TBZ) and thiram (THI), the method returns:

1. **Composition**: a PLS regression with 8 latent variables reads every pixel; the map's composition is
   the mean over the pixels that carry analyte.
2. **Apparent concentration**: a PLS regression reads the map's mean spectrum (500–2000 cm⁻¹)
   normalised to the ink band at 2137 cm⁻¹ and returns the µM of each component. It is fitted on maps
   with every component at 3–24 µM.

A non-negative least-squares gate against class-mean spectra picks out the pixels that carry analyte.

The code includes:

- preprocessing of the maps, the gate, and both PLS stages (`algorithm.py`; numpy, scipy and scikit-learn only)
- fitting both stages and comparing them with the saved model (`train.py`)
- held-out reads of every measured mixture map and their scores (`evaluate.py`)
- reading any map with the saved model (`predict.py`)

Held-out results on the 64-condition grid (3, 6, 12, 24 µM of each component), from
`results/evaluate/scores.txt`:

| | |
|---|---|
| composition error | 12.27 %p |
| apparent µM within 2-fold | 95.8 % (184/192), median 1.26× |
| written-ink map (12 µM each, 1:1:1) | DQ : TBZ : THI = 41.2 : 23.1 : 35.7 (`results/predict/maps.csv`) |

## Dataset

The dataset is in `./data` ([`data/README.md`](data/README.md) gives the file formats):

- `raw`: the measured Raman maps (CSV). They are not stored in this repository. Link the data folder
  here, or set `SERS_DATA` to it.
- `annotations`: `maps.csv`, 134 annotation rows covering 122 unique map paths, with split
  (`fit` 95, `test` 20, `template` 18, `lettering` 1), label, applied µM and pixel seed.
  Twelve pure/background maps are used both for fitting and as gate templates.
- `features`: `fit_table.npz`, the preprocessed pixel table of the 95 fitting maps

The outputs of the scripts are in `./results` ([`results/README.md`](results/README.md)):

- `model`: the saved model (`pls_composition_261110.dlm`) and the weights refitted by `train.py` (`refit.npz`)
- `evaluate`: held-out reads of the 102 mixture maps (`heldout_reads.csv`) and their scores (`scores.txt`)
- `predict`: reads of the 20 test maps and the written-ink map, per map (`maps.csv`) and per pixel (`<map>_pixels.csv`)

## Usage

1. Create the environment from `sers_mixture_reader.yml`

   ```bash
   conda env create -f sers_mixture_reader.yml
   conda activate sers-mixture-reader
   ```

2. Refit from the exact published feature table without the measured maps, and optionally compare
   with the original saved model:

   ```bash
   python train.py --allow-legacy-cache --compare --output results/model/refit_portable.npz
   ```

   The opt-in uses checksum-verified historical features; it does **not** rebuild them with the
   revised preprocessing. Annotation edits invalidate this fallback. With `SERS_DATA` available,
   `python train.py --output results/model/refit_portable.npz` builds/verifies the current feature
   cache and packages gate templates. New caches verify annotation, source checksums, code,
   settings and package versions. Without raw maps, source files cannot be rechecked.

   A model trained offline contains weights and its axis but no raw gate templates. Package those
   once while the original template maps are available:

   ```bash
   export SERS_DATA=/path/to/data
   python bundle_model.py --model results/model/refit_portable.npz --output results/model/reader_bundle.npz
   ```

   To package the original published weights instead, omit `--model` in the bundling command.
   Raw gate templates are not included in this repository and cannot be reconstructed from the
   baseline-subtracted feature table. No replacement templates are fabricated.

3. Read every mixture map held out and score the reads (needs the measured maps):

   ```bash
   python evaluate.py --allow-legacy-cache --output-dir results/evaluate_runs/run01
   ```

   Use a fresh output directory for each evaluation. To use a newly generated current cache,
   omit `--allow-legacy-cache`. `--model` selects a model; fitting maps still use the documented
   leave-group/condition-out refits, while test maps use the selected weights.

4. Read a new map with the standalone bundle. This no longer requires `SERS_DATA` or the raw
   training/template folder:

   ```bash
   python predict.py path/to/map.csv --model results/model/reader_bundle.npz --output-dir results/predict_runs/run01
   ```

   Without map arguments, the annotated test and lettering maps are selected (raw data needed).
   Without `--output-dir`, predictions use a new unique directory under `results/predict_runs/`.
   An explicit output directory must not already exist. Each input has a status/error row;
   basename collisions are avoided. Partial failures return exit code 1 after saving the batch
   summary and manifest. Historical `results/predict/` and `results/evaluate/` remain untouched.

## Input and model validation

CSV columns retain their positions: missing interior cells, non-finite values, malformed rows,
empty maps and non-increasing axes are rejected with an error. Trailing empty CSV separators and
fully blank rows are allowed. Axes must match the model/template within 1e-6 cm⁻¹; there is no
automatic interpolation. Concentration reads require channels within 2137 ± 10 cm⁻¹.

Portable NPZ models include the axis, ordered output names, linear weights, preprocessing settings,
provenance and optional gate templates, and are loaded without pickle. The old `refit.npz` is a
historical weights-only artifact, not a portable model: regenerate it with `train.py`. The bundled
legacy `.dlm` is accepted only with its published checksum and recorded reference axis. Load only
trusted legacy pickle files. Existing result tables are historical results, not new evaluations
of the revised preprocessing.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests use synthetic maps for CSV/axis errors, cache invalidation, saturation filtering, model
round-trips, template bundling, independent prediction, output collisions and partial failures.
Full evaluation from measured maps still requires the external raw data.

## Citation

```bibtex
@article{tbd,
  title   = {},
  author  = {},
  journal = {},
  year    = {}
}
```

## License

MIT, see [`LICENSE`](LICENSE).
