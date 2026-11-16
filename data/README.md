# Data

## raw

The measured Raman maps are not stored in this repository. Point the scripts at the folder that holds
them in one of two ways:

```bash
export SERS_DATA=/path/to/data
```

```bash
ln -s /path/to/data data/raw
```

Each map is a CSV file. Row 3 holds the wavenumber axis from the third column on, and every later row is
`X, Y, intensities…` (see `algorithm.load_map`).

| folder | contents |
|---|---|
| `Ratio/260814_mixture_final/` | mixture maps; the file name gives the µM of each component, e.g. `DQ6-TB3-TH12.csv` |
| `Pure/` | pure-substance maps (DQ, TBZ, THI) and background maps (BLK, INK: bare ink; LEAF) |
| `Pest/` | written-ink map, 12 µM each, 1:1:1 (`260812_12 trio_THI_TBZ_DQ.csv`) |

## annotations/maps.csv

| column | meaning |
|---|---|
| `path` | relative to the data folder |
| `split` | `fit` (both stages are fitted on these: 82 mixtures, 6 pure-substance maps, 7 bare-ink maps), `test` (mixtures kept out of fitting), `template` (class-mean spectra for the gate: DQ, TBZ, THI, BLK, INK, LEAF; 3 each) or `lettering` (written-ink map, 12 µM each, 1:1:1) |
| `label` | `mixture`, an analyte (`DQ`, `TBZ`, `THI`), `blank` (bare ink, a fitting map) or a template class |
| `DQ_uM`, `TBZ_uM`, `THI_uM` | applied µM of each component (mixtures) |
| `pixel_seed` | seed for the choice of the 100 representative pixels of a fitting map (`algorithm.representative_pixels`) |

With the seeds, `train.py` reproduces the saved weights no matter where the maps are kept. For mixtures
the labels are the file names. `Pure/LEAF` is used only as a gate template, never for fitting.

## features/fit_table.npz

The preprocessed pixel table of the 95 fitting maps (`algorithm.training_table`): `wn` (wavenumber
axis), `X` (pixel spectra), `Y` (target shares), `map` (map index of each row), `paths` (map paths,
relative to the data folder), `uM`, `group`, `analytes`. `config.fit_table` rebuilds it from the
measured maps when it does not match `annotations/maps.csv`.
