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

There are 134 annotation rows and 122 unique paths; 12 pure/background paths have both `fit` and
`template` roles. These roles are not disjoint file inventories.

## features/fit_table.npz

The preprocessed pixel table of the 95 fitting maps (`algorithm.training_table`): `wn` (wavenumber
axis), `X` (pixel spectra), `Y` (target shares), `map` (map index of each row), `paths` (map paths,
relative to the data folder), `uM`, `group`, `analytes`. `config.fit_table` rebuilds it from the
measured maps when it does not match `annotations/maps.csv`.

The included table predates the revised preprocessing. `published_table.json` identifies its exact
checksum, fitting annotation hash, original model checksum and reference-axis checksum.
`published_axis.json` preserves that axis even if the feature cache is later rebuilt.
Offline use of the historical table requires `train.py --allow-legacy-cache`. It does not certify
that the raw maps were processed by the current code. A changed label, concentration, seed, path or
row order invalidates this opt-in.

Newly built tables use numeric/Unicode arrays and a `fit_table.manifest.json` sidecar with code,
settings, versions, annotation and source checksums. They load without pickle. When raw sources
are unavailable, their current contents cannot be rechecked; the cached data and context still are.

Gate templates store raw class means and axes in portable model bundles. They need the original
`template` maps once: `python bundle_model.py --output results/model/reader_bundle.npz`.
The repository cannot supply those missing measurements; preprocessed fitting spectra are not a
substitute for raw template spectra.
