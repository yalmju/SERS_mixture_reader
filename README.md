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
- `annotations`: `maps.csv`, the list of the 134 maps used, with split (`fit` 95, `test` 20,
  `template` 18, `lettering` 1), label, applied µM and pixel seed
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

2. Fit both stages and compare them with the saved model. This uses `data/features/fit_table.npz`, so
   the measured maps are not needed.

   ```bash
   python train.py
   ```

3. Read every mixture map held out and score the reads (needs the measured maps)

   ```bash
   python evaluate.py
   ```

4. Read maps with the saved model: the test maps and the written-ink map, or any map files given

   ```bash
   python predict.py
   python predict.py path/to/map.csv
   ```

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
