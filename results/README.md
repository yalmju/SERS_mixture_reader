# Results

## model/

| file | made by | contents |
|---|---|---|
| `pls_composition_261110.dlm` | — | the saved model (a pickle): composition PLS (`sk`, `n_feat`, `subs`) and concentration weights (`uM`) |
| `refit.npz` | `train.py` | weights of both stages refitted from `data/features/fit_table.npz` |

## evaluate/ (`evaluate.py`)

Every mixture map is read from its file the way any map is read. Fitting maps are read with weights
refitted without that map's composition (and, for the concentration stage, without its condition);
test maps are read with the saved model.

| file | contents |
|---|---|
| `heldout_reads.csv` | one row per mixture map (102) |
| `scores.txt` | composition error (%p) and fold error of the apparent µM over the 64-condition grid |

`heldout_reads.csv` columns:

| column | meaning |
|---|---|
| `map` | map name (the applied µM of each component) |
| `split` | `loo` (fitting map, read held out) or `test` (map kept out of fitting) |
| `DQ_true_uM`, `TBZ_true_uM`, `THI_true_uM` | applied µM |
| `DQ_pct`, `TBZ_pct`, `THI_pct` | composition read (%, sums to 100) |
| `DQ_uM`, `TBZ_uM`, `THI_uM` | apparent µM read; only for maps with every component at 3–24 µM |
| `gate_px`, `n_px` | pixels passing the gate, pixels in the map |

## predict/ (`predict.py`)

Reads with the saved model of the 20 test maps and the written-ink map (`260812_12 trio_THI_TBZ_DQ`).

| file | contents |
|---|---|
| `maps.csv` | one row per map: `gate_px`, `n_px`, composition `DQ_pct`, `TBZ_pct`, `THI_pct`, apparent µM `DQ_uM`, `TBZ_uM`, `THI_uM` |
| `<map>_pixels.csv` | one row per pixel: `x`, `y`, `gate` (1 = passes), PLS shares `DQ`, `TBZ`, `THI`, `BLK` (bare ink; the four sum to 1) |
