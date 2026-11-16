import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import algorithm as A
import config
import predict
import evaluate
import bundle_model


def write_map(path, wn, spectra, missing=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerows([["header"], ["header"], ["X", "Y", *wn]])
        for i, y in enumerate(spectra):
            row = [i, 0, *y]
            if missing:
                row[4] = ""
            w.writerow(row)


def model_fixture():
    wn = np.arange(500., 2501., 10.)
    spectrum = 10 + 100 * np.exp(-((wn - 1200) / 30) ** 2)
    template = {"wn": wn, "spectra": {"DQ": spectrum, "TBZ": np.ones(len(wn)),
                                       "THI": np.exp(-((wn - 1800) / 50) ** 2)}}
    model = {"wn": wn, "analytes": list(A.ANALYTES), "outputs": [*A.ANALYTES, A.BLANK],
             "composition": (np.zeros((len(wn), 4)), np.array([.2, .3, .5, 0.])),
             "concentration": (np.zeros((len(wn), 3)), np.log10([6., 12., 3.]))}
    return model, template, spectrum


class InputTests(unittest.TestCase):
    def test_csv_keeps_coordinates_and_rejects_missing_cells(self):
        model, _, y = model_fixture()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "map.csv"
            write_map(p, model["wn"], [y, y])
            wn, cube, xy = A.load_map(p, return_coords=True)
            np.testing.assert_array_equal(xy, [[0, 0], [1, 0]])
            self.assertEqual(cube.shape, (2, len(wn)))
            write_map(p, wn, [y], missing=True)
            with self.assertRaisesRegex(ValueError, "row 4"):
                A.load_map(p)

    def test_shifted_axis_rejected_even_with_same_width(self):
        model, _, y = model_fixture()
        with self.assertRaisesRegex(ValueError, "axis differs"):
            A.read_pixels(model["wn"] + .1, [y], [True], model)

    def test_reversed_nonfinite_and_empty_input_rejected(self):
        model, _, y = model_fixture()
        for wn, cube in ((model["wn"][::-1], [y]), (model["wn"], []),
                         (model["wn"], [np.full(len(y), np.nan)])):
            with self.assertRaises(ValueError):
                A.prepare(wn, cube)

    def test_missing_reference_band_rejected(self):
        with self.assertRaisesRegex(ValueError, "reference band"):
            A.band_height(np.arange(500., 2001.), np.ones((1, 1501)), A.INK_CM)

    def test_zero_baseline_is_finite(self):
        np.testing.assert_array_equal(A.als_baseline(np.zeros(50)), np.zeros(50))

    def test_all_saturated_pixels_rejected_and_partial_filtered(self):
        wn = np.arange(500., 2501., 10.)
        cube = np.ones((3, len(wn)))
        with patch.object(A, "desaturate", return_value=(cube, np.ones(3))):
            with self.assertRaisesRegex(ValueError, "no training pixels"):
                A.training_spectra(wn, cube, "bad")
        with patch.object(A, "desaturate", return_value=(cube, np.array([0., .5, 0.]))):
            _, X = A.training_spectra(wn, cube, "partial")
            self.assertEqual(len(X), 2)

    def test_template_axis_mismatch_rejected(self):
        model, templates, y = model_fixture()
        with self.assertRaisesRegex(ValueError, "axis differs"):
            A.gate(model["wn"] + .1, [y], templates)


class ModelTests(unittest.TestCase):
    def test_portable_model_roundtrip_and_standalone_prediction(self):
        model, templates, y = model_fixture()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bundle = root / "model.npz"
            A.save_model(bundle, model, templates=templates)
            loaded = A.load_model(bundle)
            for stage in ("composition", "concentration"):
                for before, after in zip(model[stage], loaded[stage]):
                    np.testing.assert_array_equal(before, after)
            p = root / "sample.csv"
            write_map(p, model["wn"], [y, y])
            with patch.object(config, "templates", side_effect=AssertionError("must not load raw templates")):
                status = predict.main([str(p), "--model", str(bundle), "--output-dir", str(root / "out")])
            self.assertEqual(status, 0)
            rows = list(csv.DictReader((root / "out/maps.csv").read_text(encoding="utf-8").splitlines()))
            self.assertEqual(rows[0]["status"], "completed")
            self.assertAlmostEqual(float(rows[0]["DQ_uM"]), 6.)
            self.assertEqual(json.loads((root / "out/manifest.json").read_text())["failures"], 0)
            with self.assertRaises(FileExistsError):
                predict.main([str(p), "--model", str(bundle), "--output-dir", str(root / "out")])

    def test_batch_preserves_duplicate_basenames_and_records_failure(self):
        model, templates, y = model_fixture()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bundle = root / "model.npz"
            A.save_model(bundle, model, templates=templates)
            paths = [root / "a/map.csv", root / "b/map.csv"]
            for p in paths:
                write_map(p, model["wn"], [y])
            missing = root / "missing.csv"
            self.assertEqual(predict.main([*(str(p) for p in paths), str(missing), "--model", str(bundle),
                                           "--output-dir", str(root / "out")]), 1)
            rows = list(csv.DictReader((root / "out/maps.csv").read_text(encoding="utf-8").splitlines()))
            self.assertEqual([r["status"] for r in rows], ["completed", "completed", "failed"])
            self.assertNotEqual(rows[0]["pixel_file"], rows[1]["pixel_file"])
            self.assertEqual(len(list((root / "out").glob("*_pixels.csv"))), 2)

    def test_model_settings_change_rejected(self):
        model, _, _ = model_fixture()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.npz"
            A.save_model(p, model)
            with patch.object(A, "INK_CM", 2100.):
                with self.assertRaisesRegex(ValueError, "settings differ"):
                    A.load_model(p)

    def test_bundle_command_packages_templates_without_changing_weights(self):
        model, templates, y = model_fixture()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            weights, bundled = root / "weights.npz", root / "bundle.npz"
            A.save_model(weights, model)
            sources = {}
            for name, spectrum in templates["spectra"].items():
                p = root / f"{name}.csv"
                write_map(p, model["wn"], [spectrum])
                sources[name] = [str(p)]
            with patch.object(config, "templates", return_value=sources), patch.object(config, "data_root", return_value=str(root)):
                bundle_model.main(["--model", str(weights), "--output", str(bundled)])
            loaded = A.load_model(bundled)
            np.testing.assert_array_equal(loaded["composition"][0], model["composition"][0])
            self.assertEqual(list(loaded["templates"]["spectra"]), list(templates["spectra"]))
            read = A.read_map(model["wn"], [y], loaded)
            self.assertIsNotNone(read["composition"])

    def test_no_gate_pixels_and_zero_analyte_shares(self):
        model, _, y = model_fixture()
        self.assertIsNone(A.read_pixels(model["wn"], [y], [False], model)["composition"])
        model["composition"][1][:] = 0
        with self.assertRaisesRegex(ValueError, "No valid analyte"):
            A.read_pixels(model["wn"], [y], [True], model)

    def test_fit_save_reload_predictions_equal(self):
        rng = np.random.default_rng(42)
        wn = np.arange(500., 2501., 10.)
        c = np.array([[3., 6., 12.], [6., 12., 3.], [12., 3., 6.], [24., 6., 3.]])
        X = rng.uniform(1, 10, (8, len(wn)))
        table = {"wn": wn, "X": X, "Y": np.repeat(np.c_[c / c.sum(1, keepdims=True), np.zeros(4)], 2, axis=0),
                 "uM": c, "map": np.repeat(np.arange(4), 2), "paths": list("abcd"), "analytes": list(A.ANALYTES)}
        model = A.fit(table)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.npz"
            A.save_model(p, model)
            loaded = A.load_model(p)
            np.testing.assert_allclose(A.pixel_shares(model["composition"], X), A.pixel_shares(loaded["composition"], X))
            np.testing.assert_allclose(A.concentration(model["concentration"], A.map_rows(table)),
                                       A.concentration(loaded["concentration"], A.map_rows(table)))


class CacheTests(unittest.TestCase):
    def test_annotation_and_source_changes_rebuild_cache(self):
        model, _, y = model_fixture()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "features").mkdir()
            raw = root / "raw"
            write_map(raw / "folder/map.csv", model["wn"], [y, y])
            listing = root / "maps.csv"
            def annotations(seed):
                listing.write_text("path,split,label,DQ_uM,TBZ_uM,THI_uM,pixel_seed\n"
                                   f"folder/map.csv,fit,mixture,3,6,12,{seed}\n", encoding="utf-8")
            annotations(42)
            with patch.object(config, "MAP_LIST", str(listing)), patch.object(config, "FEATURES", str(root / "features")), \
                    patch.dict(os.environ, {"SERS_DATA": str(raw)}), patch.object(A, "training_table", wraps=A.training_table) as build:
                config.fit_table(progress=None)
                self.assertEqual(build.call_count, 1)
                config.fit_table(progress=None)
                self.assertEqual(build.call_count, 1)
                annotations(43)
                config.fit_table(progress=None)
                self.assertEqual(build.call_count, 2)
                write_map(raw / "folder/map.csv", model["wn"], [y * 1.1, y])
                config.fit_table(progress=None)
                self.assertEqual(build.call_count, 3)
                listing.write_text(listing.read_text().replace(",3,6,12,", ",6,6,12,"))
                table = config.fit_table(progress=None)
                self.assertEqual(build.call_count, 4)
                np.testing.assert_array_equal(table["uM"], [[6, 6, 12]])

    def test_published_table_requires_opt_in_and_unchanged_annotations(self):
        # The packaged manifest must reject annotation edits even if filenames are unchanged.
        published = json.loads((Path(config.FEATURES) / "published_table.json").read_text())
        self.assertEqual(published["annotation_sha256"], config.annotation_hash())
        with patch.object(config, "raw_root", return_value=None):
            with self.assertRaises(SystemExit):
                config.fit_table(progress=None)
            with patch.object(config, "annotation_hash", return_value="changed"):
                with self.assertRaises(SystemExit):
                    config.fit_table(progress=None, allow_legacy=True)


class ScoreTests(unittest.TestCase):
    def test_score_uses_actual_denominator_and_handles_empty_subsets(self):
        with tempfile.TemporaryDirectory() as d, patch.object(evaluate, "OUT", d):
            evaluate.score({"one": {"true": np.array([3., 6., 12.]), "share": np.array([1/7, 2/7, 4/7]),
                                     "uM": np.array([3., 6., 12.]), "split": "test"}})
            text = (Path(d) / "scores.txt").read_text(encoding="utf-8")
            self.assertIn("1 conditions", text)
            self.assertIn("1/1", text)
            self.assertIn("N/A", text)
            self.assertNotIn("nan", text.lower())


if __name__ == "__main__":
    unittest.main()
