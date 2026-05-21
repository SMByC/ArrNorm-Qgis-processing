"""
Integration tests for the ArrNorm QGIS plugin normalization pipeline.

Each scenario runs the full Normalization pipeline (clipper → iMad → radcal,
optionally with no_negative_value, make_mask, apply_mask) inside a fresh
temporary directory and checks:

  1. Output file exists with correct spatial properties (CRS, geotransform,
     dimensions, dtype, value range).
  2. Pixel values match the preserved expected outputs stored in
     tests/data/expected/ (regression baseline).

Fast-path tests additionally verify that the clipper() step is skipped when
the reference is already aligned to the target grid.
"""
import shutil

import numpy as np
import pytest
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly
from pathlib import Path

from ArrNorm.core.arrnorm import Normalization

DATA_DIR = Path(__file__).parent / "data"
EXPECTED_DIR = DATA_DIR / "expected"

TARGET_COLS = 536
TARGET_ROWS = 349
TARGET_BANDS = 4


class MockFeedback:
    """Minimal stand-in for QgsProcessingFeedback (no QGIS required)."""

    def __init__(self):
        self._progress = 0
        self._canceled = False
        self.messages = []

    def pushInfo(self, msg):
        self.messages.append(msg)

    def reportError(self, msg, fatalError=False):
        self.messages.append(f"ERROR: {msg}")

    def setProgress(self, value):
        self._progress = value

    def isCanceled(self):
        return self._canceled


def _read_bands(path):
    """Return all bands of a GeoTIFF as a (bands, rows, cols) array."""
    ds = gdal.Open(str(path), GA_ReadOnly)
    arr = np.array([ds.GetRasterBand(b + 1).ReadAsArray()
                    for b in range(ds.RasterCount)])
    ds = None
    return arr


def _raster_info(path):
    ds = gdal.Open(str(path), GA_ReadOnly)
    info = {
        "cols":   ds.RasterXSize,
        "rows":   ds.RasterYSize,
        "bands":  ds.RasterCount,
        "dtype":  ds.GetRasterBand(1).DataType,
        "nodata": ds.GetRasterBand(1).GetNoDataValue(),
        "proj":   ds.GetProjection(),
        "gt":     ds.GetGeoTransform(),
    }
    ds = None
    return info


def _run(workdir, ref_name, target_name="target.tif", **kw):
    feedback = MockFeedback()
    norm = Normalization(
        img_ref=str(workdir / ref_name),
        img_target=str(workdir / target_name),
        max_iters=kw.get("max_iters", 30),
        conv_threshold=kw.get("conv_threshold", 0.99),
        ncp_threshold=kw.get("ncp_threshold", 0.95),
        neg_to_nodata=kw.get("neg_to_nodata", False),
        mask_ref=kw.get("mask_ref", False),
        mask_ref_nodata=None,
        nodata_mask=kw.get("nodata_mask", False),
        nodata_mask_value=kw.get("nodata_mask_value", None),
        keep_mask_layer=kw.get("keep_mask_layer", False),
        output_file=str(workdir / "output.tif"),
        feedback=feedback,
    )
    norm.run()
    return norm


def _check_properties(norm):
    out = Path(norm.output_file)
    assert out.exists(), f"Normalized output missing: {out}"
    info = _raster_info(out)
    assert info["cols"]  == TARGET_COLS
    assert info["rows"]  == TARGET_ROWS
    assert info["bands"] == TARGET_BANDS
    assert info["dtype"] == gdal.GDT_UInt16
    data = _read_bands(out)
    assert int(data.min()) >= 0
    assert int(data.max()) <= 65535


def _regression(norm, expected_name):
    actual   = _read_bands(Path(norm.output_file))
    expected = _read_bands(EXPECTED_DIR / expected_name)
    np.testing.assert_array_equal(
        actual, expected,
        err_msg=f"Pixel values differ from baseline {expected_name}",
    )


@pytest.fixture
def workdir(tmp_path):
    """Copy test data into a temporary directory."""
    for name in ("ref.tif", "ref_adjusted2target.tif", "target.tif"):
        src = DATA_DIR / name
        if src.exists():
            shutil.copy(str(src), str(tmp_path / name))
    return tmp_path


class TestFullRef:
    def test_output_properties(self, workdir):
        norm = _run(workdir, "ref.tif")
        _check_properties(norm)

    def test_projection_matches_target(self, workdir):
        norm = _run(workdir, "ref.tif")
        assert (_raster_info(Path(norm.output_file))["proj"] ==
                _raster_info(workdir / "target.tif")["proj"])

    def test_geotransform_matches_target(self, workdir):
        norm = _run(workdir, "ref.tif")
        out_gt = _raster_info(Path(norm.output_file))["gt"]
        tgt_gt = _raster_info(workdir / "target.tif")["gt"]
        assert out_gt == pytest.approx(tgt_gt)

    def test_temp_clip_deleted_after_run(self, workdir):
        """Clipper's temporary reprojected file must be removed by clean()."""
        norm = _run(workdir, "ref.tif")
        assert not Path(norm.img_ref_clip).exists() or norm.img_ref_clip == norm.img_ref

    def test_regression(self, workdir):
        norm = _run(workdir, "ref.tif")
        _regression(norm, "target_norm_full_ref.tif")


class TestPrealignedRef:
    def test_output_properties(self, workdir):
        norm = _run(workdir, "ref_adjusted2target.tif")
        _check_properties(norm)

    def test_fast_path_no_temp_clip(self, workdir):
        """When grids already match, clipper must reuse img_ref directly."""
        norm = _run(workdir, "ref_adjusted2target.tif")
        assert norm.img_ref_clip == norm.img_ref

    def test_regression(self, workdir):
        norm = _run(workdir, "ref_adjusted2target.tif")
        _regression(norm, "target_norm_prealigned.tif")


@pytest.mark.parametrize("conv_thresh,expected_file", [
    (0.95,  "target_norm_conv095.tif"),
    (0.999, "target_norm_conv0999.tif"),
])
def test_convergence_threshold(workdir, conv_thresh, expected_file):
    norm = _run(workdir, "ref_adjusted2target.tif", conv_threshold=conv_thresh)
    _check_properties(norm)
    _regression(norm, expected_file)


def test_threshold_090(workdir):
    norm = _run(workdir, "ref_adjusted2target.tif", ncp_threshold=0.90)
    _check_properties(norm)
    _regression(norm, "target_norm_t090.tif")


class TestMask:

    def _run_masked(self, workdir, mask_value=None):
        return _run(workdir, "ref_adjusted2target.tif",
                    nodata_mask=True, nodata_mask_value=mask_value,
                    keep_mask_layer=True)

    def test_mask_file_exists(self, workdir):
        norm = self._run_masked(workdir)
        assert Path(norm.mask_file).exists()

    def test_mask_is_binary(self, workdir):
        norm = self._run_masked(workdir)
        mask = _read_bands(Path(norm.mask_file))[0]
        unique_vals = set(np.unique(mask).tolist())
        assert unique_vals.issubset({0, 1}), \
            f"Mask contains non-binary values: {unique_vals}"

    def test_mask_zero_where_target_zero(self, workdir):
        """Every pixel that is 0 in band 1 of the target must be masked out."""
        norm = self._run_masked(workdir)
        target_b1 = _read_bands(workdir / "target.tif")[0]
        mask = _read_bands(Path(norm.mask_file))[0]
        assert np.all(mask[target_b1 == 0] == 0), \
            "Some zero-target pixels were not set to 0 in the mask"

    def test_normalized_zero_where_mask_zero(self, workdir):
        """All bands of the normalized output must be 0 wherever the mask is 0."""
        norm = self._run_masked(workdir)
        mask = _read_bands(Path(norm.mask_file))[0]
        out  = _read_bands(Path(norm.output_file))
        for b in range(out.shape[0]):
            assert np.all(out[b][mask == 0] == 0), \
                f"Band {b + 1}: non-zero values found under masked pixels"

    def test_auto_nodata_equals_explicit_zero(self, tmp_path_factory):
        """Auto-detect nodata=0 and explicit nodata=0 must produce identical masks."""
        wd_auto = tmp_path_factory.mktemp("auto")
        wd_expl = tmp_path_factory.mktemp("explicit")
        for name in ("ref.tif", "ref_adjusted2target.tif", "target.tif"):
            for wd in (wd_auto, wd_expl):
                shutil.copy(str(DATA_DIR / name), str(wd / name))
        norm_auto = _run(wd_auto, "ref_adjusted2target.tif",
                         nodata_mask=True, keep_mask_layer=True)
        norm_expl = _run(wd_expl, "ref_adjusted2target.tif",
                         nodata_mask=True, nodata_mask_value=0.0, keep_mask_layer=True)
        np.testing.assert_array_equal(
            _read_bands(Path(norm_auto.mask_file)),
            _read_bands(Path(norm_expl.mask_file)),
        )

    def test_regression_normalized(self, workdir):
        norm = self._run_masked(workdir)
        _regression(norm, "target_norm_masked.tif")

    def test_regression_mask(self, workdir):
        norm = self._run_masked(workdir)
        actual   = _read_bands(Path(norm.mask_file))
        expected = _read_bands(EXPECTED_DIR / "target_mask.tif")
        np.testing.assert_array_equal(actual, expected)
