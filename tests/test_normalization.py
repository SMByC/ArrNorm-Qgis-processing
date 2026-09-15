"""End-to-end normalization: inputs, alignment, dtypes, masking and pixel values."""
import numpy as np
import pytest
from ArrNorm.core import raster_io as rio
from ArrNorm.core.arrnorm import Normalization, QgsProcessingException
from ArrNorm.tests.helpers import (
    Feedback,
    build_pair,
    make_normalization,
    read_raster,
    synthetic_pair,
    write_raster,
)
from osgeo import gdal


class TestOutputDtypeSelection:
    """Output types must preserve input ranges, independently of GDAL enum order."""

    def _dtype(self, tmp_path, ref_dtype, tgt_dtype):
        build_pair(tmp_path, ref_dtype=ref_dtype, tgt_dtype=tgt_dtype,
                   rows=16, cols=12, seed=5)
        return make_normalization(tmp_path).out_dtype

    def test_float_reference_beats_uint16_target(self, tmp_path):
        assert self._dtype(tmp_path, gdal.GDT_Float32, gdal.GDT_UInt16) == gdal.GDT_Float32

    def test_signed_unsigned_inputs_promote_to_int32(self, tmp_path):
        assert self._dtype(tmp_path, gdal.GDT_Int16, gdal.GDT_UInt16) == gdal.GDT_Int32

    def test_float64_beats_everything(self, tmp_path):
        assert self._dtype(tmp_path, gdal.GDT_Float64, gdal.GDT_Byte) == gdal.GDT_Float64

    def test_same_dtypes_are_kept(self, tmp_path):
        assert self._dtype(tmp_path, gdal.GDT_UInt16, gdal.GDT_UInt16) == gdal.GDT_UInt16


class TestInvalidInputRasters:
    def test_missing_file_raises_processing_exception(self, tmp_path):
        with pytest.raises(QgsProcessingException, match='Cannot open raster:.*ref.tif'):
            make_normalization(tmp_path)

    def test_corrupt_file_raises_processing_exception(self, tmp_path):
        (tmp_path / 'ref.tif').write_bytes(b'not a geotiff at all')
        (tmp_path / 'tgt.tif').write_bytes(b'not a geotiff at all')
        with pytest.raises(QgsProcessingException, match='Cannot open raster:.*ref.tif'):
            make_normalization(tmp_path)


class TestNegToNodata:
    DARK_ANOMALY = (slice(30, 36), slice(30, 80))

    def _with_anomaly(self, tgt):
        tgt[0][self.DARK_ANOMALY] = 1.0

    def test_negatives_become_nodata_for_float_output(self, tmp_path):
        build_pair(tmp_path, tgt_nodata=-9999.0, seed=3,
                   mutate_tgt=self._with_anomaly)
        norm = make_normalization(tmp_path, neg_to_nodata=True)
        norm.run()

        ds = gdal.Open(str(tmp_path / 'out.tif'))
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        out = band.ReadAsArray()
        ds = None

        assert nodata == -9999.0
        assert np.all((out == -9999.0) | (out >= 0.0))
        assert np.any(out == -9999.0)

    def test_without_option_negatives_are_clipped_for_int_output(self, tmp_path):
        build_pair(tmp_path, ref_dtype=gdal.GDT_UInt16, tgt_dtype=gdal.GDT_UInt16,
                   seed=3, mutate_tgt=self._with_anomaly)
        norm = make_normalization(tmp_path)
        norm.run()

        ds = gdal.Open(str(tmp_path / 'out.tif'))
        out = ds.GetRasterBand(1).ReadAsArray()
        ds = None

        assert out.min() >= 0


class TestTargetNodataExcludedFromCalibration:
    def test_declared_nodata_strip_no_longer_breaks_ir_mad(self, tmp_path):
        def _fill_strip(tgt):
            for band in tgt:
                band[0:12, :] = -9999.0

        build_pair(tmp_path, tgt_nodata=-9999.0, seed=3, mutate_tgt=_fill_strip)
        norm = make_normalization(tmp_path, nodata_mask=True, keep_mask_layer=False)
        norm.run()

        assert (tmp_path / 'out.tif').exists()
        ds = gdal.Open(str(tmp_path / 'out.tif'))
        band1 = ds.GetRasterBand(1).ReadAsArray()
        ds = None

        assert np.all(band1[0:12, :] == -9999.0)
        assert np.any(band1[12:, :] != -9999.0)


class TestClipperGuards:
    def test_rotated_target_geotransform_rejected(self, tmp_path):
        build_pair(tmp_path, ref_offset=(37.0, 11.0), seed=3)
        write_raster(tmp_path / 'tgt.tif', synthetic_pair(seed=3)[1],
                     geotransform=(0.0, 10.0, 0.5, 0.0, 0.0, -10.0))
        with pytest.raises(QgsProcessingException, match='rotated'):
            make_normalization(tmp_path).run()

    def test_unprojected_target_rejected(self, tmp_path):
        build_pair(tmp_path, ref_offset=(37.0, 11.0), seed=3)
        write_raster(tmp_path / 'tgt.tif', synthetic_pair(seed=3)[1],
                     projection='')
        with pytest.raises(QgsProcessingException, match='no coordinate reference system'):
            make_normalization(tmp_path).run()


def test_pipeline_mixed_types_preserves_known_negative_reference(tmp_path):
    rng = np.random.default_rng(123)
    target = [rng.integers(100, 180, (24, 25)), rng.integers(90, 190, (24, 25))]
    reference = [2 * band - 500 for band in target]
    write_raster(tmp_path / 'ref.tif', reference, dtype=gdal.GDT_Int16)
    write_raster(tmp_path / 'tgt.tif', target, dtype=gdal.GDT_UInt16)
    norm = make_normalization(tmp_path)
    norm.run()
    np.testing.assert_array_equal(read_raster(norm.output_file), reference)


@pytest.mark.parametrize('nodata', [-9999.0, np.nan])
def test_partial_band_gap_preserved_in_output_and_mask(tmp_path, nodata):
    build_pair(tmp_path, tgt_nodata=nodata,
               mutate_tgt=lambda bands: bands[1].__setitem__((slice(0, 10), slice(None)), nodata))
    norm = make_normalization(tmp_path, nodata_mask=True, keep_mask_layer=True)
    norm.run()
    output = read_raster(norm.output_file)
    invalid = ~rio.valid_values(output, nodata)
    assert invalid[:, :10, :].all()
    assert not invalid[:, 10:, :].any()
    mask = read_raster(norm.mask_file)
    assert not mask[:10].any() and mask[10:].all()


@pytest.mark.parametrize('sentinel', [-9999, 0.5, float('nan'), float('inf'), 70000])
@pytest.mark.parametrize('mask,negative', [(True, False), (False, True)])
def test_unrepresentable_nodata_fails_before_any_output(tmp_path, sentinel, mask, negative):
    build_pair(tmp_path, ref_dtype=gdal.GDT_UInt16, tgt_dtype=gdal.GDT_UInt16)
    with pytest.raises(QgsProcessingException, match='not representable'):
        make_normalization(tmp_path, nodata_mask=mask, neg_to_nodata=negative,
                           nodata_mask_value=sentinel)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['ref.tif', 'tgt.tif']


def test_vrt_inputs_use_real_geotiff_output(tmp_path):
    build_pair(tmp_path)
    norm = make_normalization(tmp_path)
    norm.run()
    expected = read_raster(norm.output_file)
    for stem in ('ref', 'tgt'):
        with rio.Raster(gdal.Translate(str(tmp_path / f'{stem}.vrt'),
                                       str(tmp_path / f'{stem}.tif'), format='VRT'), writable=True):
            pass
    other = Normalization(str(tmp_path / 'ref.vrt'), str(tmp_path / 'tgt.vrt'), 8, .99, .95,
                          False, False, None, False, None, False,
                          str(tmp_path / 'vrt_output.tif'), Feedback())
    other.run()
    with rio.open_raster(other.output_file) as ds:
        assert ds.GetDriver().ShortName == 'GTiff'
        np.testing.assert_array_equal(ds.ReadAsArray(), expected)


def test_reference_mask_on_aligned_rotated_grid_needs_no_warp(tmp_path):
    ref, target = synthetic_pair(rows=40, cols=30)
    gt = (100., 10., .5, 200., .25, -20.)
    ref[1][:3] = -9999
    write_raster(tmp_path / 'ref.tif', ref, nodata=-9999, geotransform=gt, projection='')
    write_raster(tmp_path / 'tgt.tif', target, geotransform=gt, projection='')
    norm = make_normalization(tmp_path, mask_ref=True)
    norm.run()
    assert norm.img_ref_clip == norm.img_ref
    with rio.open_raster(norm.output_file) as ds:
        assert ds.GetGeoTransform() == gt
        assert ds.GetProjection() == ''
        assert np.isfinite(ds.ReadAsArray()).all()


@pytest.mark.parametrize('offset', [(0., 0.), (37., 11.)])
def test_headless_normalization_accepts_none_feedback(tmp_path, offset, capsys):
    build_pair(tmp_path, ref_offset=offset)
    expected = make_normalization(tmp_path, nodata_mask=True)
    expected.output_file = str(tmp_path / 'expected.tif')
    expected.run()
    actual = Normalization(str(tmp_path / 'ref.tif'), str(tmp_path / 'tgt.tif'), 8, .99, .95,
                           False, False, None, True, None, False,
                           str(tmp_path / 'actual.tif'), None)
    assert actual.run() == str(tmp_path / 'actual.tif')
    np.testing.assert_array_equal(read_raster(actual.output_file), read_raster(expected.output_file))
    assert not list(tmp_path.glob('.arrnorm-*'))
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('side', ['reference', 'target'])
@pytest.mark.parametrize('offset', [(0., 0.), (37., 11.)])
def test_fractional_explicit_and_auto_nodata_select_identical_pixels(tmp_path, side, offset):
    def gap(bands):
        bands[1][:10] = 0.1

    build_pair(tmp_path, ref_offset=offset,
               ref_dtype=gdal.GDT_Float32 if side == 'reference' else gdal.GDT_Float64,
               tgt_dtype=gdal.GDT_Float32 if side == 'target' else gdal.GDT_Float64,
               ref_nodata=0.1 if side == 'reference' else None,
               tgt_nodata=0.1 if side == 'target' else None,
               mutate_ref=gap if side == 'reference' else None,
               mutate_tgt=gap if side == 'target' else None)
    options = {'mask_ref': side == 'reference', 'nodata_mask': side == 'target',
               'keep_mask_layer': side == 'target'}
    auto = make_normalization(tmp_path, **options)
    auto.output_file = str(tmp_path / 'auto.tif')
    auto.run()
    options['mask_ref_nodata' if side == 'reference' else 'nodata_mask_value'] = 0.1
    explicit = make_normalization(tmp_path, **options)
    explicit.run()
    if side == 'target':
        auto_mask, explicit_mask = read_raster(auto.mask_file), read_raster(explicit.mask_file)
        np.testing.assert_array_equal(auto_mask, explicit_mask)
        valid = explicit_mask == 1
        np.testing.assert_array_equal(read_raster(auto.output_file)[:, valid],
                                      read_raster(explicit.output_file)[:, valid])
        assert np.all(read_raster(explicit.output_file)[:, ~valid] == explicit.mask_nodata)
    else:
        np.testing.assert_array_equal(read_raster(auto.output_file), read_raster(explicit.output_file))


def test_fractional_resolution_warp_preserves_exact_target_grid_with_mask(tmp_path):
    ref, target = synthetic_pair()
    gt = (500000.123, 10.1234567, 0., 4500000.345, 0., -10.1234567)
    reference_gt = (gt[0] + 37., gt[1], 0., gt[3] - 11., 0., gt[5])
    write_raster(tmp_path / 'ref.tif', ref, geotransform=reference_gt, projection='EPSG:32618')
    write_raster(tmp_path / 'tgt.tif', target, geotransform=gt, projection='EPSG:32618')
    norm = make_normalization(tmp_path, nodata_mask=True, keep_mask_layer=True)
    norm.run()
    with rio.open_raster(norm.img_target) as source:
        projection = source.GetProjection()
    for path in (norm.output_file, norm.mask_file):
        with rio.open_raster(path) as ds:
            assert ds.GetGeoTransform() == gt
            assert ds.GetProjection() == projection
            assert (ds.RasterYSize, ds.RasterXSize) == target[0].shape
    assert np.isfinite(read_raster(norm.output_file)).all()


@pytest.mark.parametrize('mask_target', [False, True])
def test_negative_conversion_to_nan_nodata_completes(tmp_path, mask_target):
    anomaly = (slice(30, 36), slice(30, 80))
    build_pair(tmp_path, tgt_nodata=np.nan,
               mutate_tgt=lambda bands: bands[0].__setitem__(anomaly, 1.0))
    norm = make_normalization(tmp_path, neg_to_nodata=True, nodata_mask=mask_target)
    norm.run()
    output = read_raster(norm.output_file)
    assert np.isnan(output[0][anomaly]).all()
    assert np.isfinite(output[1:]).all()
    assert np.all(output[np.isfinite(output)] >= 0)
    with rio.open_raster(norm.output_file) as ds:
        for band in range(1, ds.RasterCount + 1):
            assert np.isnan(ds.GetRasterBand(band).GetNoDataValue())


def test_warped_mixed_dtype_reference_preserves_ranges_and_calibration(tmp_path):
    rng = np.random.default_rng(19)
    reference = [rng.integers(-2000, -500, (24, 25)),
                 rng.integers(40000, 50000, (24, 25))]
    gt = (100., 10., 0., 200., 0., -10.)
    for i, dtype in enumerate((gdal.GDT_Int16, gdal.GDT_UInt16)):
        write_raster(tmp_path / f'ref_band{i}.tif', [reference[i]], dtype=dtype,
                     geotransform=gt, projection='EPSG:32618')
    vrt_path = tmp_path / 'ref.vrt'
    with rio.Raster(gdal.BuildVRT(str(vrt_path),
                                  [str(tmp_path / f'ref_band{i}.tif') for i in range(2)],
                                  separate=True), writable=True):
        pass
    expected = np.array([band[:, 1:] for band in reference])
    target = [(expected[0] + 3000) / 2, (expected[1] - 10000) / 2]
    target_gt = (gt[0] + gt[1], *gt[1:])
    write_raster(tmp_path / 'tgt.tif', target, dtype=gdal.GDT_Float64,
                 geotransform=target_gt, projection='EPSG:32618')
    norm = Normalization(str(vrt_path), str(tmp_path / 'tgt.tif'), 8, .99, .95,
                         False, False, None, True, None, False,
                         str(tmp_path / 'out.tif'), Feedback())
    try:
        norm.clipper()
        with rio.open_raster(norm.img_ref_clip) as clipped:
            np.testing.assert_array_equal(clipped.ReadAsArray(), expected)
            assert clipped.GetRasterBand(1).DataType == gdal.GDT_Int32
            assert clipped.GetRasterBand(2).DataType == gdal.GDT_Int32
        norm.run()
        np.testing.assert_allclose(read_raster(norm.output_file), expected, atol=1e-7)
    finally:
        norm.clean()


def test_alignment_rejects_inexact_int64_before_warp_can_round_it(tmp_path):
    if not hasattr(gdal, 'GDT_Int64'):
        pytest.skip('Int64 raster support requires GDAL 3.5+')
    gt = (100., 10., 0., 200., 0., -10.)
    with rio.create_raster(tmp_path / 'ref.tif', 5, 5, 1, gdal.GDT_Int64) as ref:
        ref.SetGeoTransform(gt)
        ref.SetProjection('EPSG:32618')
        # Deliberately use raw GDAL here: the production checked reader must
        # reject this integer, which float64 would round down to exactly 2^53.
        ref.GetRasterBand(1).WriteArray(np.full((5, 5), 2 ** 53 + 1, dtype=np.int64))
    write_raster(tmp_path / 'tgt.tif', [np.arange(1., 26.).reshape(5, 5)],
                 geotransform=(110., *gt[1:]), projection='EPSG:32618')
    norm = make_normalization(tmp_path)
    try:
        with pytest.raises((ValueError, QgsProcessingException), match='2\\^53'):
            norm.clipper()
    finally:
        norm.clean()


def test_warped_mixed_float_reference_preserves_fractional_nodata(tmp_path):
    values = np.arange(10., 410.).reshape(20, 20)
    reference = [values + .25, values[::-1] + .5]
    reference[0][3:6, 3:6] = .1
    reference[1][10:13, 10:13] = .1
    gt = (100., 10., 0., 200., 0., -10.)
    for i, dtype in enumerate((gdal.GDT_Float32, gdal.GDT_Float64)):
        write_raster(tmp_path / f'ref_band{i}.tif', [reference[i]], dtype=dtype,
                     geotransform=gt, projection='EPSG:32618')
    with rio.Raster(gdal.BuildVRT(str(tmp_path / 'ref.vrt'),
                                  [str(tmp_path / f'ref_band{i}.tif') for i in range(2)],
                                  separate=True), writable=True):
        pass
    expected = np.array([band[:, 1:] for band in reference])
    write_raster(tmp_path / 'tgt.tif', expected, dtype=gdal.GDT_Float64,
                 geotransform=(110., *gt[1:]), projection='EPSG:32618')
    norm = Normalization(str(tmp_path / 'ref.vrt'), str(tmp_path / 'tgt.tif'), 8, .99, .95,
                         False, True, .1, False, None, False,
                         str(tmp_path / 'out.tif'), Feedback())
    try:
        norm.clipper()
        np.testing.assert_array_equal(read_raster(norm.img_ref_clip), expected)
    finally:
        norm.clean()
