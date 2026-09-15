"""Radiometric calibration: invariant selection, band ordering, output and plots."""
import numpy as np
import pytest
from ArrNorm.core import radcal
from ArrNorm.core import raster_io as rio
from ArrNorm.core.radcal import QgsProcessingException
from ArrNorm.tests.helpers import (
    Feedback,
    build_pair,
    make_mad_raster,
    read_raster,
    write_raster,
)
from osgeo import gdal


class TestRadcalRegressionStarvation:
    def test_band_without_nochange_pixels_after_nodata_filter_raises(self, tmp_path):
        # Only the nodata strip has high no-change probability. Filtering it
        # must raise instead of fitting an empty regression and writing NaN.
        rows, cols = 20, 20
        drv = gdal.GetDriverByName('GTiff')
        imad = drv.Create(str(tmp_path / 'MAD(ref&tgt).tif'), cols, rows, 5,
                          gdal.GDT_Float32)
        imad.SetGeoTransform((0.0, 10.0, 0.0, 0.0, 0.0, -10.0))
        imad.SetProjection('EPSG:4326')
        for b in range(1, 5):
            imad.GetRasterBand(b).WriteArray(np.zeros((rows, cols), np.float32))
        chisqr = np.full((rows, cols), 1000.0, np.float32)
        chisqr[0:12, :] = 0.0
        imad.GetRasterBand(5).WriteArray(chisqr)
        imad = None

        ref = np.full((rows, cols), 50.0, np.float32)
        tgt = np.full((rows, cols), 60.0, np.float32)
        tgt[0:12, :] = -9999.0
        write_raster(tmp_path / 'ref.tif', [ref] * 4)
        write_raster(tmp_path / 'tgt.tif', [tgt] * 4)

        with pytest.raises(QgsProcessingException, match='no-change pixel'):
            radcal.main(str(tmp_path / 'MAD(ref&tgt).tif'),
                        img_ref=str(tmp_path / 'ref.tif'),
                        img_tgt=str(tmp_path / 'tgt.tif'),
                        output=str(tmp_path / 'out.tif'),
                        out_dtype=gdal.GDT_Float32,
                        nodata_tgt=-9999.0,
                        ncp_threshold=0.95)


def test_band_order_and_block_size_do_not_change_fit(tmp_path):
    values = np.arange(1., 301.).reshape(60, 5)
    reference = [values * 2 + 3, values * 3 + 7]
    target = [values.copy(), values.copy()]
    reference[0][0, 0] = target[0][0, 0] = -9999
    reference[1][0, 0] = 3000  # must be excluded in either order
    write_raster(tmp_path / 'ref.tif', reference)
    write_raster(tmp_path / 'tgt.tif', target)
    mad = make_mad_raster(tmp_path, values.shape)
    for order, block, name in [([1, 2], 7, 'a.tif'), ([2, 1], 23, 'b.tif')]:
        radcal.main(mad, img_ref=str(tmp_path / 'ref.tif'), img_tgt=str(tmp_path / 'tgt.tif'),
                    output=str(tmp_path / name), pos=order, block_rows=block,
                    nodata_ref=-9999, nodata_tgt=-9999, feedback=Feedback())
    np.testing.assert_array_equal(read_raster(tmp_path / 'a.tif')[1], read_raster(tmp_path / 'b.tif')[0])
    np.testing.assert_allclose(read_raster(tmp_path / 'a.tif')[1][1:], reference[1][1:])


def test_integer_negative_conversion_has_known_sentinel(tmp_path):
    target = np.arange(1, 21).reshape(4, 5)
    # Fit on >= 10 only: inverse calibration maps the first pixels below zero.
    reference = np.maximum(2 * target - 15, 0)
    write_raster(tmp_path / 'ref.tif', [reference], dtype=gdal.GDT_UInt16)
    write_raster(tmp_path / 'tgt.tif', [target], dtype=gdal.GDT_UInt16)
    mad = tmp_path / 'mad.tif'
    write_raster(mad, [np.zeros(target.shape), np.where(target >= 10, 0, 1000)])
    radcal.main(str(mad), img_ref=str(tmp_path / 'ref.tif'), img_tgt=str(tmp_path / 'tgt.tif'),
                output=str(tmp_path / 'out.tif'), neg_nodata=65535, feedback=Feedback())
    np.testing.assert_array_equal(read_raster(tmp_path / 'out.tif'),
                                  np.where(2 * target - 15 < 0, 65535, 2 * target - 15))
    with rio.open_raster(tmp_path / 'out.tif') as ds:
        assert ds.GetRasterBand(1).GetNoDataValue() == 65535


def test_optional_plots_do_not_change_global_backend(tmp_path):
    matplotlib = pytest.importorskip('matplotlib')
    backend = matplotlib.get_backend()
    build_pair(tmp_path)
    mad = make_mad_raster(tmp_path, (120, 130), bands=4)
    radcal.main(mad, img_ref=str(tmp_path / 'ref.tif'), img_tgt=str(tmp_path / 'tgt.tif'),
                output=str(tmp_path / 'out.tif'), graphics=True, feedback=Feedback())
    assert matplotlib.get_backend() == backend
    assert (tmp_path / 'out_radcal.png').exists()


@pytest.mark.parametrize('side', ['reference', 'target'])
@pytest.mark.parametrize('other_dtype', [gdal.GDT_Float32, gdal.GDT_Float64])
def test_explicit_fractional_nodata_does_not_bias_known_fit(tmp_path, side, other_dtype):
    values = np.arange(10., 110.).reshape(20, 5)
    target = [values.copy(), values[::-1].copy()]
    expected = np.array([2 * band + 3 for band in target])
    reference = expected.copy()
    if side == 'reference':
        reference[0, :2] = 0.1
    else:
        target[0][:2] = 0.1
    write_raster(tmp_path / 'ref.tif', reference,
                 dtype=gdal.GDT_Float32 if side == 'reference' else other_dtype)
    write_raster(tmp_path / 'tgt.tif', target,
                 dtype=gdal.GDT_Float32 if side == 'target' else other_dtype)
    destination = tmp_path / 'out.tif'
    radcal.main(make_mad_raster(tmp_path, values.shape),
                img_ref=str(tmp_path / 'ref.tif'), img_tgt=str(tmp_path / 'tgt.tif'),
                output=str(destination), nodata_ref=0.1 if side == 'reference' else None,
                nodata_tgt=0.1 if side == 'target' else None, feedback=Feedback())
    actual = read_raster(destination)
    np.testing.assert_allclose(actual[:, 2:], expected[:, 2:], atol=1e-6)
    if side == 'target':
        with rio.open_raster(destination) as ds:
            sentinel = ds.GetRasterBand(1).GetNoDataValue()
        assert np.all(actual[:, :2] == sentinel)


def test_fractional_sentinel_matches_each_vrt_bands_storage_precision(tmp_path):
    values = np.arange(10., 110.).reshape(20, 5)
    target = [values + .25, values[::-1] + .5]
    expected = np.array([2 * band + 3 for band in target])
    target[0][0] = target[1][1] = .1
    for i, dtype in enumerate((gdal.GDT_Float32, gdal.GDT_Float64)):
        write_raster(tmp_path / f'band{i}.tif', [target[i]], dtype=dtype)
    with rio.Raster(gdal.BuildVRT(str(tmp_path / 'tgt.vrt'),
                                  [str(tmp_path / f'band{i}.tif') for i in range(2)],
                                  separate=True), writable=True):
        pass
    with rio.open_raster(tmp_path / 'tgt.vrt') as vrt:
        assert vrt.GetRasterBand(1).DataType == gdal.GDT_Float32
        assert vrt.GetRasterBand(2).DataType == gdal.GDT_Float64
    write_raster(tmp_path / 'ref.tif', expected, dtype=gdal.GDT_Float64)
    destination = tmp_path / 'out.tif'
    radcal.main(make_mad_raster(tmp_path, values.shape),
                img_ref=str(tmp_path / 'ref.tif'), img_tgt=str(tmp_path / 'tgt.vrt'),
                output=str(destination), nodata_tgt=.1, feedback=Feedback())
    actual = read_raster(destination)
    assert np.all(actual[:, :2] == .1)
    np.testing.assert_allclose(actual[:, 2:], expected[:, 2:], atol=1e-9)


@pytest.mark.parametrize('out_dtype', [gdal.GDT_Float64, gdal.GDT_UInt16])
def test_nonfinite_calibration_is_not_hidden_by_clipping_or_nodata(tmp_path, monkeypatch, out_dtype):
    values = np.arange(1., 21.).reshape(4, 5)
    write_raster(tmp_path / 'ref.tif', [2 * values + 3])
    write_raster(tmp_path / 'tgt.tif', [values])
    monkeypatch.setattr(radcal.OrthogonalFit, 'coefficients', lambda self: (-np.inf, 0., 0.))
    with pytest.raises(QgsProcessingException, match='Non-finite calibration output'):
        radcal.main(make_mad_raster(tmp_path, values.shape, bands=1),
                    img_ref=str(tmp_path / 'ref.tif'), img_tgt=str(tmp_path / 'tgt.tif'),
                    output=str(tmp_path / 'out.tif'), out_dtype=out_dtype,
                    neg_nodata=np.nan if out_dtype == gdal.GDT_Float64 else 0,
                    feedback=Feedback())
