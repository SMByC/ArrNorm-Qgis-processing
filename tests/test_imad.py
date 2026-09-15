"""IR-MAD covariance sampling and multispectral pixel validity."""
import numpy as np
import pytest
from ArrNorm.core import iMad
from ArrNorm.core import raster_io as rio
from ArrNorm.core.auxil.auxil import Cpm
from ArrNorm.tests.helpers import Feedback, build_pair, write_raster
from osgeo import gdal


@pytest.mark.parametrize('nodata', [-9999.0, np.nan])
def test_partial_band_gap_excluded_before_covariance(tmp_path, monkeypatch, nodata):
    build_pair(tmp_path, tgt_nodata=nodata,
               mutate_tgt=lambda bands: bands[1].__setitem__((slice(0, 10), slice(None)), nodata))
    original = Cpm.update
    counts = []

    def record(self, values, weights=None):
        assert np.isfinite(values).all()
        counts.append(len(values))
        return original(self, values, weights)

    monkeypatch.setattr(Cpm, 'update', record)
    iMad.main(str(tmp_path / 'ref.tif'), str(tmp_path / 'tgt.tif'), max_iters=1,
              nodata_tgt=nodata, feedback=Feedback())
    assert sum(counts) == 110 * 130


@pytest.mark.parametrize('side', ['reference', 'target'])
@pytest.mark.parametrize('ref_dtype,tgt_dtype', [
    (gdal.GDT_Float32, gdal.GDT_Float32),
    (gdal.GDT_Float64, gdal.GDT_Float32),
    (gdal.GDT_Float32, gdal.GDT_Float64),
])
def test_explicit_fractional_nodata_excluded_before_widening(tmp_path, monkeypatch, side,
                                                           ref_dtype, tgt_dtype):
    def gap(bands):
        bands[1][:10] = 0.1

    build_pair(tmp_path, ref_dtype=ref_dtype, tgt_dtype=tgt_dtype,
               mutate_ref=gap if side == 'reference' else None,
               mutate_tgt=gap if side == 'target' else None)
    original = Cpm.update
    counts = []

    def record(self, values, weights=None):
        counts.append(len(values))
        return original(self, values, weights)

    monkeypatch.setattr(Cpm, 'update', record)
    iMad.main(str(tmp_path / 'ref.tif'), str(tmp_path / 'tgt.tif'), max_iters=1,
              nodata_ref=0.1 if side == 'reference' else None,
              nodata_tgt=0.1 if side == 'target' else None, feedback=Feedback())
    assert sum(counts) == 110 * 130


@pytest.mark.parametrize('side', ['reference', 'target'])
def test_constant_nonzero_single_band_rejected_before_mad_output(tmp_path, side):
    varying = np.arange(1., 101.).reshape(10, 10)
    constant = np.full(varying.shape, 5.)
    write_raster(tmp_path / 'ref.tif', [constant if side == 'reference' else varying])
    write_raster(tmp_path / 'tgt.tif', [constant if side == 'target' else varying])
    destination = tmp_path / 'mad.tif'
    with pytest.raises(iMad._FatalInputError, match='variance|variation'):
        iMad.main(str(tmp_path / 'ref.tif'), str(tmp_path / 'tgt.tif'), max_iters=2,
                  output=str(destination), feedback=Feedback())
    assert not destination.exists()


def test_single_band_variance_underflow_has_an_explicit_error(tmp_path):
    values = np.arange(1., 101.).reshape(10, 10)
    write_raster(tmp_path / 'ref.tif', [values * 1e-200], dtype=gdal.GDT_Float64)
    write_raster(tmp_path / 'tgt.tif', [values], dtype=gdal.GDT_Float64)
    with pytest.raises(iMad._FatalInputError, match='positive variance'):
        iMad.main(str(tmp_path / 'ref.tif'), str(tmp_path / 'tgt.tif'), max_iters=1,
                  output=str(tmp_path / 'mad.tif'), feedback=Feedback())
    assert not (tmp_path / 'mad.tif').exists()


def test_single_band_affine_relationship_remains_supported(tmp_path):
    values = np.arange(1., 101.).reshape(10, 10)
    write_raster(tmp_path / 'ref.tif', [2 * values + 3], dtype=gdal.GDT_Float64)
    write_raster(tmp_path / 'tgt.tif', [values], dtype=gdal.GDT_Float64)
    result = iMad.main(str(tmp_path / 'ref.tif'), str(tmp_path / 'tgt.tif'), max_iters=2,
                       output=str(tmp_path / 'mad.tif'), feedback=Feedback())
    with rio.open_raster(result) as ds:
        assert np.isfinite(ds.ReadAsArray()).all()
