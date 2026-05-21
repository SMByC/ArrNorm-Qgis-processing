import os
import tempfile

import numpy as np
import pytest
from osgeo import gdal

from ArrNorm.core import raster_ops


def _create_test_raster(path, data, nodata=None, dtype=gdal.GDT_Float32):
    """Create a single-band GeoTIFF with the given numpy array."""
    rows, cols = data.shape
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(path, cols, rows, 1, dtype)
    ds.SetGeoTransform((0, 1, 0, 0, 0, -1))
    ds.SetProjection("")
    band = ds.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(nodata)
    band.WriteArray(data.astype(gdal_array_dtype(dtype)))
    band.FlushCache()
    ds = None


def _create_multi_band_raster(path, bands_data, nodata=None, dtype=gdal.GDT_Float32):
    """Create a multi-band GeoTIFF."""
    rows, cols = bands_data[0].shape
    nbands = len(bands_data)
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(path, cols, rows, nbands, dtype)
    ds.SetGeoTransform((0, 1, 0, 0, 0, -1))
    ds.SetProjection("")
    for b, data in enumerate(bands_data, start=1):
        band = ds.GetRasterBand(b)
        if nodata is not None:
            band.SetNoDataValue(nodata)
        band.WriteArray(data.astype(gdal_array_dtype(dtype)))
        band.FlushCache()
    ds = None


def gdal_array_dtype(gdal_dtype):
    """Map GDAL type to numpy dtype."""
    mapping = {
        gdal.GDT_Byte: np.uint8,
        gdal.GDT_UInt16: np.uint16,
        gdal.GDT_Int16: np.int16,
        gdal.GDT_UInt32: np.uint32,
        gdal.GDT_Int32: np.int32,
        gdal.GDT_Float32: np.float32,
        gdal.GDT_Float64: np.float64,
    }
    return mapping[gdal_dtype]


class TestNoNegativeValue:
    def test_negative_values_to_explicit_nodata(self, tmp_path):
        arr = np.array([[-1, 0, 1], [-5, 10, -3]], dtype=np.float32)
        inp = str(tmp_path / "in.tif")
        out = str(tmp_path / "out.tif")
        _create_test_raster(inp, arr)

        raster_ops.no_negative_value(inp, out, nodata_value=-9999)

        ds = gdal.Open(out)
        band = ds.GetRasterBand(1)
        result = band.ReadAsArray()
        expected = np.array([[-9999, 0, 1], [-9999, 10, -9999]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
        assert band.GetNoDataValue() == -9999
        ds = None

    def test_nodata_propagated(self, tmp_path):
        arr = np.array([[1, -2, 3], [4, -5, 6]], dtype=np.float32)
        inp = str(tmp_path / "in.tif")
        out = str(tmp_path / "out.tif")
        _create_test_raster(inp, arr, nodata=-5)

        raster_ops.no_negative_value(inp, out, nodata_value=-9999)

        ds = gdal.Open(out)
        result = ds.GetRasterBand(1).ReadAsArray()
        expected = np.array([[1, -9999, 3], [4, -9999, 6]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
        assert ds.GetRasterBand(1).GetNoDataValue() == -9999
        ds = None

    def test_auto_detect_nodata_from_input(self, tmp_path):
        arr = np.array([[1, -2, 3], [4, -5, 6]], dtype=np.float32)
        inp = str(tmp_path / "in.tif")
        out = str(tmp_path / "out.tif")
        _create_test_raster(inp, arr, nodata=-5)

        raster_ops.no_negative_value(inp, out)

        ds = gdal.Open(out)
        result = ds.GetRasterBand(1).ReadAsArray()
        expected = np.array([[1, -5, 3], [4, -5, 6]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
        assert ds.GetRasterBand(1).GetNoDataValue() == -5
        ds = None

    def test_fallback_to_zero_when_no_nodata(self, tmp_path):
        arr = np.array([[-1, 0, 1], [-5, 10, -3]], dtype=np.float32)
        inp = str(tmp_path / "in.tif")
        out = str(tmp_path / "out.tif")
        _create_test_raster(inp, arr)

        raster_ops.no_negative_value(inp, out)

        ds = gdal.Open(out)
        result = ds.GetRasterBand(1).ReadAsArray()
        expected = np.array([[0, 0, 1], [0, 10, 0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
        assert ds.GetRasterBand(1).GetNoDataValue() == 0
        ds = None

    def test_multi_band(self, tmp_path):
        band1 = np.array([[-1, 2], [-3, 4]], dtype=np.float32)
        band2 = np.array([[5, -6], [7, -8]], dtype=np.float32)
        inp = str(tmp_path / "in.tif")
        out = str(tmp_path / "out.tif")
        _create_multi_band_raster(inp, [band1, band2])

        raster_ops.no_negative_value(inp, out, nodata_value=0)

        ds = gdal.Open(out)
        assert ds.RasterCount == 2
        r1 = ds.GetRasterBand(1).ReadAsArray()
        r2 = ds.GetRasterBand(2).ReadAsArray()
        np.testing.assert_array_equal(r1, np.array([[0, 2], [0, 4]], dtype=np.float32))
        np.testing.assert_array_equal(r2, np.array([[5, 0], [7, 0]], dtype=np.float32))
        ds = None


class TestMakeMask:
    def test_basic_mask(self, tmp_path):
        arr = np.array([[0, 1, 2], [3, 0, 5]], dtype=np.float32)
        inp = str(tmp_path / "in.tif")
        out = str(tmp_path / "out.tif")
        _create_test_raster(inp, arr, nodata=0)

        raster_ops.make_mask(inp, out, nodata_value=0)

        ds = gdal.Open(out)
        assert ds.GetRasterBand(1).DataType == gdal.GDT_Byte
        result = ds.GetRasterBand(1).ReadAsArray()
        expected = np.array([[0, 1, 1], [1, 0, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

        ct = ds.GetRasterBand(1).GetRasterColorTable()
        assert ct is not None
        assert ct.GetColorEntry(0) == (0, 0, 0, 255)
        assert ct.GetColorEntry(1) == (0, 255, 0, 255)
        ds = None

    def test_nan_nodata_guard(self, tmp_path):
        arr = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, 6.0]], dtype=np.float32)
        inp = str(tmp_path / "in.tif")
        out = str(tmp_path / "out.tif")
        _create_test_raster(inp, arr, nodata=np.nan)

        raster_ops.make_mask(inp, out, nodata_value=np.nan)

        ds = gdal.Open(out)
        result = ds.GetRasterBand(1).ReadAsArray()
        expected = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)
        ds = None


class TestApplyMask:
    def test_basic_application(self, tmp_path):
        img = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32)
        mask = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        inp_img = str(tmp_path / "img.tif")
        inp_mask = str(tmp_path / "mask.tif")
        out = str(tmp_path / "out.tif")
        _create_test_raster(inp_img, img)
        _create_test_raster(inp_mask, mask, dtype=gdal.GDT_Byte)

        raster_ops.apply_mask(inp_img, inp_mask, out, nodata_value=0)

        ds = gdal.Open(out)
        result = ds.GetRasterBand(1).ReadAsArray()
        expected = np.array([[10, 0, 30], [0, 50, 0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
        assert ds.GetRasterBand(1).GetNoDataValue() == 0
        ds = None

    def test_multi_band(self, tmp_path):
        b1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b2 = np.array([[5, 6], [7, 8]], dtype=np.float32)
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        inp_img = str(tmp_path / "img.tif")
        inp_mask = str(tmp_path / "mask.tif")
        out = str(tmp_path / "out.tif")
        _create_multi_band_raster(inp_img, [b1, b2])
        _create_test_raster(inp_mask, mask, dtype=gdal.GDT_Byte)

        raster_ops.apply_mask(inp_img, inp_mask, out, nodata_value=0)

        ds = gdal.Open(out)
        assert ds.RasterCount == 2
        r1 = ds.GetRasterBand(1).ReadAsArray()
        r2 = ds.GetRasterBand(2).ReadAsArray()
        np.testing.assert_array_equal(r1, np.array([[1, 0], [0, 4]], dtype=np.float32))
        np.testing.assert_array_equal(r2, np.array([[5, 0], [0, 8]], dtype=np.float32))
        ds = None

    def test_dimension_mismatch_raises(self, tmp_path):
        img = np.zeros((2, 3), dtype=np.float32)
        mask = np.zeros((2, 2), dtype=np.uint8)
        inp_img = str(tmp_path / "img.tif")
        inp_mask = str(tmp_path / "mask.tif")
        out = str(tmp_path / "out.tif")
        _create_test_raster(inp_img, img)
        _create_test_raster(inp_mask, mask, dtype=gdal.GDT_Byte)

        with pytest.raises(RuntimeError, match="don't match"):
            raster_ops.apply_mask(inp_img, inp_mask, out, nodata_value=0)

    def test_block_boundary_handling(self, tmp_path):
        arr = np.arange(300 * 4, dtype=np.float32).reshape(300, 4)
        mask = np.ones((300, 4), dtype=np.uint8)
        inp_img = str(tmp_path / "img.tif")
        inp_mask = str(tmp_path / "mask.tif")
        out = str(tmp_path / "out.tif")
        _create_test_raster(inp_img, arr)
        _create_test_raster(inp_mask, mask, dtype=gdal.GDT_Byte)

        raster_ops.apply_mask(inp_img, inp_mask, out, nodata_value=0, block_rows=256)

        ds = gdal.Open(out)
        result = ds.GetRasterBand(1).ReadAsArray()
        np.testing.assert_array_equal(result, arr)
        ds = None
