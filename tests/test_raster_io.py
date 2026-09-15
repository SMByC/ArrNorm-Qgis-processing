"""GDAL I/O contracts: type promotion, bounded reads, error modes and handles.

Palette-order and lifetime checks use simulated GDAL handles on every platform;
they are not a substitute for running QGIS on Windows.
"""
import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
from ArrNorm.core import iMad, radcal, raster_ops
from ArrNorm.core import raster_io as rio
from ArrNorm.tests.helpers import Feedback, build_pair, make_mad_raster
from osgeo import gdal


class _InjectedGdalError(RuntimeError):
    pass


class _GdalHandle(SimpleNamespace):
    __slots__ = ("__weakref__",)


def _install_spy_gdal(monkeypatch, *, reject_late_palette=False,
                       fail_write=False):
    state = SimpleNamespace(
        events=[], references={}, color_entries={}, create_dtype=None,
        creation_options=None)
    source_band = _GdalHandle(
        DataType=gdal.GDT_Float32,
        ReadAsArray=lambda _x, _y, _cols, _rows: np.array(
            [[0, 1], [2, 0]], dtype=np.float32))
    output_band = _GdalHandle()
    source_ds = _GdalHandle(RasterXSize=2, RasterYSize=2, RasterCount=1)
    output_ds = _GdalHandle(RasterXSize=2, RasterYSize=2)
    driver = _GdalHandle()
    source_ds.GetDriver = lambda: driver
    source_ds.GetRasterBand = lambda _index: source_band
    source_ds.GetGeoTransform = lambda: None
    source_ds.GetProjection = lambda: None
    output_ds.GetRasterBand = lambda _index: output_band
    output_ds.FlushCache = lambda: None

    state.references = {
        "source": weakref.ref(source_ds),
        "source_band": weakref.ref(source_band),
        "destination": weakref.ref(output_ds),
        "destination_band": weakref.ref(output_band),
    }
    source_box = [source_ds]
    output_box = [output_ds]

    def create(_path, _cols, _rows, _bands, dtype, options):
        state.create_dtype = dtype
        state.creation_options = options
        return output_box.pop()

    def set_color_table(color_table):
        state.events.append("set_color_table")
        if reject_late_palette and "write_array" in state.events:
            raise RuntimeError(
                "Cannot modify tag PhotometricInterpretation while writing")
        state.color_entries = {
            index: color_table.GetColorEntry(index) for index in (0, 1)}

    def write_array(_data, _x_off, _y_off):
        state.events.append("write_array")
        if fail_write:
            raise _InjectedGdalError("injected GDAL write failure")

    driver.Create = create
    output_band.SetRasterColorTable = set_color_table
    output_band.WriteArray = write_array
    output_band.FlushCache = lambda: state.events.append("flush_band")
    monkeypatch.setattr(
        raster_ops.gdal, "Open", lambda _path, _access: source_box.pop())
    monkeypatch.setattr(raster_ops.gdal, 'GetDriverByName', lambda _name: driver)
    return state


class TestMaskHandleLifecycle:
    def test_palette_is_attached_before_first_write(self, monkeypatch):
        state = _install_spy_gdal(monkeypatch, reject_late_palette=True)

        raster_ops.make_mask("input.tif", "output.tif", nodata_value=0)

        assert state.events.index("set_color_table") < state.events.index(
            "write_array")
        assert state.create_dtype == gdal.GDT_Byte
        assert state.creation_options == ["COMPRESS=PACKBITS", "NBITS=1"]
        assert state.color_entries == {
            0: (0, 0, 0, 255),
            1: (0, 255, 0, 255),
        }
        gc.collect()
        assert all(reference() is None for reference in state.references.values())

    def test_gdal_references_are_released_when_write_raises(self, monkeypatch):
        state = _install_spy_gdal(monkeypatch, fail_write=True)

        with pytest.raises(_InjectedGdalError) as error:
            raster_ops.make_mask("input.tif", "output.tif", nodata_value=0)

        assert error.traceback
        gc.collect()
        assert all(reference() is None for reference in state.references.values())


@pytest.mark.parametrize('ref_type,tgt_type,expected', [
    (gdal.GDT_Int16, gdal.GDT_UInt16, gdal.GDT_Int32),
    (gdal.GDT_Int32, gdal.GDT_UInt32, gdal.GDT_Float64),
    (gdal.GDT_Float32, gdal.GDT_UInt32, gdal.GDT_Float64),
    (gdal.GDT_Float32, gdal.GDT_UInt16, gdal.GDT_Float32),
])
def test_type_promotion_preserves_input_ranges(ref_type, tgt_type, expected):
    assert rio.promote_dtype(ref_type, tgt_type) == expected
    assert rio.promote_dtype(tgt_type, ref_type) == expected


@pytest.mark.parametrize('where', ['imad', 'radcal'])
def test_large_height_processing_never_reads_whole_bands(tmp_path, monkeypatch, where):
    build_pair(tmp_path, rows=600, cols=20)
    read_array = gdal.Band.ReadAsArray

    def bounded(band, x=0, y=0, cols=None, rows=None, *args, **kwargs):
        assert rows is not None and rows <= 256, 'Whole-band read would scale with scene height'
        return read_array(band, x, y, cols, rows, *args, **kwargs)

    monkeypatch.setattr(gdal.Band, 'ReadAsArray', bounded)
    if where == 'imad':
        iMad.main(str(tmp_path / 'ref.tif'), str(tmp_path / 'tgt.tif'),
                  max_iters=1, feedback=Feedback())
    else:
        mad = make_mad_raster(tmp_path, (600, 20), bands=4)
        radcal.main(mad, img_ref=str(tmp_path / 'ref.tif'), img_tgt=str(tmp_path / 'tgt.tif'),
                    output=str(tmp_path / 'out.tif'), feedback=Feedback())


@pytest.mark.parametrize('exceptions', [True, False])
def test_gdal_error_modes_are_supported_without_changing_them(tmp_path, exceptions):
    if not hasattr(gdal, 'ExceptionMgr'):
        pytest.skip('Scoped exception modes require GDAL 3.7+')
    with gdal.ExceptionMgr(useExceptions=exceptions):
        state = gdal.GetUseExceptions()
        with pytest.raises(RuntimeError, match='Cannot open raster'), rio.open_raster(tmp_path / 'missing.tif'):
            pass
        assert gdal.GetUseExceptions() == state


@pytest.mark.parametrize('sentinel,dtype', [(0.5, gdal.GDT_Int16), (-9999, gdal.GDT_Byte)])
def test_comparison_sentinel_does_not_round_into_a_valid_integer_pixel(sentinel, dtype):
    values = np.array([0, 1], dtype=rio.numpy_dtype(dtype))
    comparison = rio.input_nodata(sentinel, dtype)
    assert comparison == sentinel
    assert rio.valid_values(values, comparison).all()
