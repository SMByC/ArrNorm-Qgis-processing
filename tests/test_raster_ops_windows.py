import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
from ArrNorm.core import raster_ops
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
    source_ds = _GdalHandle(RasterXSize=2, RasterYSize=2)
    output_ds = _GdalHandle(RasterXSize=2, RasterYSize=2)
    driver = _GdalHandle()
    source_ds.GetDriver = lambda: driver
    source_ds.GetRasterBand = lambda _index: source_band
    source_ds.GetGeoTransform = lambda: None
    source_ds.GetProjection = lambda: None
    output_ds.GetRasterBand = lambda _index: output_band

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
    return state


class TestMakeMaskWindows:
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
