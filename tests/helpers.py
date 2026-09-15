"""Shared deterministic raster builders and feedback for the permanent test suites.

Import helpers from here, never from a test module. These utilities require
GDAL/NumPy, but do not create a QGIS application or configure Qt.
"""
import numpy as np
from ArrNorm.core.arrnorm import Normalization
from osgeo import gdal

_GDAL_TO_NP = {
    gdal.GDT_Byte: np.uint8,
    gdal.GDT_UInt16: np.uint16,
    gdal.GDT_Int16: np.int16,
    gdal.GDT_UInt32: np.uint32,
    gdal.GDT_Int32: np.int32,
    gdal.GDT_Float32: np.float32,
    gdal.GDT_Float64: np.float64,
}


class Feedback:
    """Processing feedback stub, optionally cancelling on a log message."""

    def __init__(self, cancel_when=None):
        self.messages = []
        self._cancel_when = cancel_when
        self._canceled = False
        self._progress = 0

    def pushInfo(self, msg):
        self.messages.append(msg)
        if self._cancel_when and self._cancel_when in msg:
            self._canceled = True

    def reportError(self, msg, fatalError=False):
        self.messages.append('ERROR: ' + str(msg))

    def setProgress(self, value):
        self._progress = value

    def isCanceled(self):
        return self._canceled


def write_raster(path, bands, dtype=gdal.GDT_Float32, nodata=None,
                 geotransform=(0.0, 10.0, 0.0, 0.0, 0.0, -10.0),
                 projection='EPSG:4326'):
    """Write a multiband fixture using GDAL directly, independently of core I/O."""
    rows, cols = bands[0].shape
    ds = gdal.GetDriverByName('GTiff').Create(str(path), cols, rows, len(bands), dtype)
    band_ds = None
    try:
        ds.SetGeoTransform(geotransform)
        ds.SetProjection(projection)
        for i, band in enumerate(bands, start=1):
            band_ds = ds.GetRasterBand(i)
            band_ds.WriteArray(band.astype(_GDAL_TO_NP[dtype]))
            if nodata is not None:
                band_ds.SetNoDataValue(float(nodata))
            band_ds.FlushCache()
    finally:
        band_ds = None
        ds = None


def read_raster(path):
    """Read a fixture/output; a single band is 2-D, multiple bands are 3-D."""
    ds = gdal.Open(str(path))
    try:
        return ds.ReadAsArray()
    finally:
        ds = None


def synthetic_pair(seed=3, rows=120, cols=130):
    """Correlated four-band reference/target arrays with mild radiometric drift."""
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 90, (rows, cols)).astype(np.float64)

    def noise():
        return rng.normal(0.0, 2.0, (rows, cols))

    ref = [base * 1.0 + noise(), base * 1.5 + 3.0 + noise(),
           base * 0.8 + 10.0 + noise(), base * 1.1 + 7.0 + noise()]
    tgt = [band * 1.2 + 2.0 + noise() for band in ref]
    return ref, tgt


def build_pair(workdir, *, ref_dtype=gdal.GDT_Float32, tgt_dtype=gdal.GDT_Float32,
               ref_nodata=None, tgt_nodata=None, ref_offset=(0.0, 0.0),
               seed=3, mutate_ref=None, mutate_tgt=None, rows=120, cols=130):
    """Write ref.tif and tgt.tif, optionally changing pixels or reference alignment."""
    ref, tgt = synthetic_pair(seed=seed, rows=rows, cols=cols)
    if mutate_ref:
        mutate_ref(ref)
    if mutate_tgt:
        mutate_tgt(tgt)
    write_raster(workdir / 'ref.tif', ref, dtype=ref_dtype, nodata=ref_nodata,
                 geotransform=(0.0 + ref_offset[0], 10.0, 0.0,
                               0.0 - ref_offset[1], 0.0, -10.0))
    write_raster(workdir / 'tgt.tif', tgt, dtype=tgt_dtype, nodata=tgt_nodata)


def make_normalization(workdir, **kw):
    """Construct, without running, a normalization of the synthetic fixture pair."""
    params = {
        'max_iters': kw.pop('max_iters', 8),
        'conv_threshold': kw.pop('conv_threshold', 0.99),
        'ncp_threshold': kw.pop('ncp_threshold', 0.95),
        'neg_to_nodata': kw.pop('neg_to_nodata', False),
        'mask_ref': kw.pop('mask_ref', False),
        'mask_ref_nodata': kw.pop('mask_ref_nodata', None),
        'nodata_mask': kw.pop('nodata_mask', False),
        'nodata_mask_value': kw.pop('nodata_mask_value', None),
        'keep_mask_layer': kw.pop('keep_mask_layer', False),
    }
    assert not kw, kw
    return Normalization(
        img_ref=str(workdir / 'ref.tif'),
        img_target=str(workdir / 'tgt.tif'),
        output_file=str(workdir / 'out.tif'),
        feedback=Feedback(),
        **params)


def make_mad_raster(workdir, shape, bands=2):
    """Write dummy MAD bands and zero chi-square (all pixels are no-change).

The arbitrary filename intentionally cannot be used to infer input paths.
"""
    path = workdir / 'arbitrary statistics name.tif'
    write_raster(path, [np.zeros(shape)] * (bands + 1))
    return str(path)
