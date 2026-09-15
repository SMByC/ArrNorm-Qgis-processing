"""Shared raster validity, numeric types and checked GDAL I/O.

Invalid means non-finite or equal to the configured nodata sentinel. A
multispectral pixel is valid only when every participating band is valid.
GDAL's process-global exception mode is deliberately left untouched.
"""
import math

import numpy as np
from osgeo import gdal, gdal_array

DEFAULT_BLOCK_ROWS = 256


class Cancelled(Exception):
    """Cooperative cancellation, handled by the pipeline boundary."""


def check_cancel(feedback):
    if feedback is not None and feedback.isCanceled():
        raise Cancelled()


def row_blocks(rows, block_rows=DEFAULT_BLOCK_ROWS):
    if block_rows < 1:
        raise ValueError('Block height must be positive.')
    for y in range(0, rows, block_rows):
        yield y, min(block_rows, rows - y)


def valid_values(data, nodata=None):
    valid = np.isfinite(data)
    if nodata is not None:
        # A NaN sentinel needs no special comparison: non-finite pixels have
        # already been excluded. This also supports one sentinel per column.
        valid &= data != nodata
    return valid


def valid_rows(tile, nodata=None):
    return valid_values(tile, nodata).all(axis=1)


def numpy_dtype(code):
    dtype = gdal_array.GDALTypeCodeToNumericTypeCode(code)
    if dtype is None or np.dtype(dtype).kind not in 'iuf':
        raise ValueError(f'Unsupported real raster data type: {gdal.GetDataTypeName(code)}')
    return np.dtype(dtype)


def input_nodata(value, code):
    """Express a comparison sentinel in a band's storage precision.

This is distinct from output validation: an integer band cannot contain a
fractional/out-of-range sentinel, so keep it non-matching rather than round
it to a valid pixel. Float sentinels must match stored pixels even after
those pixels have been widened to float64 for statistics.
"""
    if value is None:
        return None
    dtype = numpy_dtype(code)
    if dtype.kind == 'f' and math.isfinite(value) and abs(value) <= np.finfo(dtype).max:
        return float(dtype.type(value))
    return value


def band_nodata(dataset, value, bands):
    """Comparison sentinels for selected bands, including mixed-dtype VRTs."""
    if value is None:
        return None
    return [input_nodata(value, dataset.GetRasterBand(b).DataType) for b in bands]


def promote_dtype(*codes):
    """Preserve both input ranges; Int16 + UInt16 -> Int32, Float32 + Int32 -> Float64.

The numerical pipeline uses float64. 64-bit integer input values outside
its exact integer range are rejected by read_band instead of rounded silently.
"""
    dtype = np.result_type(*(numpy_dtype(code) for code in codes))
    if dtype.kind in 'iu' and dtype.itemsize == 8:
        dtype = np.dtype('float64')
    code = gdal_array.NumericTypeCodeToGDALTypeCode(dtype)
    if code is None:
        raise ValueError(f'Unsupported promoted data type: {dtype}')
    return code


def validate_nodata(value, code):
    """Return the stored sentinel, or reject an unrepresentable value before I/O."""
    dtype = numpy_dtype(code)
    if dtype.kind in 'iu':
        limits = np.iinfo(dtype)
        if not math.isfinite(value) or int(value) != value or not limits.min <= value <= limits.max:
            raise ValueError(f'nodata value {value} is not representable in {dtype}; '
                             f'use an integer in [{limits.min}, {limits.max}].')
        return int(value)
    if np.isnan(value):
        return float('nan')
    if not math.isfinite(value) or abs(value) > np.finfo(dtype).max:
        raise ValueError(f'nodata value {value} is not representable in {dtype}.')
    # Metadata and pixels must agree after rounding to the storage precision.
    return float(dtype.type(value))


def clip_values(data, code):
    dtype = numpy_dtype(code)
    if dtype.kind in 'iu':
        limits = np.iinfo(dtype)
        return np.clip(data, limits.min, limits.max)
    return data


def checked(status, operation):
    # Older GDAL methods return None on success; newer ones return CE_None.
    if status not in (None, gdal.CE_None):
        raise RuntimeError(f'{operation}: {gdal.GetLastErrorMsg() or "GDAL operation failed"}')


class Raster:
    """Own a GDAL handle, including on exceptions which retain Python tracebacks.

The proxy clears its handle on exit. On GDAL >= 3.8 Close is explicit;
older bindings release their last dataset reference without depending on
the caller's local variable (or its traceback) going out of scope.
"""
    def __init__(self, dataset, writable=False):
        if dataset is None:
            raise RuntimeError(gdal.GetLastErrorMsg() or 'Cannot open/create raster')
        self._dataset = dataset
        self._writable = writable

    def __getattr__(self, name):
        return getattr(self._dataset, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        dataset, self._dataset = self._dataset, None
        if dataset is None:
            return
        try:
            if self._writable and exc_type is None:
                checked(dataset.FlushCache(), 'Flush raster')
        finally:
            try:
                close = getattr(dataset, 'Close', None)
                if close is not None:
                    try:
                        status = close()
                        if exc_type is None:
                            checked(status, 'Close raster')
                    except RuntimeError:
                        if exc_type is None:
                            raise
            finally:
                dataset = None


def open_raster(path):
    try:
        return Raster(gdal.Open(str(path), gdal.GA_ReadOnly))
    except (RuntimeError, TypeError) as exc:
        raise RuntimeError(f'Cannot open raster: {path}: {exc}') from exc


def create_raster(path, cols, rows, bands, code, options=None):
    numpy_dtype(code)
    return Raster(gdal.GetDriverByName('GTiff').Create(
        str(path), cols, rows, bands, code,
        list(options) if options is not None else ['BIGTIFF=IF_SAFER']), writable=True)


def read_band(band, x, y, cols, rows):
    data = band.ReadAsArray(x, y, cols, rows)
    if data is None:
        raise RuntimeError(f'Cannot read raster block at ({x}, {y}): {gdal.GetLastErrorMsg()}')
    if (data.dtype.kind in 'iu' and data.dtype.itemsize == 8
            and (np.any(data > 2 ** 53) or (data.dtype.kind == 'i' and np.any(data < -(2 ** 53))))):
        raise ValueError('64-bit integer pixels outside [-2^53, 2^53] cannot be '
                         'normalized exactly using float64 arithmetic.')
    return data


def write_band(band, data, x=0, y=0):
    try:
        checked(band.WriteArray(data, x, y), f'Write raster block at ({x}, {y})')
    finally:
        # Drop this frame's reference when a caller retains the exception
        # traceback. Dataset ownership/closing remains with Raster, not here.
        band = None


def copy_spatial(src, dst):
    gt, projection = src.GetGeoTransform(), src.GetProjection()
    if gt is not None:
        checked(dst.SetGeoTransform(gt), 'Set geotransform')
    if projection is not None:
        checked(dst.SetProjection(projection), 'Set projection')


def validate_options(max_iters, conv_threshold, ncp_threshold):
    if not isinstance(max_iters, (int, np.integer)) or max_iters < 1:
        raise ValueError('Maximum iterations must be a positive integer.')
    for name, value in (('Convergence threshold', conv_threshold),
                        ('No-change probability threshold', ncp_threshold)):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{name} must be between 0 and 1.')
