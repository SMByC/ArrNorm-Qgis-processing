#!/usr/bin/env python3
"""Block-wise raster utilities; outputs are writable GeoTIFFs (GPLv2+).

Validity is shared with calibration: a spectral pixel is usable only when
all bands are finite and different from the configured nodata value.
no_negative_value is retained as a standalone utility.
"""
import numpy as np
from osgeo import gdal

from . import raster_io as rio

DEFAULT_BLOCK_ROWS = rio.DEFAULT_BLOCK_ROWS


def _check_nodata_representable(nodata_value, gdal_dtype):
    try:
        return rio.validate_nodata(nodata_value, gdal_dtype)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def no_negative_value(input_path, output_path, nodata_value=None,
                      creation_options=None, block_rows=DEFAULT_BLOCK_ROWS, feedback=None):
    """Convert negatives and existing nodata to a validated output sentinel."""
    with rio.open_raster(input_path) as src:
        dtype = src.GetRasterBand(1).DataType
        nodata = []
        for b in range(1, src.RasterCount + 1):
            source_nd = src.GetRasterBand(b).GetNoDataValue()
            value = nodata_value if nodata_value is not None else source_nd
            nodata.append(_check_nodata_representable(value if value is not None else 0, dtype))
        rio.check_cancel(feedback)
        with rio.create_raster(output_path, src.RasterXSize, src.RasterYSize,
                               src.RasterCount, dtype, creation_options) as dst:
            rio.copy_spatial(src, dst)
            for b, nd in enumerate(nodata, 1):
                source, output = src.GetRasterBand(b), dst.GetRasterBand(b)
                rio.checked(output.SetNoDataValue(nd), 'Set nodata')
                for y, nr in rio.row_blocks(src.RasterYSize, block_rows):
                    rio.check_cancel(feedback)
                    data = rio.read_band(source, 0, y, src.RasterXSize, nr)
                    valid = rio.valid_values(data, source.GetNoDataValue()) & (data >= 0)
                    rio.write_band(output, np.where(valid, data, nd), 0, y)
                output.SetDescription(source.GetDescription())
                rio.checked(output.FlushCache(), 'Flush converted band')


def make_mask(input_path, output_path, nodata_value, block_rows=DEFAULT_BLOCK_ROWS,
              feedback=None):
    """Create a single-band Byte mask: 1 if every band is valid, otherwise 0."""
    with rio.open_raster(input_path) as src:
        cols, rows = src.RasterXSize, src.RasterYSize
        rio.check_cancel(feedback)
        with rio.create_raster(output_path, cols, rows, 1, gdal.GDT_Byte,
                               ['COMPRESS=PACKBITS', 'NBITS=1']) as dst:
            rio.copy_spatial(src, dst)
            output = dst.GetRasterBand(1)
            sentinels = rio.band_nodata(src, nodata_value, range(1, src.RasterCount + 1))
            try:
                colors = gdal.ColorTable()
                colors.SetColorEntry(0, (0, 0, 0, 255))
                colors.SetColorEntry(1, (0, 255, 0, 255))
                rio.checked(output.SetRasterColorTable(colors), 'Set mask palette')
                for y, nr in rio.row_blocks(rows, block_rows):
                    rio.check_cancel(feedback)
                    valid = np.ones((nr, cols), dtype=bool)
                    for b in range(1, src.RasterCount + 1):
                        data = rio.read_band(src.GetRasterBand(b), 0, y, cols, nr)
                        valid &= rio.valid_values(data, sentinels[b - 1] if sentinels is not None else None)
                    rio.write_band(output, valid.astype(np.uint8), 0, y)
                rio.checked(output.FlushCache(), 'Flush mask')
            finally:
                output = None


def apply_mask(image_path, mask_path, output_path, nodata_value,
               creation_options=None, block_rows=DEFAULT_BLOCK_ROWS, feedback=None):
    """Fill masked-out pixels with nodata, preserving existing source nodata."""
    with rio.open_raster(image_path) as src, rio.open_raster(mask_path) as mask:
        cols, rows = src.RasterXSize, src.RasterYSize
        if (mask.RasterXSize, mask.RasterYSize) != (cols, rows):
            raise RuntimeError("Mask dimensions don't match image dimensions")
        if (mask.GetGeoTransform() != src.GetGeoTransform()
                or mask.GetProjection() != src.GetProjection()):
            raise RuntimeError('Mask and image must share the same pixel grid and CRS.')
        dtype = src.GetRasterBand(1).DataType
        nd = _check_nodata_representable(nodata_value, dtype)
        rio.check_cancel(feedback)
        with rio.create_raster(output_path, cols, rows, src.RasterCount,
                               dtype, creation_options) as dst:
            rio.copy_spatial(src, dst)
            for b in range(1, src.RasterCount + 1):
                source, output = src.GetRasterBand(b), dst.GetRasterBand(b)
                rio.checked(output.SetNoDataValue(nd), 'Set nodata')
                for y, nr in rio.row_blocks(rows, block_rows):
                    rio.check_cancel(feedback)
                    data = rio.read_band(source, 0, y, cols, nr)
                    valid = rio.read_band(mask.GetRasterBand(1), 0, y, cols, nr) == 1
                    valid &= rio.valid_values(data, source.GetNoDataValue())
                    rio.write_band(output, np.where(valid, data, nd), 0, y)
                output.SetDescription(source.GetDescription())
                rio.checked(output.FlushCache(), 'Flush masked band')
