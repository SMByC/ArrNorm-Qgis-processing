#!/usr/bin/env python3
"""IR-MAD invariant-pixel radiometric calibration (GPLv2+).

Fits reference = intercept + slope * target with streaming TLS moments.
Both fitting and application are block-wise. Report diagnostics are collected
during the fitting/application passes: affine-model agreement from the moments,
a bounded uniform sample, invariant-pixel density, full fit/application ranges
and conversion counts. See `report.py`; diagnostics never change the fit.
"""
import ast
import getopt
import importlib.util
import os
import sys
from contextlib import ExitStack
from typing import Any

import numpy as np
from scipy import stats

from . import raster_io as rio
from . import report
from .auxil.auxil import OrthogonalFit

try:
    from qgis.core import QgsProcessingException
except ImportError:
    class _ProcessingException(RuntimeError):
        pass
    QgsProcessingException = _ProcessingException

def _tile(dataset, pos, x, y, cols, rows):
    return np.column_stack([rio.read_band(dataset.GetRasterBand(b), x, y, cols, rows).ravel()
                            for b in pos]).astype(np.float64)


def _paths(img_imad, img_ref, img_tgt, output):
    if img_ref is None or img_tgt is None:
        root, ext = os.path.splitext(os.path.basename(img_imad))
        if not root.startswith('MAD(') or not root.endswith(')') or root.count('&') != 1:
            raise QgsProcessingException('Cannot infer inputs from the MAD filename; '
                                         'supply img_ref and img_tgt explicitly.')
        reference, target = root[4:-1].split('&')
        directory = os.path.dirname(os.path.abspath(img_imad))
        img_ref = img_ref or os.path.join(directory, reference + ext)
        img_tgt = img_tgt or os.path.join(directory, target)
    if output is None:
        output = os.path.splitext(img_tgt)[0] + '_norm.tif'
    return img_ref, img_tgt, output


def _apply(target, output, pos, coefficients, out_dtype, nodata_tgt, neg_nodata,
           feedback, block_rows, spatial, window=None, *, ranges=None, conversion=None):
    x0, y0, cols, rows = window or (0, 0, target.RasterXSize, target.RasterYSize)
    output_nd = nodata_tgt if nodata_tgt is not None else neg_nodata
    if output_nd is not None:
        output_nd = rio.validate_nodata(output_nd, out_dtype)
    target_sentinels = rio.band_nodata(target, nodata_tgt, pos)
    with rio.create_raster(output, cols, rows, len(pos), out_dtype) as dst:
        rio.copy_spatial(spatial, dst)
        for j in range(1, len(pos) + 1):
            if output_nd is not None:
                rio.checked(dst.GetRasterBand(j).SetNoDataValue(output_nd), 'Set output nodata')
        for y, nr in rio.row_blocks(rows, block_rows):
            rio.check_cancel(feedback)
            tile = _tile(target, pos, x0, y0 + y, cols, nr)
            valid = rio.valid_rows(tile, target_sentinels)
            if ranges is not None:
                ranges.update(tile[valid])
            for j, (slope, intercept, _) in enumerate(coefficients, 1):
                normalized = intercept + slope * tile[:, j - 1]
                # Reject arithmetic failures before clipping can hide them or
                # intentional NaN nodata is inserted into otherwise valid pixels.
                if not np.isfinite(normalized[valid]).all():
                    raise QgsProcessingException('Non-finite calibration output for valid pixels.')
                negative = normalized < 0
                clipped = rio.clip_values(normalized, out_dtype)
                if conversion is not None:
                    masked = negative if neg_nodata is not None else np.zeros(len(valid), bool)
                    conversion[j - 1]['valid'] += int(np.count_nonzero(valid))
                    conversion[j - 1]['negative_nodata'] += int(np.count_nonzero(valid & masked))
                    conversion[j - 1]['clipped'] += int(np.count_nonzero(
                        valid & ~masked & (normalized != clipped)))
                normalized = clipped
                if neg_nodata is not None:
                    normalized[negative] = neg_nodata
                if nodata_tgt is not None:
                    normalized[~valid] = output_nd
                rio.write_band(dst.GetRasterBand(j), normalized.reshape(nr, cols), 0, y)


def matplotlib_available():
    """True when matplotlib is importable; the report figures are skipped otherwise."""
    try:
        return importlib.util.find_spec('matplotlib') is not None
    except (ImportError, ValueError):
        return False


def main(img_imad, ncp_threshold=0.95, pos=None, dims=None, img_target=None,
         graphics=False, out_dtype=None, img_ref=None, img_tgt=None,
         output=None, feedback=None, *, neg_nodata=None, nodata_ref=None,
         nodata_tgt=None, report_path=None, report_staging=None, report_written=None,
         convergence=None, block_rows=rio.DEFAULT_BLOCK_ROWS, report_context=None,
         report_inputs=()):
    """Calibrate all selected bands, preserving their order and shared validity.

    Explicit input paths bypass legacy MAD filename inference. Cancellation
    returns None only after every dataset has been closed.

    `report_path` names the report destination. A caller that publishes complete
    outputs only passes `report_staging` to have the figures written in its own
    workspace and `report_written` to collect what to publish from there.
    """
    def info(message):
        if feedback is None:
            print(message)
        else:
            feedback.pushInfo(message)

    rio.validate_options(1, 0.99, ncp_threshold)
    img_ref, img_tgt, output = _paths(img_imad, img_ref, img_tgt, output)
    try:
        with ExitStack() as stack:
            rio.check_cancel(feedback)
            mad = stack.enter_context(rio.open_raster(img_imad))
            ref = stack.enter_context(rio.open_raster(img_ref))
            tgt = stack.enter_context(rio.open_raster(img_tgt))
            protected = [img_imad, img_ref, img_tgt, output, *report_inputs]
            for dataset in (mad, ref, tgt):
                protected.extend(dataset.GetFileList() or [])
            pos = list(pos) if pos is not None else list(range(1, ref.RasterCount + 1))
            if not pos or any(b < 1 or b > min(ref.RasterCount, tgt.RasterCount) for b in pos):
                raise QgsProcessingException('Invalid calibration band selection.')
            x0, y0, cols, rows = dims or (0, 0, mad.RasterXSize, mad.RasterYSize)
            if (cols <= 0 or rows <= 0 or x0 < 0 or y0 < 0 or mad.RasterCount < 2
                    or cols > mad.RasterXSize or rows > mad.RasterYSize
                    or any(x0 + cols > ds.RasterXSize or y0 + rows > ds.RasterYSize
                           for ds in (ref, tgt))):
                raise QgsProcessingException('Calibration window is outside the input rasters.')
            if out_dtype is None:
                out_dtype = rio.promote_dtype(*(ds.GetRasterBand(b).DataType
                                               for ds in (ref, tgt) for b in pos))
            for value in (nodata_tgt, neg_nodata):
                if value is not None:
                    rio.validate_nodata(value, out_dtype)
            if neg_nodata is not None:
                neg_nodata = rio.validate_nodata(neg_nodata, out_dtype)

            ref_sentinels = rio.band_nodata(ref, nodata_ref, pos)
            tgt_sentinels = rio.band_nodata(tgt, nodata_tgt, pos)

            fits = [OrthogonalFit() for _ in pos]
            # Diagnostics reduce the masks and tiles this pass already reads, so
            # the report costs no extra I/O and no unbounded memory.
            sample = report.UniformSample() if graphics else None
            density = report.DensityGrid(cols, rows) if graphics else None
            fit_ranges = report.ValueRange(len(pos)) if graphics else None
            # `ranges` is filled by the application pass below, not here: it
            # spans every valid target value the correction is applied to, which
            # is wider than the invariant pixels this fit pass sees.
            ranges = report.ValueRange(len(pos)) if graphics else None
            total = 0
            for y, nr in rio.row_blocks(rows, block_rows):
                rio.check_cancel(feedback)
                ref_tile = _tile(ref, pos, x0, y0 + y, cols, nr)
                tgt_tile = _tile(tgt, pos, x0, y0 + y, cols, nr)
                chisqr = rio.read_band(mad.GetRasterBand(mad.RasterCount), 0, y, cols, nr).ravel()
                # One fresh shared mask per block; no band can mutate another
                # band's regression sample set. Exclude gaps in any channel.
                valid = (rio.valid_rows(ref_tile, ref_sentinels)
                         & rio.valid_rows(tgt_tile, tgt_sentinels))
                selected = (stats.chi2.sf(chisqr, mad.RasterCount - 1) > ncp_threshold) & valid
                x, y_ref = tgt_tile[selected], ref_tile[selected]
                total += len(x)
                for j, fit in enumerate(fits):
                    fit.update(x[:, j], y_ref[:, j])
                # Equivalent to `if graphics`, written per collector so the type
                # checker can narrow each one away from None.
                if sample is not None and density is not None and fit_ranges is not None:
                    sample.update(x, y_ref)
                    density.update(y, selected, valid, cols)
                    fit_ranges.update(x)
            info(f'no-change pixels used for calibration: {total:,}')
            if total < 2:
                raise QgsProcessingException(f'Only {total} no-change pixels remain after excluding '
                                             'nodata. Check overlap, nodata settings and threshold.')
            try:
                coefficients = [fit.coefficients() for fit in fits]
            except ValueError as exc:
                raise QgsProcessingException(str(exc)) from exc
            info('Applying calibration to the target...')
            conversion = [{'valid': 0, 'clipped': 0, 'negative_nodata': 0} for _ in pos]
            _apply(tgt, output, pos, coefficients, out_dtype, nodata_tgt, neg_nodata,
                   feedback, block_rows, mad, (x0, y0, cols, rows),
                   ranges=ranges if img_target is None else None,
                   conversion=conversion if img_target is None else None)
            if img_target is not None:
                full = stack.enter_context(rio.open_raster(img_target))
                output = os.path.splitext(img_target)[0] + '_norm_all.tif'
                protected.extend([img_target, output, *(full.GetFileList() or [])])
                _apply(full, output, pos, coefficients, out_dtype, nodata_tgt,
                       neg_nodata, feedback, block_rows, full, ranges=ranges, conversion=conversion)
            rio.check_cancel(feedback)
            # These describe the affine model, not saved pixels. Application
            # ranges and conversion counts come from the existing output pass.
            context = {'reference': img_ref, 'target': img_target or img_tgt, 'output': output,
                       **(report_context or {}), 'threshold': ncp_threshold,
                       'dtype': rio.gdal.GetDataTypeName(out_dtype), 'conversion': conversion}
            figures = (report_path or os.path.splitext(output)[0] + '_report.png'
                       if graphics else None)
            report.emit(feedback, info, pos, coefficients, fits, path=figures,
                        staging=report_staging, written=report_written, sample=sample,
                        density=density, ranges=ranges, convergence=convergence,
                        fit_ranges=fit_ranges,
                        context=context, protected_paths=protected)
            completion = f'Calibration complete ({len(pos)} bands).'
            # The pipeline announces its final publication itself; direct callers
            # still need the destination, without exposing a pipeline workspace.
            if context['output'] == output:
                completion += f' Output: {output}'
            info(completion)
            return output
    except rio.Cancelled:
        return None


if __name__ == '__main__':
    options, args = getopt.getopt(sys.argv[1:], 'hnp:d:t:')
    kwargs: dict[str, Any] = {'graphics': True}
    for option, value in options:
        if option == '-h':
            print('Usage: python -m ArrNorm.core.radcal [-n] [-p "bands"] [-d "window"] '
                  '[-t probability] MADfile [fullSceneFile]')
            sys.exit()
        if option == '-n':
            kwargs['graphics'] = False
        elif option == '-p':
            kwargs['pos'] = ast.literal_eval(value)
        elif option == '-d':
            kwargs['dims'] = ast.literal_eval(value)
        elif option == '-t':
            kwargs['ncp_threshold'] = float(value)
    if len(args) not in (1, 2):
        sys.exit('Expected MADfile and optionally fullSceneFile.')
    main(args[0], img_target=args[1] if len(args) == 2 else None, **kwargs)
