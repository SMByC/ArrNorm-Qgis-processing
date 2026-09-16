#!/usr/bin/env python3
"""IR-MAD invariant-pixel radiometric calibration (GPLv2+).

Fits reference = intercept + slope * target with streaming TLS moments.
Both fitting and application are block-wise; optional plots keep at most
10,000 samples and do not select or change the fit's calibration samples.
"""
import ast
import getopt
import os
import sys
from contextlib import ExitStack
from typing import Any

import numpy as np
from scipy import stats

from . import raster_io as rio
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
           feedback, block_rows, spatial, window=None):
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
            for j, (slope, intercept, _) in enumerate(coefficients, 1):
                normalized = intercept + slope * tile[:, j - 1]
                # Reject arithmetic failures before clipping can hide them or
                # intentional NaN nodata is inserted into otherwise valid pixels.
                if not np.isfinite(normalized[valid]).all():
                    raise QgsProcessingException('Non-finite calibration output for valid pixels.')
                negative = normalized < 0
                normalized = rio.clip_values(normalized, out_dtype)
                if neg_nodata is not None:
                    normalized[negative] = neg_nodata
                if nodata_tgt is not None:
                    normalized[~valid] = output_nd
                rio.write_band(dst.GetRasterBand(j), normalized.reshape(nr, cols), 0, y)


def _plot(samples, coefficients, pos, output, total):
    # Use an Agg canvas directly; never change another plugin's pyplot backend.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    count = min(len(pos), 6)
    figure = Figure(figsize=(4.2 * min(count, 3), 4.2 * (1 if count <= 3 else 2)))
    FigureCanvasAgg(figure)
    for j in range(count):
        ax = figure.add_subplot(1 if count <= 3 else 2, min(count, 3), j + 1)
        x, y = samples[0][:, j], samples[1][:, j]
        slope, intercept, corr = coefficients[j]
        ax.scatter(x, y, s=1, alpha=0.25)
        bounds = np.array([x.min(), x.max()])
        ax.plot(bounds, intercept + slope * bounds, color='crimson')
        ax.set_title(f'Band {pos[j]}: R²={corr ** 2:.3f}')
        ax.set_xlabel('Target')
        ax.set_ylabel('Reference')
    figure.suptitle(f'Radcal: {total:,} invariant pixels; displaying up to 10,000')
    figure.tight_layout()
    plot_path = os.path.splitext(output)[0] + '_radcal.png'
    figure.savefig(plot_path, dpi=150)
    return plot_path


def main(img_imad, ncp_threshold=0.95, pos=None, dims=None, img_target=None,
         graphics=False, out_dtype=None, img_ref=None, img_tgt=None,
         output=None, feedback=None, *, neg_nodata=None, nodata_ref=None,
         nodata_tgt=None, block_rows=rio.DEFAULT_BLOCK_ROWS):
    """Calibrate all selected bands, preserving their order and shared validity.

    Explicit input paths bypass legacy MAD filename inference. Cancellation
    returns None only after every dataset has been closed.
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
            samples_ref, samples_tgt = [], []
            sample_count = total = 0
            for y, nr in rio.row_blocks(rows, block_rows):
                rio.check_cancel(feedback)
                ref_tile = _tile(ref, pos, x0, y0 + y, cols, nr)
                tgt_tile = _tile(tgt, pos, x0, y0 + y, cols, nr)
                chisqr = rio.read_band(mad.GetRasterBand(mad.RasterCount), 0, y, cols, nr).ravel()
                # One fresh shared mask per block; no band can mutate another
                # band's regression sample set. Exclude gaps in any channel.
                selected = ((stats.chi2.sf(chisqr, mad.RasterCount - 1) > ncp_threshold)
                            & rio.valid_rows(ref_tile, ref_sentinels)
                            & rio.valid_rows(tgt_tile, tgt_sentinels))
                x, y_ref = tgt_tile[selected], ref_tile[selected]
                total += len(x)
                for j, fit in enumerate(fits):
                    fit.update(x[:, j], y_ref[:, j])
                if graphics and sample_count < 10000:
                    size = min(len(x), 10000 - sample_count)
                    samples_ref.append(y_ref[:size].copy())
                    samples_tgt.append(x[:size].copy())
                    sample_count += size
            info(f'no-change pixels used for calibration: {total}')
            if total < 2:
                raise QgsProcessingException(f'Only {total} no-change pixels remain after excluding '
                                             'nodata. Check overlap, nodata settings and threshold.')
            try:
                coefficients = [fit.coefficients() for fit in fits]
            except ValueError as exc:
                raise QgsProcessingException(str(exc)) from exc
            for b, (slope, intercept, corr) in zip(pos, coefficients):
                info(f'band {b}: slope={slope:.6f}, intercept={intercept:.6f}, '
                     f'correlation={corr:.6f}')
            _apply(tgt, output, pos, coefficients, out_dtype, nodata_tgt, neg_nodata,
                   feedback, block_rows, mad, (x0, y0, cols, rows))
            if graphics:
                try:
                    plot_path = _plot((np.concatenate(samples_tgt), np.concatenate(samples_ref)),
                                      coefficients, pos, output, total)
                    info(f'plot written to: {plot_path}')
                except ImportError:
                    info('Matplotlib is unavailable; graphics disabled.')
            if img_target is not None:
                full = stack.enter_context(rio.open_raster(img_target))
                output = os.path.splitext(img_target)[0] + '_norm_all.tif'
                _apply(full, output, pos, coefficients, out_dtype, nodata_tgt,
                       neg_nodata, feedback, block_rows, full)
            rio.check_cancel(feedback)
            info(f'\nRadiometric calibration applied to the target image ({len(pos)} bands)')
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
