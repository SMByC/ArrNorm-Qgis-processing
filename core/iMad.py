#!/usr/bin/env python3
# ******************************************************************************
#  Name:     iMad.py
#  Purpose:  Iteratively Reweighted Multivariate Alteration Detection
#            (IR-MAD) for bi-temporal multispectral imagery.
#
#  Original algorithm: M. J. Canty (2014). Refactored for numerical
#  stability and performance: see comments inline.
#
#  License: GPLv2+
# ******************************************************************************

import os
import sys
import time
from contextlib import ExitStack
from operator import itemgetter

import numpy as np
from osgeo import gdal
from osgeo.gdalconst import GDT_Float32
from scipy import stats

from ArrNorm.core import raster_io as rio
from ArrNorm.core.auxil import auxil

try:
    from qgis.core import QgsProcessingException
except ImportError:
    QgsProcessingException = Exception


class _FatalInputError(QgsProcessingException):
    """Deliberate fatal error for degenerate input; never swallowed by the
    per-iteration fallback handler."""

# Block height (in rows) used when streaming both images band-by-band into
# the running covariance accumulator. Reading 256 rows at a time amortizes
# the per-call GDAL overhead by ~256x vs the original row-by-row loop
# while keeping peak memory bounded (256 * cols * 2*bands * 8 bytes).
DEFAULT_BLOCK_ROWS = 256

# Shared with Processing: the tolerance and iteration cap follow Canty's
# reference implementation. See README.md for the rationale and references.
DEFAULT_MAX_ITERS = 50
DEFAULT_CONV_THRESHOLD = 0.999


_iter_row_blocks = rio.row_blocks


def _has_nonzero(band, x, y, cols, rows, block_rows, feedback):
    for offset, count in rio.row_blocks(rows, block_rows):
        rio.check_cancel(feedback)
        if rio.read_band(band, x, y + offset, cols, count).any():
            return True
    return False


def _read_block(raster_bands, x0, y0, cols, n_rows):
    """Read a (n_rows, cols, bands) float64 tile for the given band list."""
    bands = len(raster_bands)
    tile = np.empty((n_rows * cols, bands), dtype=np.float64)
    for k, rb in enumerate(raster_bands):
        arr = rio.read_band(rb, x0, y0, cols, n_rows)
        tile[:, k] = arr.ravel()
    return tile


def _valid_rows(tile, nodata):
    """True only where all bands are finite and differ from any configured sentinels."""
    return rio.valid_rows(tile, nodata)


def main(img_ref, img_target, max_iters=DEFAULT_MAX_ITERS,
         conv_threshold=DEFAULT_CONV_THRESHOLD, band_pos=None, dims=None,
         graphics=False, ref_text='', block_rows=DEFAULT_BLOCK_ROWS, feedback=None, *,
         output_dir=None, nodata_ref=None, nodata_tgt=None, output=None,
         convergence=None, convergence_info=None):
    """Write the MAD variates and chi-square band; return the written path.

    Defaults stop at a maximum inter-iteration correlation change below 0.001,
    with a 50-iteration cap. This controls numerical convergence, not confidence
    or guaranteed radiometric accuracy.

    A list passed as `convergence` receives the canonical correlations of every
    completed iteration, which the calibration report plots. The return value
    stays the output path so existing callers are unaffected. `graphics` is
    accepted for signature compatibility only and is not forwarded any further:
    this module no longer draws anything, because a Processing worker must not
    open pyplot windows. `convergence_info` receives the termination status,
    the completed `iterations`, the `selected_iteration`, the `delta_threshold`
    and the `final_delta` (unavailable after only one completed iteration) —
    all of which the report's convergence panel labels.
    """
    rio.validate_options(max_iters, conv_threshold, 0.95)
    try:
        with ExitStack() as stack:
            return _main(img_ref, img_target, max_iters, conv_threshold, band_pos, dims,
                         ref_text, block_rows, output_dir, nodata_ref,
                         nodata_tgt, feedback, output, stack, convergence, convergence_info)
    except rio.Cancelled:
        return None


def _main(img_ref, img_target, max_iters, conv_threshold, band_pos, dims,
          ref_text, block_rows, output_dir, nodata_ref, nodata_tgt,
          feedback, output, stack, convergence=None, convergence_info=None):
    gdal.AllRegister()
    start = time.time()

    # -- Logging helpers: use QGIS feedback when available, print otherwise --
    def _info(msg):
        if feedback is not None:
            feedback.pushInfo(msg)
        else:
            print(msg)

    def _error(msg):
        if feedback is None:
            print(msg, file=sys.stderr)
        raise _FatalInputError(msg)

    def _canceled():
        return feedback is not None and feedback.isCanceled()

    path = os.path.dirname(os.path.abspath(img_ref))
    basename1 = os.path.basename(img_ref)
    root1 = os.path.splitext(basename1)[0]
    basename2 = os.path.basename(img_target)
    root2, _ext2 = os.path.splitext(basename2)
    directory = output_dir if output_dir is not None else path
    outfn = output or os.path.join(directory, f'MAD({root1}&{basename2}).tif')

    rio.check_cancel(feedback)
    inDataset1 = stack.enter_context(rio.open_raster(img_ref))
    inDataset2 = stack.enter_context(rio.open_raster(img_target))

    cols = inDataset1.RasterXSize
    rows = inDataset1.RasterYSize
    bands = inDataset1.RasterCount
    cols2 = inDataset2.RasterXSize
    rows2 = inDataset2.RasterYSize
    bands2 = inDataset2.RasterCount

    if bands != bands2:
        _error(
            f"Band count mismatch between reference ({bands}) "
            f"and target ({bands2}).")

    if band_pos is None:
        band_pos = list(range(1, bands + 1))
    else:
        bands = len(band_pos)
    if not bands or any(b < 1 or b > inDataset1.RasterCount for b in band_pos):
        _error('Invalid band selection.')

    if dims is None:
        x0 = y0 = 0
    else:
        x0, y0, cols, rows = dims

    # If the target was warped during registration we assume it is already
    # aligned to the reference origin; otherwise use the same window.
    if root2.find('_warp') != -1:
        x2 = y2 = 0
    else:
        x2, y2 = x0, y0

    # The pipeline aligns the reference before calling IR-MAD. These checks
    # also validate windows supplied by direct callers.
    if (cols < 1 or rows < 1 or x0 < 0 or y0 < 0 or x2 < 0 or y2 < 0
            or x0 + cols > inDataset1.RasterXSize or y0 + rows > inDataset1.RasterYSize
            or x2 + cols > cols2 or y2 + rows > rows2
            or (dims is None and (cols != cols2 or rows != rows2))):
        _error(
            f"\n ERROR: Reference clip ({cols}x{rows}) and target "
            f"({cols2}x{rows2}) have different pixel dimensions.\n"
            f" Pixel-for-pixel alignment is required for IR-MAD. Ensure both "
            f"images share the same CRS, pixel size, and spatial extent "
            f"before running the normalization.\n")

    rasterBands1 = [inDataset1.GetRasterBand(b) for b in band_pos]
    rasterBands2 = [inDataset2.GetRasterBand(b) for b in band_pos]
    ref_sentinels = rio.band_nodata(inDataset1, nodata_ref, band_pos)
    tgt_sentinels = rio.band_nodata(inDataset2, nodata_tgt, band_pos)

    # Sanity-check: any band that is entirely zero would make the algorithm
    # degenerate (singular covariance). Bail out early with a clear message.
    for k, rb in enumerate(rasterBands1):
        if not _has_nonzero(rb, x0, y0, cols, rows, block_rows, feedback):
            _error(f"\nERROR: band {band_pos[k]} of '{basename1}' has only "
                   f"zeros — please check it.\n")
    for k, rb in enumerate(rasterBands2):
        if not _has_nonzero(rb, x2, y2, cols, rows, block_rows, feedback):
            _error(f"\nERROR: band {band_pos[k]} of '{basename2}' has only "
                   f"zeros — please check it.\n")

    cpm = auxil.Cpm(2 * bands)
    oldrho = np.zeros(bands)
    rhos = np.zeros((max_iters, bands))
    # Counted separately from current_iter, which the fallback handler jumps to
    # max_iters; only rows actually filled describe the convergence history.
    completed = 0
    termination = 'iteration limit'
    selected_iteration = None
    final_delta = None
    results = []
    sigMADs = means1 = means2 = A = B = None
    minimums = np.full(2, np.inf)
    maximums = np.full(2, -np.inf)

    delta_thres = 1.0 - conv_threshold
    delta_text = round(delta_thres, 5)
    prefix = f'{ref_text} -> ' if ref_text else ''
    _info(f'Stop condition: delta < {delta_text} or {max_iters} iterations')
    previous_time = time.monotonic()

    current_iter = 0
    while current_iter < max_iters:
        if _canceled():
            return

        try:
            # ---- pass 1: accumulate weighted covariance over the full image
            for ry, nr in _iter_row_blocks(rows, block_rows):
                rio.check_cancel(feedback)
                tile_ref = _read_block(rasterBands1, x0, y0 + ry, cols, nr)
                tile_tgt = _read_block(rasterBands2, x2, y2 + ry, cols, nr)
                tile = np.concatenate((tile_ref, tile_tgt), axis=1)

                # Exclude rows where any image has a fully-zero pixel
                # (treated as no-data) — preserves the original behaviour —
                # plus any row whose bands contain the declared nodata value.
                nz_ref = tile_ref.any(axis=1)
                nz_tgt = tile_tgt.any(axis=1)
                keep = (nz_ref & nz_tgt & _valid_rows(tile_ref, ref_sentinels)
                        & _valid_rows(tile_tgt, tgt_sentinels))
                tile = tile[keep]

                if bands == 1 and current_iter == 0 and len(tile):
                    minimums = np.minimum(minimums, tile.min(axis=0))
                    maximums = np.maximum(maximums, tile.max(axis=0))

                if current_iter > 0:
                    # MAD variates and chi-square statistic for weighting
                    mads = ((tile[:, 0:bands] - means1[0]) @ A
                            - (tile[:, bands:] - means2[0]) @ B)
                    chisqr = np.sum((mads / sigMADs[0]) ** 2, axis=1)
                    # chi2.sf == 1 - chi2.cdf, but stable in the upper tail
                    wts = stats.chi2.sf(chisqr, bands)
                    cpm.update(tile, wts)
                else:
                    cpm.update(tile)

            # ---- canonical-correlation step
            if bands == 1 and current_iter == 0 and np.any(minimums == maximums):
                side = 'reference' if minimums[0] == maximums[0] else 'target'
                _error(f'The single-band {side} has no variation among valid overlap pixels; '
                       'IR-MAD requires nonzero variance in both images.')
            S = cpm.covariance()
            means = cpm.means()
            cpm.reset()

            s11 = S[0:bands, 0:bands]
            s22 = S[bands:, bands:]
            s12 = S[0:bands, bands:]
            s21 = s12.T  # S is symmetric

            # Solve the two coupled generalized eigenproblems
            #   s12 s22^-1 s21  a = mu^2  s11  a
            #   s21 s11^-1 s12  b = mu^2  s22  b
            if bands > 1:
                # scipy.linalg.solve is more stable than forming inv() explicitly
                from scipy.linalg import solve
                c1 = s12 @ solve(s22, s21, assume_a='pos')
                c2 = s21 @ solve(s11, s12, assume_a='pos')
                mu2a, A = auxil.geneiv(c1, s11)
                mu2b, B = auxil.geneiv(c2, s22)
                idx_a = np.argsort(mu2a)
                idx_b = np.argsort(mu2b)
                A = A[:, idx_a]
                B = B[:, idx_b]
                mu2 = mu2b[idx_b]
            else:
                if not (np.isfinite(s11[0, 0]) and np.isfinite(s22[0, 0])
                        and s11[0, 0] > 0 and s22[0, 0] > 0):
                    _error('Single-band IR-MAD requires finite, positive variance in both images.')
                mu2 = np.ravel((s12 * s21 / s22) / s11)
                A = np.array([[1.0 / np.sqrt(s11[0, 0])]])
                B = np.array([[1.0 / np.sqrt(s22[0, 0])]])

            # Clamp to [0, 1] before sqrt — round-off can push mu^2 slightly
            # negative or slightly above 1, which would yield NaN.
            mu2 = np.clip(mu2, 0.0, 1.0)
            rho = np.sqrt(mu2)
            # An exactly affine-related channel is a valid invariant. Bound
            # its variance at numerical precision instead of dividing by zero
            # or rejecting the other channels. Ordinary variates are unchanged.
            sigma = np.sqrt(np.maximum(2.0 * (1.0 - rho), np.finfo(float).eps))
            delta = float(np.max(np.abs(rho - oldrho)))

            oldrho = rho

            # Tile sigma and means to (1, ...) — broadcast over (n_pixels, bands)
            sigMADs = sigma[None, :]
            means1 = means[None, 0:bands]
            means2 = means[None, bands:]

            # Sign-fix: ensure each canonical variate has a positive sum of
            # correlations with the X channels (otherwise eigenvectors can
            # flip sign between iterations, breaking the stopping criterion).
            D = 1.0 / np.sqrt(np.diag(s11))            # vector form of diag(D)
            sgn_a = np.sign(np.sum(D[:, None] * s11 @ A, axis=0))
            sgn_a[sgn_a == 0] = 1.0
            A = A * sgn_a
            sgn_cov = np.sign(np.diag(A.T @ s12 @ B))
            sgn_cov[sgn_cov == 0] = 1.0
            B = B * sgn_cov

            current_iter += 1
            now = time.monotonic()
            elapsed = now - previous_time
            previous_time = now
            time_text = f'{elapsed:.2f}s' if elapsed < 60 else f'{elapsed / 60:.2f}min'
            _info(f'{prefix}iteration {current_iter}/{max_iters}: '
                  f'delta={delta:.5f}, time={time_text}')
            results.append((delta, {"iter": current_iter, "A": A, "B": B,
                                    "means1": means1, "means2": means2,
                                    "sigMADs": sigMADs, "rho": rho}))
            rhos[completed, :] = rho
            completed += 1
            selected_iteration = current_iter
            final_delta = delta if completed > 1 else None

            # Convergence check: stop when the maximum change in canonical
            # correlations falls below the threshold derived from conv_threshold.
            # Skip on the first iteration because oldrho starts at zero,
            # making delta a magnitude estimate rather than a convergence measure.
            if current_iter > 1 and delta < delta_thres:
                termination = 'converged'
                _info(f'{prefix}converged after {current_iter} iterations '
                      f'(delta={delta:.5f} < {delta_text})')
                break

            if feedback is not None:
                # Report progress in the 10–90% range that arrnorm.py
                # allocates for the IR-MAD step (0→10% = clipper, 90→100% = radcal+mask).
                feedback.setProgress(10 + int(80 * current_iter / max_iters))

        except (rio.Cancelled, _FatalInputError):
            raise  # deliberate fatal error (degenerate input) — do not swallow

        except (np.linalg.LinAlgError, FloatingPointError, ValueError) as err:
            termination = 'numerical fallback'
            _info(f'\n WARNING: iteration {current_iter + 1}/{max_iters} failed: {err}\n'
                  f' Falling back to the best result computed so far; verify the input bands.\n')
            current_iter = max_iters  # exit the while-loop

        if current_iter == max_iters:
            # Guard: if every iteration failed, results is empty
            if not results:
                _error(
                    f"\n ERROR: All {max_iters} iteration(s) failed without producing "
                    f"any valid result.\n"
                    f" Common causes:\n"
                    f"  - Reference and target have different pixel dimensions or "
                    f"extents after clipping\n"
                    f"  - Mismatched coordinate reference systems\n"
                    f"  - One or more bands contain only zeros or nodata\n"
                    f" Check the warnings above for the specific error that occurred.\n")

            # Pick the iteration with the smallest delta — the run with the
            # most-converged canonical correlations.
            best = min(results, key=itemgetter(0))
            selected_iteration = best[1]['iter']
            _info(f'\nBest delta: {best[0]:.5f} (iteration {best[1]["iter"]}); '
                  "using this iteration's parameters for the final result.")
            delta = best[0]
            A = best[1]["A"]
            B = best[1]["B"]
            means1 = best[1]["means1"]
            means2 = best[1]["means2"]
            sigMADs = best[1]["sigMADs"]
            rho = best[1]["rho"]
            del results

    correlations = ', '.join(f'{value:.5f}' for value in rho)
    _info(f'\nFinal canonical correlations: {correlations}')

    # ---- write MAD variates + chi-square band to disk
    rio.check_cancel(feedback)
    outDataset = stack.enter_context(rio.create_raster(outfn, cols, rows, bands + 1, GDT_Float32))
    projection = inDataset1.GetProjection()
    geotransform = inDataset1.GetGeoTransform()
    if geotransform is not None:
        gt = list(geotransform)
        gt[0] = gt[0] + x0 * gt[1] + y0 * gt[2]
        gt[3] = gt[3] + x0 * gt[4] + y0 * gt[5]
        rio.checked(outDataset.SetGeoTransform(tuple(gt)), 'Set MAD geotransform')
    if projection is not None:
        rio.checked(outDataset.SetProjection(projection), 'Set MAD projection')
    outBands = [outDataset.GetRasterBand(k + 1) for k in range(bands + 1)]

    for ry, nr in _iter_row_blocks(rows, block_rows):
        rio.check_cancel(feedback)
        tile_ref = _read_block(rasterBands1, x0, y0 + ry, cols, nr)
        tile_tgt = _read_block(rasterBands2, x2, y2 + ry, cols, nr)
        keep = (_valid_rows(tile_ref, ref_sentinels) & _valid_rows(tile_tgt, tgt_sentinels)
                & tile_ref.any(axis=1) & tile_tgt.any(axis=1))
        # Sanitize only after validity is known; excluded pixels receive inf
        # chi-square so they cannot re-enter Radcal as high-probability samples.
        tile_ref[~keep] = 0
        tile_tgt[~keep] = 0
        mads = (tile_ref - means1[0]) @ A - (tile_tgt - means2[0]) @ B
        chisqr = np.sum((mads / sigMADs[0]) ** 2, axis=1)
        chisqr[~keep] = np.inf
        for k in range(bands):
            rio.write_band(outBands[k], mads[:, k].reshape(nr, cols), 0, ry)
        rio.write_band(outBands[bands], chisqr.reshape(nr, cols), 0, ry)
    for outBand in outBands:
        rio.checked(outBand.FlushCache(), 'Flush MAD band')
    outDataset = None
    inDataset1 = None
    inDataset2 = None

    _info(f'MAD variates and chi-square computed ({bands} bands)')
    elapsed = time.time() - start
    time_text = f'{elapsed:.2f}s' if elapsed < 60 else f'{elapsed / 60:.2f}min'
    _info(f'elapsed time: {time_text}')

    if convergence is not None:
        # Hand the history to the caller instead of opening a blocking pyplot
        # window: the calibration report draws it, and a Processing worker must
        # never create widgets or touch another plugin's matplotlib state.
        convergence.extend(rhos[:completed].tolist())
    if convergence_info is not None:
        convergence_info.update(termination=termination, iterations=completed,
                                selected_iteration=selected_iteration, final_delta=final_delta,
                                delta_threshold=delta_thres)

    return outfn
