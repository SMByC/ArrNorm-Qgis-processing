#!/usr/bin/env python3
# ******************************************************************************
#  Name:     auxil.py
#  Purpose:  Math primitives used by the IR-MAD / RadCal / Register pipeline.
#
#  Primitives used by normalization and registration:
#     Cpm           -- weighted streaming mean / covariance accumulator
#     geneiv        -- symmetric generalized eigenproblem  A x = lambda B x
#     orthoregress  -- orthogonal (total-least-squares) regression
#     OrthogonalFit -- bounded-memory regression moments
#     similarity    -- log-polar Fourier image-image similarity transform
#
#  Original auxiliaries: M. Canty 2012 (DWT, ATWT, PCA, MNF, kernels, PNG
#  output, ENVI header parsing, supervised classifiers, congrid, ...) were
#  not used by the IR-MAD pipeline and have been removed.
#
#  License: GPLv2+
# ******************************************************************************

import math

import numpy as np
import scipy.linalg
import scipy.ndimage as ndii
from numpy.fft import fft2, fftshift, ifft2

# An older ArrNormPlugin.unload() is still in memory during a live QGIS plugin
# upgrade. Its Windows path imports `lib` from this newly replaced module before
# unregistering the provider. No native library is loaded anymore; None lets
# that old code skip FreeLibrary via its existing exception handler.
lib = None


# -----------------
# provisional means
# -----------------

class Cpm(object):
    """Weighted running mean / cross-product accumulator.

    Pure-numpy replacement for the original ctypes 'provmeans' shared
    library. Tracks the weighted mean and the weighted sum of squared/
    cross-product deviations (SSCP) for an N-variate stream of
    observations, using West's algorithm (numerically stable single
    pass).

    With incoming batch x_i and weights w_i:
        SW_new   = SW + sum(w_i)
        delta_i  = x_i - mean_old
        mean_new = mean_old + sum(w_i * delta_i) / SW_new
        cov_new  = cov_old + sum(w_i * delta_i * (x_i - mean_new).T)
    covariance() returns cov_new / (SW - 1).
    """

    def __init__(self, N):
        self.N = N
        self.reset()

    def reset(self):
        self.mn = np.zeros(self.N)
        self.cov = np.zeros((self.N, self.N))
        self.sw = 1e-7  # tiny seed avoids divide-by-zero on covariance()

    def update(self, Xs, Ws=None):
        Xs = np.asarray(Xs, dtype=np.float64)
        if Xs.ndim == 1:
            Xs = Xs.reshape(1, -1)
        n = Xs.shape[0]
        if n == 0:
            return
        if Ws is None:
            Ws = np.ones(n, dtype=np.float64)
        else:
            Ws = np.asarray(Ws, dtype=np.float64)

        sw_new = self.sw + Ws.sum()
        delta = Xs - self.mn                              # (n, N)
        weighted_delta = Ws[:, None] * delta              # (n, N)
        self.mn = self.mn + weighted_delta.sum(axis=0) / sw_new
        delta2 = Xs - self.mn                             # (n, N) with NEW mean
        self.cov = self.cov + weighted_delta.T @ delta2   # (N, N)
        self.sw = sw_new

    def covariance(self):
        if not np.isfinite(self.sw) or self.sw <= 1:
            raise ValueError('Insufficient effective pixel weight for covariance.')
        c = self.cov / (self.sw - 1.0)
        return 0.5 * (c + c.T)  # symmetrize tiny asymmetric drift

    def means(self):
        return self.mn


# ---------------------------------
# symmetric generalized eigenproblem
# ---------------------------------

def geneiv(A, B):
    """Solve the symmetric generalized eigenproblem  A x = lambda B x.

    Returns (eigenvalues, eigenvectors-in-columns), eigenvalues sorted
    ascending. Uses scipy.linalg.eigh which dispatches to LAPACK's
    DSYGVD: it handles the Cholesky reduction internally with a single
    call and is more numerically stable than building inv(chol(B))
    explicitly.
    """
    A_s = 0.5 * (A + A.T)
    B_s = 0.5 * (B + B.T)
    return scipy.linalg.eigh(A_s, B_s)


# ---------------------
# orthogonal regression
# ---------------------

def orthoregress(x, y):
    """Total-least-squares (orthogonal) regression of y on x.

    Returns (slope, intercept, Pearson R). Uses the closed-form
    solution for the 2-variable case (major axis of the (x, y)
    covariance ellipse):
        b = (Syy - Sxx + sqrt((Syy - Sxx)^2 + 4*Sxy^2)) / (2*Sxy)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or x.shape != y.shape or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError('Regression requires at least two finite paired samples.')
    xm = x.mean()
    ym = y.mean()
    dx = x - xm
    dy = y - ym
    n = x.size - 1
    sxx = np.dot(dx, dx) / n
    syy = np.dot(dy, dy) / n
    sxy = np.dot(dx, dy) / n
    denom = math.sqrt(sxx * syy)
    R = sxy / denom if denom > 0.0 else 0.0
    if sxy == 0.0:
        return [0.0, ym, R]
    b = (syy - sxx + math.sqrt((syy - sxx) ** 2 + 4.0 * sxy * sxy)) / (2.0 * sxy)
    return [b, ym - b * xm, R]


class OrthogonalFit:
    """Streaming centered moments for an exact TLS fit, independent of block size.

Chan's merge formula avoids subtracting large uncentered sums. Memory is
constant in the number of pixels; only the current input block is retained.
"""
    def __init__(self):
        self.count = 0
        self.mean = np.zeros(2)
        self.cross = np.zeros((2, 2))

    def update(self, x, y):
        if len(x) == 0:
            return
        values = np.column_stack((x, y)).astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError('Regression samples must be finite.')
        count = len(values)
        mean = values.mean(axis=0)
        centered = values - mean
        delta = mean - self.mean
        total = self.count + count
        self.cross += centered.T @ centered + np.outer(delta, delta) * (self.count * count / total)
        self.mean += delta * (count / total)
        self.count = total

    def coefficients(self):
        if self.count < 2:
            raise ValueError('Regression requires at least two finite paired samples.')
        sxx, sxy, syy = self.cross[0, 0], self.cross[0, 1], self.cross[1, 1]
        if sxx <= 0 or not np.isfinite(self.cross).all():
            raise ValueError('No target variance remains in the regression samples.')
        if sxy == 0:
            if syy > sxx:
                raise ValueError('A vertical orthogonal regression cannot calibrate the target.')
            return 0.0, float(self.mean[1]), 0.0
        difference = syy - sxx
        length = math.hypot(difference, 2 * sxy)
        slope = ((difference + length) / (2 * sxy) if difference >= 0
                 else 2 * sxy / (length - difference))
        return (slope, self.mean[1] - slope * self.mean[0],
                float(np.clip(sxy / math.sqrt(sxx * syy), -1, 1)))

    def statistics(self):
        """Affine-model agreement over every accumulated pair, before storage conversion.

        The accumulated moments already determine these in closed form, so the
        whole population is described without a second pass or a sample:

            after  RMSE^2 = (Syy - 2 b Sxy + b^2 Sxx) / n
            before RMSE^2 = (Sxx - 2 Sxy + Syy) / n + (ym - xm)^2
            variance ratio (calibrated / reference) = b^2 Sxx / Syy

        The mean residual after calibration is omitted deliberately: the fitted
        line passes through the centroid, so it is exactly zero here and cannot
        measure accuracy. Use a hold-out subset for that. Near cancellation,
        report an estimated round-off bound rather than a spurious zero. This is
        a numerical-resolution estimate, not a statistical confidence interval.
        """
        slope, intercept, correlation = self.coefficients()
        sxx, sxy, syy = self.cross[0, 0], self.cross[0, 1], self.cross[1, 1]
        difference = float(self.mean[1] - self.mean[0])
        before = (sxx - 2 * sxy + syy) / self.count + difference ** 2
        after = (syy - 2 * slope * sxy + slope ** 2 * sxx) / self.count
        # Allow for both accumulated moment error and subtraction of nearly equal
        # terms. The deliberately conservative scale grows with population size;
        # it does not claim correctly rounded residuals below moment resolution.
        roundoff = 64 * np.finfo(float).eps * self.count
        before_scale = (abs(sxx) + 2 * abs(sxy) + abs(syy)) / self.count + difference ** 2
        after_scale = (abs(syy) + 2 * abs(slope * sxy) + slope ** 2 * abs(sxx)) / self.count

        def rmse(value, scale, offset_scale):
            # Large offsets can also lose precision in accumulated centroids and
            # evaluation of intercept + slope*x, even with modest centered spread.
            resolution = (math.sqrt(roundoff * scale) + roundoff * offset_scale) ** 2
            if np.isfinite(value) and abs(value) <= resolution:
                return float('nan'), math.sqrt(max(float(value), 0.) + resolution)
            return (math.sqrt(value) if value > 0 else float('nan')), float('nan')

        before_rmse, before_bound = rmse(before, before_scale, np.abs(self.mean).sum())
        after_rmse, after_bound = rmse(
            after, after_scale, abs(self.mean[1]) + abs(slope * self.mean[0]) + abs(intercept))
        return {'count': int(self.count), 'slope': slope, 'intercept': intercept,
                'correlation': correlation, 'mean_difference': difference,
                'rmse_before': before_rmse, 'rmse_after': after_rmse,
                'rmse_before_bound': before_bound, 'rmse_after_bound': after_bound,
                'variance_ratio': (float(slope ** 2 * sxx / syy) if syy > 0
                                   else float('nan'))}


# -----------------------------
# image-image similarity (Fourier log-polar)
# -----------------------------

def similarity(bn0, bn1):
    """Estimate (scale, angle, [t_row, t_col]) registering bn1 -> bn0.

    Uses the Reddy-Chatterji log-polar / Fourier cross-correlation
    method: scale and rotation are recovered from the magnitude
    spectrum in log-polar coordinates, then translation is recovered
    from phase correlation of the rectified images.

    Adapted from M. Canty 2012 / Christoph Gohlke's Imreg.py.
    """

    def highpass(shape):
        """High-pass cosine filter to suppress DC before log-polar mapping."""
        x = np.outer(
            np.cos(np.linspace(-math.pi / 2., math.pi / 2., shape[0])),
            np.cos(np.linspace(-math.pi / 2., math.pi / 2., shape[1])))
        return (1.0 - x) * (2.0 - x)

    def logpolar(image, angles=None, radii=None):
        """Map `image` into log-polar coordinates, return (image, log_base)."""
        shape = image.shape
        center = shape[0] / 2, shape[1] / 2
        if angles is None:
            angles = shape[0]
            if radii is None:
                radii = shape[1]
        theta = np.empty((angles, radii), dtype=np.float64)
        theta.T[:] = -np.linspace(0, np.pi, angles, endpoint=False)
        d = np.hypot(shape[0] - center[0], shape[1] - center[1])
        log_base = 10.0 ** (math.log10(d) / radii)
        radius = np.empty_like(theta)
        radius[:] = np.power(log_base, np.arange(radii, dtype=np.float64)) - 1.0
        x = radius * np.sin(theta) + center[0]
        y = radius * np.cos(theta) + center[1]
        output = np.empty_like(x)
        ndii.map_coordinates(image, [x, y], output=output)
        return output, log_base

    lines0, samples0 = bn0.shape
    bn1 = bn1[0:lines0, 0:samples0]  # crop to reference shape

    # ---- scale + angle from log-polar of magnitude spectra
    f0 = fftshift(abs(fft2(bn0)))
    f1 = fftshift(abs(fft2(bn1)))
    h = highpass(f0.shape)
    f0 *= h
    f1 *= h
    del h
    f0, log_base = logpolar(f0)
    f1, log_base = logpolar(f1)
    f0 = fft2(f0)
    f1 = fft2(f1)
    r0 = abs(f0) * abs(f1)
    ir = abs(ifft2((f0 * f1.conjugate()) / r0))
    i0, i1 = np.unravel_index(np.argmax(ir), ir.shape)
    angle = 180.0 * i0 / ir.shape[0]
    scale = log_base ** i1
    if scale > 1.8:
        # try the inverse direction
        ir = abs(ifft2((f1 * f0.conjugate()) / r0))
        i0, i1 = np.unravel_index(np.argmax(ir), ir.shape)
        angle = -180.0 * i0 / ir.shape[0]
        scale = 1.0 / (log_base ** i1)
        if scale > 1.8:
            raise ValueError("Images are not compatible. Scale change > 1.8")
    if angle < -90.0:
        angle += 180.0
    elif angle > 90.0:
        angle -= 180.0

    # ---- translation from phase-correlation of the rectified image
    bn2 = ndii.zoom(bn1, 1.0 / scale)
    bn2 = ndii.rotate(bn2, angle)
    if bn2.shape < bn0.shape:
        t = np.zeros_like(bn0)
        t[:bn2.shape[0], :bn2.shape[1]] = bn2
        bn2 = t
    elif bn2.shape > bn0.shape:
        bn2 = bn2[:bn0.shape[0], :bn0.shape[1]]
    f0 = fft2(bn0)
    f1 = fft2(bn2)
    ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    t0, t1 = np.unravel_index(np.argmax(ir), ir.shape)
    if t0 > f0.shape[0] // 2:
        t0 -= f0.shape[0]
    if t1 > f0.shape[1] // 2:
        t1 -= f0.shape[1]
    return (scale, angle, [t0, t1])
