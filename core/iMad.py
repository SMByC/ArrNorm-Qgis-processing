#!/usr/bin/env python
# ******************************************************************************
#  Name:     iMad.py
#  Purpose:  Perfrom IR-MAD change detection on bitemporal, multispectral
#            imagery 
#  Usage:             
#    python iMad.py -h
#
#  Copyright (c) 2013, Mort Canty
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

import os
from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly, GDT_Float32
from qgis.core import QgsProcessingException
from scipy import linalg, stats

from ArrNorm.core.auxil import auxil

usage = '''
Usage:
-----------------------------------------------------
python %s [-h] [-n] [-i max iterations] [-p bandPositions]
[-d spatialDimensions] filename1 filename2
-----------------------------------------------------
bandPositions and spatialDimensions are lists,
e.g., -p [1,2,3] -d [0,0,400,400]
-n stops any graphics output
-----------------------------------------------------
The output MAD variate file is has the same format
as filename1 and is named

      path/MAD(filebasename1-filebasename2).ext1

where filename1 = path/filebasename1.ext1
      filename2 = path/filebasename2.ext2

For ENVI files, ext1 or ext2 is the empty string.
-----------------------------------------------------'''


def main(img_ref, img_target, max_iters=30, band_pos=None, dims=None, graphics=False, feedback=None):

    gdal.AllRegister()

    path = os.path.dirname(os.path.abspath(img_ref))
    basename1 = os.path.basename(img_ref)
    root1, ext1 = os.path.splitext(basename1)
    basename2 = os.path.basename(img_target)
    root2, ext2 = os.path.splitext(basename2)
    outfn = os.path.join(path, 'MAD({0}&{1}){2}'.format(root1, basename2, ext1))
    inDataset1 = gdal.Open(img_ref, GA_ReadOnly)
    inDataset2 = gdal.Open(img_target, GA_ReadOnly)
    try:
        cols = inDataset1.RasterXSize
        rows = inDataset1.RasterYSize
        bands = inDataset1.RasterCount
        cols2 = inDataset2.RasterXSize
        rows2 = inDataset2.RasterYSize
        bands2 = inDataset2.RasterCount
    except Exception as err:
        raise QgsProcessingException('Error: {}  --Images could not be read.'.format(err))
    if bands != bands2:
        raise QgsProcessingException("\nERROR: The number of bands does not match between reference and target image\n")
    if band_pos is None:
        band_pos = list(range(1, bands + 1))
    else:
        bands = len(band_pos)
    if dims is None:
        x0 = 0
        y0 = 0
    else:
        x0, y0, cols, rows = dims
    # if second image is warped, assume it has same dimensions as dims
    if root2.find('_warp') != -1:
        x2 = 0
        y2 = 0
    else:
        x2 = x0
        y2 = y0

    feedback.pushInfo('\n------------IRMAD -------------')
    # iteration of MAD
    cpm = auxil.Cpm(2 * bands)
    delta = 1.0
    oldrho = np.zeros(bands)
    current_iter = 0
    tile = np.zeros((cols, 2 * bands))
    sigMADs = 0
    means1 = 0
    means2 = 0
    A = 0
    B = 0
    rasterBands1 = []
    rasterBands2 = []
    rhos = np.zeros((max_iters, bands))
    results = []
    for band in band_pos:
        rasterBands1.append(inDataset1.GetRasterBand(band))
        rasterBands2.append(inDataset2.GetRasterBand(band))

    # check if the band data has only zeros
    for band in range(bands):
        # check the reference image
        if not rasterBands1[band].ReadAsArray().any():
            raise QgsProcessingException("\nERROR: the band No. {0} for the file '{1}'\n"
                  "has only zeros! please check it.\n".format(band+1, basename1))
        # check the target image
        if not rasterBands2[band].ReadAsArray().any():
            raise QgsProcessingException("\nERROR: the band No. {0} for the file '{1}'\n"
                  "has only zeros! please check it.\n".format(band+1, basename2))

    feedback.pushInfo('\nStop condition: max iteration {iter} with auto selection\n'
                      'the best delta for the final result:'.format(iter=max_iters))

    feedback.pushInfo(' {img_target} -> iteration: 0, delta: 1.0'.format(img_target=os.path.basename(img_target)))

    while current_iter < max_iters:
        if feedback.isCanceled():
            return
        try:
            # spectral tiling for statistics
            for row in range(rows):
                for k in range(bands):
                    tile[:, k] = rasterBands1[k].ReadAsArray(x0, y0 + row, cols, 1)
                    tile[:, bands + k] = rasterBands2[k].ReadAsArray(x2, y2 + row, cols, 1)
                # eliminate no-data pixels
                tile = np.nan_to_num(tile)
                tst1 = np.sum(tile[:, 0:bands], axis=1)
                tst2 = np.sum(tile[:, bands::], axis=1)
                idx1 = set(np.where((tst1 != 0))[0])
                idx2 = set(np.where((tst2 != 0))[0])
                idx = list(idx1.intersection(idx2))
                if current_iter > 0:
                    mads = np.asarray((tile[:, 0:bands] - means1) * A - (tile[:, bands::] - means2) * B)
                    chisqr = np.sum(np.square(mads / sigMADs), axis=1)
                    wts = 1 - stats.chi2.cdf(chisqr, [bands])
                    cpm.update(tile[idx, :], wts[idx])
                else:
                    cpm.update(tile[idx, :])

            # weighted covariance matrices and means
            S = cpm.covariance()
            means = cpm.means()
            # reset prov means object
            cpm.__init__(2 * bands)
            s11 = S[0:bands, 0:bands]
            s22 = S[bands:, bands:]
            s12 = S[0:bands, bands:]
            s21 = S[bands:, 0:bands]
            c1 = s12 * linalg.inv(s22) * s21
            b1 = s11
            c2 = s21 * linalg.inv(s11) * s12
            b2 = s22
            # solution of generalized eigenproblems
            if bands > 1:
                mu2a, A = auxil.geneiv(c1, b1)
                mu2b, B = auxil.geneiv(c2, b2)
                # sort a
                idx = np.argsort(mu2a)
                A = A[:, idx]
                # sort b
                idx = np.argsort(mu2b)
                B = B[:, idx]
                mu2 = mu2b[idx]
            else:
                mu2 = c1 / b1
                A = 1 / np.sqrt(b1)
                B = 1 / np.sqrt(b2)

            # canonical correlations
            rho = np.sqrt(mu2)
            b2 = np.diag(B.T * B)
            sigma = np.sqrt(2 * (1 - rho))
            # stopping criterion
            delta = max(abs(rho - oldrho))
            if bands == 1:
                delta = delta.item()

            rhos[current_iter, :] = rho
            oldrho = rho
            # tile the sigmas and means
            sigMADs = np.tile(sigma, (cols, 1))
            means1 = np.tile(means[0:bands], (cols, 1))
            means2 = np.tile(means[bands::], (cols, 1))
            # ensure sum of positive correlations between X and U is positive
            D = np.diag(1 / np.sqrt(np.diag(s11)))
            s = np.ravel(np.sum(D * s11 * A, axis=0))
            A = A * np.diag(s / np.abs(s))
            # ensure positive correlation between each pair of canonical variates
            cov = np.diag(A.T * s12 * B)
            B = B * np.diag(cov / np.abs(cov))
            current_iter += 1

            feedback.pushInfo(' {img_target} -> iteration: {iter}, delta: {delta}'.format(
                img_target=os.path.basename(img_target), iter=current_iter, delta=round(delta, 6)))
            feedback.setProgress(feedback.progress() + 2)

            # save parameters
            results.append((delta, {"iter": current_iter, "A": A, "B": B, "means1": means1, "means2": means2,
                                    "sigMADs": sigMADs, "rho": rho}))
        except Exception as err:
            feedback.pushInfo(
                "\n WARNING: Occurred a exception value error for the last iteration No. {iter},\n"
                " then the ArrNorm will be use the best result at the moment calculated, you\n"
                " should check the result and all bands in input file if everything is correct.\n\n"
                " Error: {err}".format(iter=current_iter, err=err))
            # ending the iteration
            current_iter = max_iters

        if feedback.isCanceled():
            return

        if current_iter == max_iters:  # end iteration
            # select the result with the best delta
            best_results = sorted(results, key=itemgetter(0))[0]
            feedback.pushInfo(
                "\n The best delta for all iterations is {0}, iter num: {1},\n"
                " making the final result normalization with this parameters.".
                format(round(best_results[0], 5), best_results[1]["iter"]))

            # set the parameter for the best delta
            delta = best_results[0]
            A = best_results[1]["A"]
            B = best_results[1]["B"]
            means1 = best_results[1]["means1"]
            means2 = best_results[1]["means2"]
            sigMADs = best_results[1]["sigMADs"]
            rho = best_results[1]["rho"]

            del results

    feedback.pushInfo('\nRHO: {}'.format(rho))
    # write results to disk
    driver = inDataset1.GetDriver()
    outDataset = driver.Create(outfn, cols, rows, bands + 1, GDT_Float32)
    projection = inDataset1.GetProjection()
    geotransform = inDataset1.GetGeoTransform()
    if geotransform is not None:
        gt = list(geotransform)
        gt[0] = gt[0] + x0 * gt[1]
        gt[3] = gt[3] + y0 * gt[5]
        outDataset.SetGeoTransform(tuple(gt))
    if projection is not None:
        outDataset.SetProjection(projection)
    outBands = []
    for k in range(bands + 1):
        outBands.append(outDataset.GetRasterBand(k + 1))
    for row in range(rows):
        for k in range(bands):
            tile[:, k] = rasterBands1[k].ReadAsArray(x0, y0 + row, cols, 1)
            tile[:, bands + k] = rasterBands2[k].ReadAsArray(x2, y2 + row, cols, 1)
        mads = np.asarray((tile[:, 0:bands] - means1) * A - (tile[:, bands::] - means2) * B)
        chisqr = np.sum(np.square(mads / sigMADs), axis=1)
        for k in range(bands):
            outBands[k].WriteArray(np.reshape(mads[:, k], (1, cols)), 0, row)
        outBands[bands].WriteArray(np.reshape(chisqr, (1, cols)), 0, row)
    for outBand in outBands:
        outBand.FlushCache()
    outDataset = None
    inDataset1 = None
    inDataset2 = None
    x = np.array(list(range(current_iter - 1)))
    if graphics:
        plt.plot(x, rhos[0:current_iter - 1, :])
        plt.title('Canonical correlations')
        plt.show()

    return outfn
