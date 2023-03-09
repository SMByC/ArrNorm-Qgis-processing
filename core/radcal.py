#!/usr/bin/env python
# ******************************************************************************
#  Name:     radcal.py
#  Purpose:  Automatic radiometric normalization
#  Usage:             
#       python radcal.py  [-p 'bandPositions' -d 'spatialDimensions' -t NoChangeProbThresh] imadFile [FullSceneFile] 
#
#  Copyright (c) 2011, Mort Canty
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
import getopt
import time
import matplotlib.pyplot as plt
from numpy import *
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly, GDT_UInt16
from scipy import stats

from ArrNorm.core.auxil.auxil import orthoregress

usage = '''
Usage:
--------------------------------------------------------
python %s  [-p "bandPositions"] [-d "spatialDimensions"]
[-t no-change prob threshold] imadFile [fullSceneFile]'
--------------------------------------------------------
bandPositions and spatialDimensions are quoted lists,
e.g., -p [4,5,6] -d [0,0,400,400]
-n stops graphics output

SpatialDimensions MUST match those of imadFile
spectral dimension of fullSceneFile, if present,
MUST match those of target and reference images
--------------------------------------------------------
imadFile is of form path/MAD(filename1-filename2).ext and
the output file is named

            path/filename2_norm.ext.

That is, it is assumed that filename1 is reference and
filename2 is target and the output retains the format
of the imadFile. A similar convention is used to
name the normalized full scene, if present:

         fullSceneFile_norm.ext

Note that, for ENVI format, ext is the empty string.
-------------------------------------------------------'''


def main(img_imad, output, ncpThresh=0.95, pos=None, dims=None, img_target=None, graphics=False, out_dtype=None):

    if img_target is not None:
        path = os.path.dirname(img_target)
        basename = os.path.basename(img_target)
        root, ext = os.path.splitext(basename)
        fsoutfn = path + '/' + root + '_norm_all' + ext

    path = os.path.dirname(img_imad)
    basename = os.path.basename(img_imad)
    root, ext = os.path.splitext(basename)
    b = root.find('(')
    err = root.find(')')
    referenceroot, targetbasename = root[b + 1:err].split('&')
    referencefn = path + '/' + referenceroot + ext
    targetfn = path + '/' + targetbasename
    outfn = output
    imadDataset = gdal.Open(img_imad, GA_ReadOnly)
    try:
        imadbands = imadDataset.RasterCount
        cols = imadDataset.RasterXSize
        rows = imadDataset.RasterYSize
    except Exception as err:
        print('Error: {}  --Image could not be read'.format(err))
        sys.exit(1)
    referenceDataset = gdal.Open(referencefn, GA_ReadOnly)
    targetDataset = gdal.Open(targetfn, GA_ReadOnly)
    if pos is None:
        pos = list(range(1, referenceDataset.RasterCount + 1))
    if dims is None:
        x0 = 0
        y0 = 0
    else:
        x0, y0, cols, rows = dims
    chisqr = imadDataset.GetRasterBand(imadbands).ReadAsArray(0, 0, cols, rows).ravel()
    ncp = 1 - stats.chi2.cdf(chisqr, [imadbands - 1])
    idx = where(ncp > ncpThresh)
    print(time.asctime())
    print('reference: ' + referencefn)
    print('target   : ' + targetfn)
    print('no-change probability threshold: ' + str(ncpThresh))
    print('no-change pixels: ' + str(len(idx[0])))
    start = time.time()
    driver = targetDataset.GetDriver()
    outDataset = driver.Create(outfn, cols, rows, len(pos), out_dtype)
    projection = imadDataset.GetProjection()
    geotransform = imadDataset.GetGeoTransform()
    if geotransform is not None:
        outDataset.SetGeoTransform(geotransform)
    if projection is not None:
        outDataset.SetProjection(projection)
    aa = []
    bb = []
    if graphics:
        plt.figure(1, (9, 6))
    j = 1
    bands = len(pos)
    for k in pos:
        x = referenceDataset.GetRasterBand(k).ReadAsArray(x0, y0, cols, rows).astype(float).ravel()
        y = targetDataset.GetRasterBand(k).ReadAsArray(x0, y0, cols, rows).astype(float).ravel()
        b, a, R = orthoregress(y[idx], x[idx])
        print('band: {}  slope: {} intercept: {}  correlation: {}'.format(k, b, a, R))
        my = max(y[idx])
        if (j < 7) and graphics:
            plt.subplot(2, 3, j)
            plt.plot(y[idx], x[idx], '.')
            plt.plot([0, my], [a, a + b * my])
            plt.title('Band {}'.format(k))
            if ((j < 4) and (bands < 4)) or j > 3:
                plt.xlabel('Target')
            if (j == 1) or (j == 4):
                plt.ylabel('Reference')
        aa.append(a)
        bb.append(b)
        outBand = outDataset.GetRasterBand(j)
        outBand.WriteArray(resize(a + b * y, (rows, cols)), 0, 0)
        outBand.FlushCache()
        j += 1
    if graphics:
        plt.show()
        plt.close()
    referenceDataset = None
    targetDataset = None
    outDataset = None
    print('result written to: ' + outfn)
    if img_target is not None:
        print('normalizing ' + img_target + '...')
        fsDataset = gdal.Open(img_target, GA_ReadOnly)
        try:
            cols = fsDataset.RasterXSize
            rows = fsDataset.RasterYSize
        except Exception as err:
            print('Error %s  -- Image could not be read in')
            sys.exit(1)
        driver = fsDataset.GetDriver()
        outDataset = driver.Create(fsoutfn, cols, rows, len(pos), out_dtype)
        projection = fsDataset.GetProjection()
        geotransform = fsDataset.GetGeoTransform()
        if geotransform is not None:
            outDataset.SetGeoTransform(geotransform)
        if projection is not None:
            outDataset.SetProjection(projection)
        j = 1
        for k in pos:
            inBand = fsDataset.GetRasterBand(k)
            outBand = outDataset.GetRasterBand(j)
            for i in range(rows):
                y = inBand.ReadAsArray(0, i, cols, 1)
                outBand.WriteArray(aa[j - 1] + bb[j - 1] * y, 0, i)
            outBand.FlushCache()
            j += 1
        outDataset = None
        fsDataset = None
        print('full result written to: ' + fsoutfn)
        return fsoutfn

    print('elapsed time: {}'.format(time.time() - start))
    return outfn


if __name__ == '__main__':

    options, args = getopt.getopt(sys.argv[1:], 'hnp:d:t:')
    pos = None
    dims = None
    ncpThresh = 0.95
    fsfn = None
    graphics = True
    for option, value in options:
        if option == '-h':
            print(usage)
            sys.exit()
        elif option == '-n':
            graphics = False
        elif option == '-p':
            pos = eval(value)
        elif option == '-d':
            dims = eval(value)
        elif option == '-t':
            ncpThresh = eval(value)
    if (len(args) != 1) and (len(args) != 2):
        print('Incorrect number of arguments')
        print(usage)
        sys.exit(1)
    imadfn = args[0]
    if len(args) == 2:
        fsfn = args[1]

    main(imadfn, ncpThresh, pos, dims, fsfn)
