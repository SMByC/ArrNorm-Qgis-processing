# -*- coding: utf-8 -*-
"""
/***************************************************************************
 ArrNorm
                          A QGIS plugin processing
 Automatic relative radiometric normalization
                              -------------------
        copyright            : (C) 2021 by Xavier Corredor Llano, SMByC
        email                : xavier.corredor.llano@gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
import os
import platform
import sys
import subprocess

from qgis.core import QgsProcessingException

# add project dir to pythonpath
project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from ArrNorm.core import iMad, radcal

header = '''
==============================================================

ArrNorm - Automatic Relative Radiometric Normalization

Some code base on: Dr. Mort Canty
                   https://github.com/mortcanty/CRCDocker

Copyright (c) 2016 SMBYC-IDEAM
Authors: Xavier Corredor Llano <xcorredorl@ideam.gov.co>
Sistema de Monitoreo de Bosques y Carbono - SMBYC
IDEAM, Colombia

==============================================================
'''


class Normalization:
    def __init__(self, img_ref, img_target, max_iters, prob_thres, neg_to_nodata, nodata_mask, output, feedback):
        self.img_ref = img_ref
        self.img_target = img_target
        self.max_iters = max_iters
        self.prob_thres = prob_thres
        self.nodata_mask = nodata_mask
        self.neg_to_nodata = neg_to_nodata
        self.output = output
        self.feedback = feedback

    def run(self):

        self.feedback.pushInfo("PROCESSING IMAGE: {target}".format(target=os.path.basename(self.img_target)))

        self.feedback.setProgress(0)
        self.clipper()
        if self.feedback.isCanceled():
            return

        self.feedback.setProgress(10)
        self.imad()
        if self.feedback.isCanceled():
            return

        self.feedback.setProgress(70)
        self.radcal()
        if self.feedback.isCanceled():
            return

        if self.neg_to_nodata:
            self.feedback.setProgress(80)
            self.no_negative_value(self.img_norm)

        if self.nodata_mask:
            self.feedback.setProgress(90)
            self.make_mask()
            self.apply_mask()

        self.clean()

        self.feedback.setProgress(100)
        self.feedback.pushInfo('\nDONE: {img_target} PROCESSED\n'
              '      image normalized saved in: {img_norm}\n'.format(
                img_target=os.path.basename(self.img_target),
                img_norm=os.path.basename(self.img_norm)))

    def clipper(self):
        """Clip the reference image with the target image"""
        from osgeo import gdal
        from osgeo.gdalconst import GA_ReadOnly

        gdal_file = gdal.Open(self.img_target, GA_ReadOnly)

        # pixel size
        pixel_size_x = abs(round(gdal_file.GetGeoTransform()[1], 3))
        pixel_size_y = abs(round(gdal_file.GetGeoTransform()[5], 3))
        if pixel_size_x != pixel_size_y:
            raise QgsProcessingException('\nError: target and reference image have different pixel sizes\n')

        # extent
        x_size = gdal_file.RasterXSize
        y_size = gdal_file.RasterYSize
        if x_size == y_size:
            self.feedback.pushInfo("\nImages with the same extent, not neccessary to clip\n")
            self.img_ref_clip = self.img_ref
            return

        self.feedback.pushInfo("\nClipping the ref image with target\n")

        geoTransform = gdal_file.GetGeoTransform()
        minx = geoTransform[0]
        maxy = geoTransform[3]
        maxx = minx + geoTransform[1] * gdal_file.RasterXSize
        miny = maxy + geoTransform[5] * gdal_file.RasterYSize
        gdal_file = None
        filename, ext = os.path.splitext(os.path.basename(self.img_ref))
        self.img_ref_clip = os.path.join(os.path.dirname(os.path.abspath(self.img_target)),
                                         filename + "_" + os.path.splitext(os.path.basename(self.img_target))[0]
                                         + "_clip" + ext)

        cmd = 'gdal_translate -projwin ' + ' '.join([str(x) for x in [minx, maxy, maxx, miny]]) + \
              ' -of GTiff "' + self.img_ref + '" "' + self.img_ref_clip + '"'
        cmd_out = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if cmd_out.returncode == 0:  # successfully
            self.feedback.pushInfo('Clipped ref image successfully: ' + os.path.basename(self.img_ref_clip))
        else:
            raise QgsProcessingException('\nError clipping reference image: ' + str(cmd_out.stderr.decode()))

    def imad(self):
        # ======================================
        # iMad process

        img_target = self.img_target

        self.feedback.pushInfo("\niMad process for:\n" +
              os.path.basename(self.img_ref) + " " + os.path.basename(self.img_target))
        self.img_imad = iMad.main(self.img_ref_clip, img_target, max_iters=self.max_iters, feedback=self.feedback)

    def radcal(self):
        # ======================================
        # Radcal process

        self.feedback.pushInfo("\nRadcal process for\n" +
              os.path.basename(self.img_ref) + " " + os.path.basename(self.img_target) + " " +
              " with iMad image: " + os.path.basename(self.img_imad))
        self.img_norm = radcal.main(self.img_imad, self.output, ncpThresh=self.prob_thres)

    def no_negative_value(self, image):
        # ======================================
        # Convert negative values to NoData to image normalized

        self.feedback.pushInfo('\nConverting negative values for\n' +
              os.path.basename(os.path.basename(image)))
        cmd = ['gdal_calc' if platform.system() == 'Windows' else 'gdal_calc.py', '-A', '"{}"'.format(image),
               '--overwrite', '--calc', '"A*(A>=0)"', '--allBands=A', '--NoDataValue=0', '--type=UInt16', '--quiet',
               '--outfile', '"{}"'.format(image)]
        cmd_out = subprocess.run(" ".join(cmd), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if cmd_out.returncode == 0:  # successfully
            self.feedback.pushInfo('Negative values converted successfully: ' + os.path.basename(image))
        else:
            raise QgsProcessingException('\nError converting values: ' + str(cmd_out.stderr.decode()))

    def make_mask(self):
        # ======================================
        # Make mask

        img_to_process = self.img_target

        self.feedback.pushInfo('\nMaking mask for\n' +
              os.path.basename(self.img_target) + " " + os.path.basename(img_to_process))

        filename, ext = os.path.splitext(os.path.basename(img_to_process))
        self.mask_file = os.path.join(os.path.dirname(os.path.abspath(img_to_process)), filename + "_mask" + ext)
        cmd = 'gdal_calc' if platform.system() == 'Windows' else 'gdal_calc.py' + ' -A "' + img_to_process + \
              '" --type=Byte --co COMPRESS=PACKBITS --outfile="' \
              + self.mask_file + '" --calc="1*(A>0)" --NoDataValue=0 --quiet'
        cmd_out = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if cmd_out.returncode == 0:  # successfully
            self.feedback.pushInfo('Mask created successfully: ' + os.path.basename(self.mask_file))
        else:
            raise QgsProcessingException('\nError creating mask: ' + str(cmd_out.stderr.decode()))

    def apply_mask(self):
        # ======================================
        # Apply mask to image normalized

        self.feedback.pushInfo('\nApplying mask for\n' +
              os.path.basename(self.img_target) + " " + os.path.basename(self.img_norm))
        cmd = 'gdal_calc' if platform.system() == 'Windows' else 'gdal_calc.py' + ' -A "' + self.img_norm + '" -B "' + \
              self.mask_file + '" --type=UInt16 --co COMPRESS=LZW --co PREDICTOR=2 TILED=YES --outfile="' + \
              self.img_norm + '" --calc="A*(B==1)" --NoDataValue=0  --allBands=A  --overwrite --quiet'
        cmd_out = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if cmd_out.returncode == 0:  # successfully
            self.feedback.pushInfo('Mask applied successfully: ' + os.path.basename(self.mask_file))
        else:
            raise QgsProcessingException('\nError applied mask: ' + str(cmd_out.stderr.decode()))

    def clean(self):
        # delete the MAD file
        os.remove(self.img_imad)
        # delete the clip reference image
        if self.img_ref_clip != self.img_ref:
            os.remove(self.img_ref_clip)

