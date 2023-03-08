# -*- coding: utf-8 -*-
"""
/***************************************************************************
 ArrNorm
                          A QGIS plugin processing
 Automatic relative radiometric normalization
                              -------------------
        copyright            : (C) 2021-2022 by Xavier Corredor Llano, SMByC
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
import shutil
import subprocess

from qgis.core import QgsProcessingException

from ArrNorm.core import iMad, radcal


class Normalization:
    def __init__(self, img_ref, img_target, max_iters, prob_thres, neg_to_nodata, nodata_mask, output_file, feedback):
        self.img_ref = img_ref
        self.img_target = img_target
        self.max_iters = max_iters
        self.prob_thres = prob_thres
        self.nodata_mask = nodata_mask
        self.neg_to_nodata = neg_to_nodata
        self.output_file = output_file
        self.feedback = feedback

        self.img_imad = None
        self.img_norm = None
        self.no_neg = None
        self.norm_masked = None

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

        self.feedback.setProgress(90)
        self.radcal()
        if self.feedback.isCanceled():
            return

        if self.neg_to_nodata:
            self.feedback.setProgress(93)
            self.no_negative_value(self.img_norm)

        if self.nodata_mask:
            self.feedback.setProgress(97)
            self.make_mask()
            self.apply_mask(self.no_neg if self.no_neg else self.img_norm)

        # finish
        if self.norm_masked:
            shutil.move(self.norm_masked, self.output_file)
        elif self.no_neg:
            shutil.move(self.no_neg, self.output_file)
        else:
            shutil.move(self.img_norm, self.output_file)

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
            self.clean()
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
            self.clean()
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

        filename, ext = os.path.splitext(os.path.basename(self.img_target))
        self.img_norm = os.path.join(os.path.dirname(os.path.abspath(self.img_target)), filename + "_radcal" + ext)

        self.feedback.pushInfo("\nRadcal process for\n" +
              os.path.basename(self.img_ref) + " " + os.path.basename(self.img_target) + " " +
              " with iMad image: " + os.path.basename(self.img_imad))
        radcal.main(self.img_imad, self.img_norm, ncpThresh=self.prob_thres)

    def no_negative_value(self, image):
        # ======================================
        # Convert negative values to NoData to image normalized

        filename, ext = os.path.splitext(os.path.basename(self.img_target))
        self.no_neg = os.path.join(os.path.dirname(os.path.abspath(self.img_target)), filename + "_no_neg" + ext)

        self.feedback.pushInfo('\nConverting negative values for\n' +
              os.path.basename(os.path.basename(image)))
        cmd = ['gdal_calc' if platform.system() == 'Windows' else 'gdal_calc.py', '-A', '"{}"'.format(image),
               '--overwrite', '--calc', '"A*(A>=0)"', '--allBands=A', '--NoDataValue=0', '--type=UInt16', '--quiet',
               '--outfile', '"{}"'.format(self.no_neg)]
        cmd_out = subprocess.run(" ".join(cmd), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if cmd_out.returncode == 0:  # successfully
            self.feedback.pushInfo('Negative values converted successfully: ' + os.path.basename(image))
        else:
            self.clean()
            raise QgsProcessingException('\nError converting values: ' + str(cmd_out.stderr.decode()))

    def make_mask(self):
        # ======================================
        # Make mask

        img_to_process = self.img_target

        self.feedback.pushInfo('\nMaking mask for\n' +
              os.path.basename(self.img_target) + " " + os.path.basename(img_to_process))

        filename, ext = os.path.splitext(os.path.basename(self.output_file))
        self.mask_file = os.path.join(os.path.dirname(os.path.abspath(self.output_file)), filename + "_Mask" + ext)

        cmd = ['gdal_calc' if platform.system() == 'Windows' else 'gdal_calc.py', '-A', '"{}"'.format(img_to_process),
               '--overwrite', '--calc', '"1*(A>0)"', '--type=Byte', '--NoDataValue=0', '--co', 'COMPRESS=PACKBITS',
               '--quiet', '--outfile', '"{}"'.format(self.mask_file)]
        cmd_out = subprocess.run(" ".join(cmd), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if cmd_out.returncode == 0:  # successfully
            self.feedback.pushInfo('Mask created successfully: ' + os.path.basename(self.mask_file))
        else:
            self.clean()
            raise QgsProcessingException('\nError creating mask: ' + str(cmd_out.stderr.decode()))

    def apply_mask(self, image):
        # ======================================
        # Apply mask to image normalized

        filename, ext = os.path.splitext(os.path.basename(self.img_target))
        self.norm_masked = os.path.join(os.path.dirname(os.path.abspath(self.img_target)), filename + "_norm_masked" + ext)

        self.feedback.pushInfo('\nApplying mask for\n' +
              os.path.basename(self.img_target) + " " + os.path.basename(image))

        cmd = ['gdal_calc' if platform.system() == 'Windows' else 'gdal_calc.py', '-A', '"{}"'.format(image),
               '-B', '"{}"'.format(self.mask_file), '--overwrite', '--calc', '"A*(B==1)"', '--type=UInt16',
               '--NoDataValue=0', '--co', 'COMPRESS=LZW', '--co', 'PREDICTOR=2', '--quiet', '--allBands=A',
               '--outfile', '"{}"'.format(self.norm_masked)]
        cmd_out = subprocess.run(" ".join(cmd), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if cmd_out.returncode == 0:  # successfully
            self.feedback.pushInfo('Mask applied successfully: ' + os.path.basename(self.mask_file))
        else:
            self.clean()
            raise QgsProcessingException('\nError applied mask: ' + str(cmd_out.stderr.decode()))

    def clean(self):
        # delete the MAD file
        os.remove(self.img_imad) if self.img_imad and os.path.exists(self.img_imad) else None
        # delete the clip reference image
        if self.img_ref_clip != self.img_ref:
            os.remove(self.img_ref_clip) if os.path.exists(self.img_ref_clip) else None
        os.remove(self.img_norm) if self.img_norm and os.path.exists(self.img_norm) else None
        os.remove(self.no_neg) if self.no_neg and os.path.exists(self.no_neg) else None
        os.remove(self.norm_masked) if self.norm_masked and os.path.exists(self.norm_masked) else None

