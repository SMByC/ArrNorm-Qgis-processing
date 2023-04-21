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
import shutil
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly
from osgeo_utils.gdal_calc import Calc as gdal_calc

from qgis.core import QgsProcessingException

from ArrNorm.core import iMad, radcal


class Normalization:
    def __init__(self, img_ref, img_target, max_iters, prob_thres, neg_to_nodata, nodata_mask, output_file, feedback):
        self.img_ref = img_ref
        self.img_target = img_target
        self.max_iters = max_iters
        self.prob_thres = prob_thres
        self.neg_to_nodata = neg_to_nodata
        self.nodata_mask = nodata_mask
        self.output_file = output_file
        self.feedback = feedback

        self.img_imad = None
        self.img_norm = None
        self.no_neg = None
        self.norm_masked = None

        # define the output type
        ref_dtype = gdal.Open(self.img_ref, GA_ReadOnly).GetRasterBand(1).DataType
        target_dtype = gdal.Open(self.img_ref, GA_ReadOnly).GetRasterBand(1).DataType
        self.out_dtype = ref_dtype if ref_dtype > target_dtype else target_dtype
        ref_dtype = None
        target_dtype = None

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

        # check if the reference and target images have the same rows and columns
        if get_rows_cols_from_raster(self.img_ref) == get_rows_cols_from_raster(self.img_target):
            self.feedback.pushInfo("\nImages have the same dimensions, not necessary to clip\n")
            self.img_ref_clip = self.img_ref
            return

        self.feedback.pushInfo("\nClipping the ref image with target\n")

        filename, ext = os.path.splitext(os.path.basename(self.img_ref))
        self.img_ref_clip = os.path.join(os.path.dirname(os.path.abspath(self.img_target)),
                                         filename + "_" + os.path.splitext(os.path.basename(self.img_target))[0]
                                         + "_clip" + ext)

        try:
            gdal.Translate(self.img_ref_clip, self.img_ref, projWin=get_extent_from_raster(self.img_target), format='GTiff')
            self.feedback.pushInfo('Clipped ref image successfully: ' + os.path.basename(self.img_ref_clip))
        except Exception as e:
            self.clean()
            raise QgsProcessingException('\nError clipping reference image: ' + str(e))

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
        radcal.main(self.img_imad, img_ref=self.img_ref_clip, img_tgt=self.img_target, output=self.img_norm,
                    ncpThresh=self.prob_thres, out_dtype=self.out_dtype)

    def no_negative_value(self, image):
        # ======================================
        # Convert negative values to NoData to image normalized

        filename, ext = os.path.splitext(os.path.basename(self.img_target))
        self.no_neg = os.path.join(os.path.dirname(os.path.abspath(self.img_target)), filename + "_no_neg" + ext)

        self.feedback.pushInfo('\nConverting negative values for\n' + os.path.basename(os.path.basename(image)))

        try:
            gdal_calc(A=image, outfile=self.no_neg, calc="A*(A>=0)", NoDataValue=get_no_data_value_from_raster(self.img_target),
                      allBands="A", overwrite=True, quiet=True, creation_options=["BIGTIFF=YES"])
            self.feedback.pushInfo('Negative values converted successfully: ' + os.path.basename(image))
        except Exception as e:
            self.clean()
            raise QgsProcessingException('\nError converting values: ' + str(e))

    def make_mask(self):
        # ======================================
        # Make mask

        img_to_process = self.img_target

        self.feedback.pushInfo('\nMaking mask for\n' +
              os.path.basename(self.img_target) + " " + os.path.basename(img_to_process))

        filename, ext = os.path.splitext(os.path.basename(self.output_file))
        self.mask_file = os.path.join(os.path.dirname(os.path.abspath(self.output_file)), filename + "_Mask" + ext)

        try:
            gdal_calc(A=img_to_process, outfile=self.mask_file, calc="1*(A>0)", type=gdal.GDT_Byte,
                      allBands="A", overwrite=True, quiet=True, creation_options=["COMPRESS=PACKBITS"])
            self.feedback.pushInfo('Mask created successfully: ' + os.path.basename(self.mask_file))
        except Exception as e:
            self.clean()
            raise QgsProcessingException('\nError creating mask: ' + str(e))

    def apply_mask(self, image):
        # ======================================
        # Apply mask to image normalized

        filename, ext = os.path.splitext(os.path.basename(self.img_target))
        self.norm_masked = os.path.join(os.path.dirname(os.path.abspath(self.img_target)), filename + "_norm_masked" + ext)

        self.feedback.pushInfo('\nApplying mask for\n' +
              os.path.basename(self.img_target) + " " + os.path.basename(image))

        try:
            gdal_calc(A=image, B=self.mask_file, outfile=self.norm_masked, calc="A*(B==1)",
                      NoDataValue=get_no_data_value_from_raster(self.img_target),
                      allBands="A", overwrite=True, quiet=True)
            self.feedback.pushInfo('Mask applied successfully: ' + os.path.basename(self.mask_file))
        except Exception as e:
            self.clean()
            raise QgsProcessingException('\nError applied mask: ' + str(e))

    def clean(self):
        # delete the MAD file
        os.remove(self.img_imad) if self.img_imad and os.path.exists(self.img_imad) else None
        # delete the clip reference image
        if self.img_ref_clip != self.img_ref:
            os.remove(self.img_ref_clip) if os.path.exists(self.img_ref_clip) else None
        os.remove(self.img_norm) if self.img_norm and os.path.exists(self.img_norm) else None
        os.remove(self.no_neg) if self.no_neg and os.path.exists(self.no_neg) else None
        os.remove(self.norm_masked) if self.norm_masked and os.path.exists(self.norm_masked) else None


def get_extent_from_raster(raster_path):
    raster = gdal.Open(raster_path, gdal.GA_ReadOnly)
    geotransform = raster.GetGeoTransform()
    xmin = geotransform[0]
    ymax = geotransform[3]
    xmax = xmin + geotransform[1] * raster.RasterXSize
    ymin = ymax + geotransform[5] * raster.RasterYSize
    del raster
    return [xmin, ymax, xmax, ymin]


def get_rows_cols_from_raster(raster_path):
    raster = gdal.Open(raster_path, gdal.GA_ReadOnly)
    rows = raster.RasterYSize
    cols = raster.RasterXSize
    del raster
    return rows, cols


def get_no_data_value_from_raster(raster_path):
    raster = gdal.Open(raster_path, gdal.GA_ReadOnly)
    no_data_value = raster.GetRasterBand(1).GetNoDataValue()
    del raster
    return no_data_value
