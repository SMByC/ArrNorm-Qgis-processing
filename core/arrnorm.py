# -*- coding: utf-8 -*-
"""
/***************************************************************************
 ArrNorm
                          A QGIS plugin processing
 Automatic relative radiometric normalization
                              -------------------
        copyright            : (C) 2021-2026 by Xavier Corredor Llano, SMByC
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
from qgis.core import QgsProcessingException

from ArrNorm.core import iMad, radcal
from ArrNorm.core import raster_ops


class Normalization:
    def __init__(self, img_ref, img_target, max_iters, conv_threshold, ncp_threshold, neg_to_nodata,
                 mask_ref, mask_ref_nodata, nodata_mask, nodata_mask_value, keep_mask_layer,
                 output_file, feedback):
        self.img_ref = img_ref
        self.img_target = img_target
        self.max_iters = max_iters
        self.conv_threshold = conv_threshold
        self.ncp_threshold = ncp_threshold
        self.neg_to_nodata = neg_to_nodata
        self.mask_ref = mask_ref
        self.nodata_mask = nodata_mask
        self.keep_mask_layer = keep_mask_layer
        self.output_file = output_file
        self.feedback = feedback

        self.img_ref_clip = img_ref  # safe default if clean() is called before clipper()
        self.img_imad = None
        self.img_norm = None
        self.no_neg = None
        self.norm_masked = None
        self.mask_file = None

        # Output dtype: use the higher-precision type of the two inputs.
        ref_ds = gdal.Open(self.img_ref, GA_ReadOnly)
        ref_band = ref_ds.GetRasterBand(1)
        ref_dtype_code = ref_band.DataType
        ref_nodata = ref_band.GetNoDataValue()
        ref_band = None
        ref_ds = None

        target_ds = gdal.Open(self.img_target, GA_ReadOnly)
        target_band = target_ds.GetRasterBand(1)
        target_dtype_code = target_band.DataType
        target_nodata = target_band.GetNoDataValue()
        target_band = None
        target_ds = None

        self.out_dtype = max(ref_dtype_code, target_dtype_code)

        # Nodata value for output masking: user-provided > auto-detect from target > 0.
        self.mask_nodata = (nodata_mask_value if nodata_mask_value is not None
                            else (target_nodata if target_nodata is not None else 0))

        # Nodata value for reference masking: user-provided > auto-detect from reference > 0.
        self.ref_mask_nodata = (mask_ref_nodata if mask_ref_nodata is not None
                                else (ref_nodata if ref_nodata is not None else 0))

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
        """Reproject and clip the reference image onto the target's exact pixel grid.

        The output reference clip will have:
          - the same CRS as the target,
          - the same upper-left origin (to full floating-point precision),
          - the same pixel width and height,
          - and therefore the same number of columns and rows.

        This guarantees that every (row, col) position in the clipped reference
        corresponds spatially to the identical ground location in the target,
        which is required for the pixel-by-pixel IR-MAD and Radcal correlation.

        When mask_ref is enabled, nodata masking is applied in the same gdal.Warp
        pass via srcNodata/dstNodata — no extra processing step is needed.
        """

        # Read target and reference geotransform metadata
        target_ds = gdal.Open(self.img_target, GA_ReadOnly)
        target_proj = target_ds.GetProjection()
        target_gt   = target_ds.GetGeoTransform()   # (originX, pixW, rot, originY, rot, pixH)
        target_cols = target_ds.RasterXSize
        target_rows = target_ds.RasterYSize
        target_ds   = None

        ref_ds   = gdal.Open(self.img_ref, GA_ReadOnly)
        ref_proj = ref_ds.GetProjection()
        ref_gt   = ref_ds.GetGeoTransform()
        ref_cols = ref_ds.RasterXSize
        ref_rows = ref_ds.RasterYSize
        ref_ds   = None

        # Perfect alignment: same CRS, same origin + pixel size (geotransform), same dimensions.
        # Only then is the reference already on the identical pixel grid as the target.
        already_aligned = (ref_proj == target_proj
                           and ref_gt == target_gt
                           and ref_cols == target_cols
                           and ref_rows == target_rows)

        if already_aligned and not self.mask_ref:
            self.feedback.pushInfo(
                "\nReference image is already perfectly aligned with target "
                "(same CRS, geotransform and dimensions). No clipping needed.\n")
            self.img_ref_clip = self.img_ref
            return

        if already_aligned:
            self.feedback.pushInfo(
                "\nReference image is already aligned with target. "
                "Applying nodata masking (nodata value: {nd}).\n".format(nd=self.ref_mask_nodata))
        else:
            self.feedback.pushInfo(
                "\nReprojecting/aligning reference image to target pixel grid "
                "(CRS: {crs}, resolution: {pixW:.6g} x {pixH:.6g}, "
                "dimensions: {cols} x {rows}{mask})\n".format(
                    crs=target_proj[:50] if target_proj else 'unknown',
                    pixW=target_gt[1], pixH=abs(target_gt[5]),
                    cols=target_cols, rows=target_rows,
                    mask=", with nodata masking (nodata value: {nd})".format(
                        nd=self.ref_mask_nodata) if self.mask_ref else ""))

        filename, ext = os.path.splitext(os.path.basename(self.img_ref))
        self.img_ref_clip = os.path.join(
            os.path.dirname(os.path.abspath(self.img_target)),
            filename + "_" + os.path.splitext(os.path.basename(self.img_target))[0] + "_clip" + ext
        )

        try:
            if already_aligned:
                # Reference is on the correct grid — single-pass copy with nodata masking applied.
                result = gdal.Warp(
                    self.img_ref_clip, self.img_ref,
                    format='GTiff',
                    srcNodata=self.ref_mask_nodata,
                    dstNodata=self.ref_mask_nodata,
                )
            else:
                # Derive exact output bounds from the target's geotransform values.
                # Using the raw geotransform avoids any floating-point rounding that
                # would occur from intermediate extent string/float conversions.
                xmin = target_gt[0]                              # left edge
                ymax = target_gt[3]                              # top edge
                xmax = xmin + target_gt[1] * target_cols        # right edge
                ymin = ymax + target_gt[5] * target_rows        # bottom edge (gt[5] < 0)

                # gdal.Warp with dstSRS + geotransform-derived outputBounds + exact width/height:
                #   dstSRS        → reproject reference to target CRS (no-op if already the same)
                #   outputBounds  → clip/pad to the exact target extent (in target-CRS coordinates)
                #   width/height  → force exactly target_cols x target_rows output pixels
                #
                # Together these parameters make the output geotransform identical to the target's,
                # guaranteeing pixel-for-pixel spatial coincidence for IR-MAD and Radcal.
                warp_kwargs = dict(
                    format='GTiff',
                    dstSRS=target_proj,
                    outputBounds=(xmin, ymin, xmax, ymax),  # (minX, minY, maxX, maxY)
                    width=target_cols,
                    height=target_rows,
                    resampleAlg=gdal.GRA_Bilinear,
                )
                if self.mask_ref:
                    warp_kwargs['srcNodata'] = self.ref_mask_nodata
                    warp_kwargs['dstNodata'] = self.ref_mask_nodata

                result = gdal.Warp(self.img_ref_clip, self.img_ref, **warp_kwargs)

            if result is None:
                raise RuntimeError('gdal.Warp returned None — check GDAL error log.')
            result = None  # close/release the output dataset
            self.feedback.pushInfo(
                'Reference prepared successfully: ' + os.path.basename(self.img_ref_clip))
        except Exception as e:
            self.clean()
            raise QgsProcessingException('\nError clipping/reprojecting reference image: ' + str(e))

    def imad(self):
        # ======================================
        # iMad process

        self.feedback.pushInfo("\niMad process for:\n" +
              os.path.basename(self.img_ref_clip) + " " + os.path.basename(self.img_target))
        self.img_imad = iMad.main(self.img_ref_clip, self.img_target, max_iters=self.max_iters, conv_threshold=self.conv_threshold, feedback=self.feedback)

    def radcal(self):
        # ======================================
        # Radcal process

        filename, ext = os.path.splitext(os.path.basename(self.img_target))
        self.img_norm = os.path.join(os.path.dirname(os.path.abspath(self.img_target)), filename + "_radcal" + ext)

        self.feedback.pushInfo("\nRadcal process for\n" +
              os.path.basename(self.img_ref_clip) + " " + os.path.basename(self.img_target) +
              " with iMad image: " + os.path.basename(self.img_imad))
        radcal.main(self.img_imad, img_ref=self.img_ref_clip, img_tgt=self.img_target, output=self.img_norm,
                    ncp_threshold=self.ncp_threshold, out_dtype=self.out_dtype, feedback=self.feedback)

    def no_negative_value(self, image):
        # ======================================
        # Convert negative values to NoData to image normalized

        filename, ext = os.path.splitext(os.path.basename(self.img_target))
        self.no_neg = os.path.join(os.path.dirname(os.path.abspath(self.img_target)), filename + "_no_neg" + ext)

        self.feedback.pushInfo('\nConverting negative values for\n' + os.path.basename(os.path.basename(image)))

        try:
            raster_ops.no_negative_value(
                image, self.no_neg, nodata_value=self.mask_nodata,
                creation_options=["BIGTIFF=YES"])
            self.feedback.pushInfo('Negative values converted successfully: ' + os.path.basename(image))
        except Exception as e:
            self.clean()
            raise QgsProcessingException('\nError converting values: ' + str(e))

    def make_mask(self):
        # ======================================
        # Make nodata mask from target image

        img_to_process = self.img_target

        self.feedback.pushInfo('\nMaking nodata mask (nodata value: {nd}) for target image:\n'.format(
              nd=self.mask_nodata) + os.path.basename(img_to_process))

        filename, ext = os.path.splitext(os.path.basename(self.output_file))
        self.mask_file = os.path.join(os.path.dirname(os.path.abspath(self.output_file)), filename + "_Mask" + ext)

        try:
            # "1*(A!=nodata)" is more general than the old "1*(A>0)" which incorrectly
            # treated negative values (valid in some sensor products) as nodata.
            raster_ops.make_mask(img_to_process, self.mask_file, nodata_value=self.mask_nodata)

            self.feedback.pushInfo('Mask created successfully: ' + os.path.basename(self.mask_file))
        except Exception as e:
            self.clean()
            raise QgsProcessingException('\nError creating mask: ' + str(e))

    def apply_mask(self, image):
        # ======================================
        # Apply nodata mask to normalized output

        filename, ext = os.path.splitext(os.path.basename(self.img_target))
        self.norm_masked = os.path.join(os.path.dirname(os.path.abspath(self.img_target)), filename + "_norm_masked" + ext)

        self.feedback.pushInfo('\nApplying nodata mask to\n' +
              os.path.basename(self.img_target) + " " + os.path.basename(image))

        try:
            raster_ops.apply_mask(
                image, self.mask_file, self.norm_masked, nodata_value=self.mask_nodata,
                creation_options=["BIGTIFF=YES"])
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
        # delete mask layer only if user did not ask to keep it
        if not self.keep_mask_layer:
            os.remove(self.mask_file) if self.mask_file and os.path.exists(self.mask_file) else None


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
