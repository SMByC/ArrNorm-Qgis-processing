"""Automatic relative radiometric normalization.

Copyright (C) 2021-2026 Xavier Corredor Llano, SMByC.
Licensed under the GNU GPL, version 2 or (at your option) any later version.
"""
import os

from osgeo import gdal

try:
    from qgis.core import QgsProcessingException
except ImportError:
    class _ProcessingException(RuntimeError):
        pass
    QgsProcessingException = _ProcessingException

from . import iMad, radcal, raster_ops
from . import raster_io as rio
from .artifacts import ArtifactError, RunArtifacts


class _NullFeedback:
    """Quiet feedback for direct library callers without QGIS or a progress UI."""
    def pushInfo(self, message):
        pass

    def setProgress(self, value):
        pass

    def reportError(self, message, fatalError=False):
        pass

    def isCanceled(self):
        return False


class Normalization:
    def __init__(self, img_ref, img_target, max_iters, conv_threshold, ncp_threshold, neg_to_nodata,
                 mask_ref, mask_ref_nodata, nodata_mask, nodata_mask_value, keep_mask_layer,
                 output_file, feedback=None):
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
        self.feedback = feedback if feedback is not None else _NullFeedback()
        self.img_ref_clip = img_ref
        self.img_imad = self.img_norm = self.norm_masked = self.mask_file = None

        try:
            rio.validate_options(max_iters, conv_threshold, ncp_threshold)
            with rio.open_raster(img_ref) as ref, rio.open_raster(img_target) as target:
                if ref.RasterCount != target.RasterCount or ref.RasterCount == 0:
                    raise ValueError('Reference and target must have the same nonzero band count.')
                codes = [ds.GetRasterBand(b).DataType for ds in (ref, target)
                         for b in range(1, ds.RasterCount + 1)]
                self.out_dtype = rio.promote_dtype(*codes)
                ref_nd = ref.GetRasterBand(1).GetNoDataValue()
                target_nd = target.GetRasterBand(1).GetNoDataValue()
                inputs = [img_ref, img_target] + (ref.GetFileList() or []) + (target.GetFileList() or [])
            # Readers normalize these configured values per input band using
            # rio.band_nodata; one rounded scalar is insufficient for a mixed-
            # dtype VRT. Output nodata has its own storage requirements.
            self.target_nodata = (nodata_mask_value if nodata_mask_value is not None
                                  else target_nd if target_nd is not None else 0)
            self.ref_mask_nodata = (mask_ref_nodata if mask_ref_nodata is not None
                                    else ref_nd if ref_nd is not None else 0)
            self.mask_nodata = self.target_nodata
            if nodata_mask or neg_to_nodata:
                self.mask_nodata = rio.validate_nodata(self.mask_nodata, self.out_dtype)
            self._artifacts = RunArtifacts(output_file, inputs, self.feedback,
                                           keep_mask_layer and nodata_mask)
            self._inputs = inputs
        except (ValueError, RuntimeError, TypeError) as exc:
            raise QgsProcessingException(str(exc)) from exc

    def run(self):
        # A reused instance may have a new destination/feedback. Each execution
        # owns a fresh workspace and validates the current destination.
        if self.feedback is None:
            self.feedback = _NullFeedback()
        self.clean()
        try:
            self._artifacts = RunArtifacts(self.output_file, self._inputs, self.feedback,
                                           self.keep_mask_layer and self.nodata_mask)
            self.img_ref_clip = self.img_ref
            self.img_imad = self.img_norm = self.norm_masked = self.mask_file = None
            rio.check_cancel(self.feedback)
            self.feedback.pushInfo(f'Processing target image: {os.path.basename(self.img_target)}')
            self.feedback.setProgress(0)
            self.clipper()
            rio.check_cancel(self.feedback)
            self.feedback.setProgress(10)
            self.imad()
            rio.check_cancel(self.feedback)
            self.feedback.setProgress(90)
            self.radcal()
            rio.check_cancel(self.feedback)
            if self.nodata_mask:
                self.feedback.setProgress(93)
                self.make_mask()
                rio.check_cancel(self.feedback)
                self.feedback.setProgress(97)
                self.apply_mask(self.img_norm)
            rio.check_cancel(self.feedback)
            self._artifacts.publish(self.norm_masked or self.img_norm, self.mask_file)
            if self.keep_mask_layer and self.nodata_mask:
                self.mask_file = self._artifacts.mask_output
            self.feedback.setProgress(100)
            self.feedback.pushInfo(f'\nDONE: normalized image saved in: {self.output_file}\n')
            return self.output_file
        except rio.Cancelled:
            return None
        except ArtifactError as exc:
            raise QgsProcessingException(str(exc)) from exc
        finally:
            self.clean()

    def clipper(self):
        """Align once; an already aligned reference needs no copy for masking.

        Masking is enforced by IR-MAD/Radcal validity rules, including rotated
        and unprojected grids which are already aligned. A required warp needs
        CRS and a north-up target, preserves the reference bands' combined
        range, and receives the target's exact spatial metadata.
        """
        rio.check_cancel(self.feedback)
        with rio.open_raster(self.img_target) as target, rio.open_raster(self.img_ref) as ref:
            proj, gt = target.GetProjection(), target.GetGeoTransform()
            cols, rows = target.RasterXSize, target.RasterYSize
            aligned = (ref.GetProjection() == proj and ref.GetGeoTransform() == gt
                       and ref.RasterXSize == cols and ref.RasterYSize == rows)
            reference_codes = [ref.GetRasterBand(b).DataType for b in range(1, ref.RasterCount + 1)]
            source_sentinels = (rio.band_nodata(ref, self.ref_mask_nodata,
                                               range(1, ref.RasterCount + 1))
                                if self.mask_ref else None)
        if aligned:
            self.img_ref_clip = self.img_ref
            self.feedback.pushInfo('Reference is already pixel-aligned; no clipping needed.')
            return
        if gt[2] or gt[4] or gt[1] <= 0 or gt[5] >= 0:
            raise QgsProcessingException('A rotated or non-north-up target cannot be warped '
                                         'onto this grid. Reproject the target to a north-up grid first.')
        if not proj:
            raise QgsProcessingException('The target has no coordinate reference system. '
                                          'Assign its CRS or supply an already aligned reference.')
        reference_dtype = rio.promote_dtype(*reference_codes)
        wide_integer_bands = [b for b, code in enumerate(reference_codes, 1)
                              if rio.numpy_dtype(code).kind in 'iu' and rio.numpy_dtype(code).itemsize == 8]
        if wide_integer_bands:
            # Validate exactness before GDAL can round an Int64/UInt64 source
            # into the Float64 alignment raster. Other types need no extra pass.
            with rio.open_raster(self.img_ref) as ref:
                for b in wide_integer_bands:
                    for y, nr in rio.row_blocks(ref.RasterYSize):
                        rio.check_cancel(self.feedback)
                        try:
                            rio.read_band(ref.GetRasterBand(b), 0, y, ref.RasterXSize, nr)
                        except ValueError as exc:
                            raise QgsProcessingException(str(exc)) from exc
        self.img_ref_clip = self._artifacts.path('reference.tif')
        options = {
            'format': 'GTiff', 'dstSRS': proj,
            'outputBounds': (gt[0], gt[3] + gt[5] * rows, gt[0] + gt[1] * cols, gt[3]),
            'width': cols, 'height': rows, 'resampleAlg': gdal.GRA_Bilinear,
            'outputType': reference_dtype,
            'creationOptions': ['BIGTIFF=IF_SAFER'],
            'callback': lambda *_: 0 if self.feedback.isCanceled() else 1,
        }
        if source_sentinels is not None:
            # Explicit srcNodata otherwise enables GDAL's all-bands-must-match
            # rule. Preserve gaps in individual bands when unifying their dtype.
            options.update(srcNodata=' '.join(str(value) for value in source_sentinels),
                           dstNodata=self.ref_mask_nodata,
                           warpOptions=['UNIFIED_SRC_NODATA=PARTIAL'])
        try:
            with rio.Raster(gdal.Warp(self.img_ref_clip, self.img_ref, **options), writable=True) as warped:
                # Warp derives resolution from bounds/dimensions, which can
                # round fractional pixel sizes. Preserve the requested grid
                # exactly rather than relaxing downstream grid validation.
                rio.checked(warped.SetGeoTransform(gt), 'Set aligned reference geotransform')
                rio.checked(warped.SetProjection(proj), 'Set aligned reference projection')
        except RuntimeError as exc:
            rio.check_cancel(self.feedback)
            raise QgsProcessingException(f'Error aligning reference image: {exc}') from exc

    def _step(self, title):
        """Log a highlighted header for a processing stage."""
        self.feedback.pushInfo(f'\n-------------- {title} ---------------')

    def imad(self):
        self._step('iMad process')
        self.img_imad = self._artifacts.path('mad.tif')
        iMad.main(self.img_ref_clip, self.img_target, max_iters=self.max_iters,
                  conv_threshold=self.conv_threshold, output=self.img_imad,
                  nodata_ref=self.ref_mask_nodata if self.mask_ref else None,
                  nodata_tgt=self.target_nodata if self.nodata_mask else None,
                  feedback=self.feedback)

    def radcal(self):
        self._step('Radcal process')
        self.img_norm = self._artifacts.path('calibrated.tif')
        radcal.main(self.img_imad, img_ref=self.img_ref_clip, img_tgt=self.img_target,
                    output=self.img_norm, out_dtype=self.out_dtype,
                    ncp_threshold=self.ncp_threshold,
                    neg_nodata=self.mask_nodata if self.neg_to_nodata else None,
                    nodata_ref=self.ref_mask_nodata if self.mask_ref else None,
                    nodata_tgt=self.target_nodata if self.nodata_mask else None,
                    feedback=self.feedback)

    def make_mask(self):
        self.feedback.pushInfo('\nMaking nodata mask')
        self.mask_file = self._artifacts.path('mask.tif')
        raster_ops.make_mask(self.img_target, self.mask_file, self.target_nodata,
                             feedback=self.feedback)

    def apply_mask(self, image):
        self.feedback.pushInfo('\nApplying nodata mask')
        self.norm_masked = self._artifacts.path('masked.tif')
        raster_ops.apply_mask(image, self.mask_file, self.norm_masked, self.mask_nodata,
                              creation_options=['BIGTIFF=IF_SAFER'], feedback=self.feedback)

    def clean(self):
        """Clean only the workspace this run owns, including incomplete writes."""
        self._artifacts.clean()
