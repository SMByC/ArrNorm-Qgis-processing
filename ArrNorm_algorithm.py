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
import tempfile

from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (Qgis, QgsProcessingAlgorithm, QgsProcessingException,
                       QgsProcessingParameterRasterDestination, QgsProcessingParameterNumber,
                       QgsProcessingParameterRasterLayer, QgsProcessingParameterBoolean)

from ArrNorm.core.arrnorm import Normalization
from ArrNorm.core.iMad import DEFAULT_CONV_THRESHOLD, DEFAULT_MAX_ITERS


class ArrNormAlgorithm(QgsProcessingAlgorithm):
    """Normalize a target raster to a reference using IR-MAD and per-band regression."""

    # Constants used to refer to parameters and outputs. They will be
    # used when calling the algorithm from another algorithm, or when
    # calling from the QGIS console.

    IMG_REF = 'IMG_REF'
    IMG_TARGET = 'IMG_TARGET'
    MAX_ITERS = 'MAX_ITERS'
    CONV_THRESHOLD = 'CONV_THRESHOLD'
    NCP_THRESHOLD = 'NCP_THRESHOLD'
    NEG_TO_NODATA = 'NEG_TO_NODATA'
    MASK_REF = 'MASK_REF'
    MASK_REF_NODATA = 'MASK_REF_NODATA'
    NODATA_MASK = 'NODATA_MASK'
    NODATA_MASK_VALUE = 'NODATA_MASK_VALUE'
    KEEP_MASK_LAYER = 'KEEP_MASK_LAYER'
    REPORT = 'REPORT'
    REPORT_BESIDE_OUTPUT = 'REPORT_BESIDE_OUTPUT'
    OUTPUT = 'OUTPUT'

    # Value-less parameters used only to render section headers in the dialog.
    SECTION_REF_IMAGE = 'SECTION_REF_IMAGE'
    SECTION_TAR_IMAGE = 'SECTION_TAR_IMAGE'
    SECTION_OUTPUT = 'SECTION_OUTPUT'
    SECTION_REPORT = 'SECTION_REPORT'

    # Dotted path so the Processing framework imports the GUI wrapper lazily
    # (only when building the dialog), keeping headless execution import-safe.
    _NODATA_WRAPPER = 'ArrNorm.gui.wrappers.ImageNodataWidgetWrapper'
    _SECTION_WRAPPER = 'ArrNorm.gui.wrappers.SectionHeaderWidgetWrapper'
    _DEPENDENT_BOOL_WRAPPER = 'ArrNorm.gui.wrappers.DependentBooleanWidgetWrapper'

    def __init__(self):
        super().__init__()

    def tr(self, string, context=''):
        if context == '':
            context = self.__class__.__name__
        return QCoreApplication.translate(context, string)

    def shortHelpString(self):
        """
        Returns a localised short helper string for the algorithm. This string
        should provide a basic description about what the algorithm does and the
        parameters and outputs associated with it.
        """
        html_help = f'''
        <p>ArrNorm applies relative radiometric normalization to a <b>target image</b> using a \
        <b>reference image</b>. By leveraging the linear and affine invariance of the MAD \
        transformation, the IR-MAD algorithm identifies spectrally invariant pixels between the \
        two images, and Radcal fits a per-band linear regression on those pixels to produce the \
        normalized output.</p>

        <p>If the reference and target images are not on the same pixel grid, the reference is \
        reprojected and clipped to match the target before processing. This requires a \
        georeferenced, north-up target. Already aligned rasters are reused directly.</p>

        <p><b>&#9888; Nodata masking is strongly recommended when nodata pixels are present.</b> \
        Nodata values are arbitrary fill numbers that do not represent actual surface reflectance. \
        Because IR-MAD relies on the multivariate covariance structure of all pixel pairs, these \
        fill values act as extreme outliers that distort the statistical model. Leaving nodata \
        unmasked can corrupt the covariance matrix, shift the canonical variates away from true \
        spectral change directions, and bias the per-band regression coefficients, propagating \
        radiometric errors across the entire normalized output. Masking nodata before processing \
        removes these outliers and yields a more accurate normalization.</p>

        <p><b>IR-MAD convergence threshold</b> (default {DEFAULT_CONV_THRESHOLD:g}) — Stops when the \
        maximum absolute change in canonical correlations (δ) between successive iterations falls \
        below 1 − threshold (δ &lt; {1 - DEFAULT_CONV_THRESHOLD:.3g} by default). Higher values tighten \
        this numerical stopping tolerance; it is not a confidence level.</p>

        <p><b>Maximum number of iterations</b> (default {DEFAULT_MAX_ITERS}) — An upper limit; converged \
        runs stop earlier. If the limit is reached first, the iteration with the smallest δ is \
        used, but the run is not marked as converged.</p>

        <p><b>No-change pixel probability threshold</b> (default 0.95) — Determines which pixels are used for \
        radiometric calibration. After IR-MAD, each pixel receives a chi-square-based no-change \
        score; only pixels above this threshold are included in the per-band regression. Higher values \
        select fewer pixels with higher model-based no-change scores, but may reduce calibration \
        coverage. If set too high, the algorithm may fail due to insufficient no-change pixels.</p>

        <p><b>Report</b> — Renders two diagnostic figures: a summary (agreement with the \
        reference per band, the applied correction, where the no-change pixels are, and IR-MAD \
        convergence) and a per-band page (affine fit and reference / target-before / target-after \
        value distributions).</p>
        '''
        return html_help

    def createInstance(self):
        return ArrNormAlgorithm()

    def name(self):
        """Retain the published algorithm ID for existing models, scripts and history."""
        return 'Automatic relative radiometric normalization'

    def displayName(self):
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr(self.name())

    def group(self):
        """
        Returns the name of the group this algorithm belongs to. This string
        should be localised.
        """
        return None

    def groupId(self):
        """
        Returns the unique ID of the group this algorithm belongs to. This
        string should be fixed for the algorithm, and must not be localised.
        The group id should be unique within each provider. Group id should
        contain lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return None

    def icon(self):
        return QIcon(':/plugins/ArrNorm/arrnorm.svg')

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """

        # Section-header pseudo parameters give the dialog real visual groups.
        # They carry no value and are import-guarded so headless execution
        # (where qgis.gui may be absent) is never affected.
        try:
            from ArrNorm.gui.wrappers import ParameterSectionHeader
        except Exception:
            ParameterSectionHeader = None

        def add_section(name, title):
            if ParameterSectionHeader is None:
                return
            section = ParameterSectionHeader(name, self.tr(title))
            section.setMetadata({'widget_wrapper': {'class': self._SECTION_WRAPPER}})
            self.addParameter(section)

        # =====================================================================
        # Reference image masking (applied before processing)
        #
        # The custom ImageNodataWidgetWrapper enables the nodata field only
        # while its checkbox is checked, and auto-fills it with the nodata
        # value embedded in the bound raster layer. When the wrapper is not
        # used (batch / modeler / headless) the value contract is unchanged:
        # an empty value (None) still means "auto-detect from the image".
        # =====================================================================

        add_section(self.SECTION_REF_IMAGE, 'Reference image')

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.IMG_REF,
                self.tr('Reference image as a basis for normalization'),
                optional=False
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.MASK_REF,
                self.tr('Mask nodata in reference image before processing'),
                defaultValue=True,
                optional=True
            )
        )

        mask_ref_nodata = QgsProcessingParameterNumber(
            self.MASK_REF_NODATA,
            self.tr('Reference nodata value'),
                type=Qgis.ProcessingNumberParameterType.Double,
            optional=True,
            defaultValue=None
        )
        mask_ref_nodata.setMetadata({
            'widget_wrapper': {
                'class': self._NODATA_WRAPPER,
                'enabled_by': self.MASK_REF,
                'layer_param': self.IMG_REF,
            }
        })
        self.addParameter(mask_ref_nodata)

        # =====================================================================
        # Target image processing
        # =====================================================================

        add_section(self.SECTION_TAR_IMAGE, 'Target image')

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.IMG_TARGET,
                self.tr('Target image to normalize')
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.NODATA_MASK,
                self.tr('Mask nodata in target image before processing (and output)'),
                defaultValue=True,
                optional=True
            )
        )

        nodata_mask_value = QgsProcessingParameterNumber(
            self.NODATA_MASK_VALUE,
            self.tr('Target and output nodata value'),
                type=Qgis.ProcessingNumberParameterType.Double,
            optional=True,
            defaultValue=None
        )
        nodata_mask_value.setMetadata({
            'widget_wrapper': {
                'class': self._NODATA_WRAPPER,
                'enabled_by': self.NODATA_MASK,
                'layer_param': self.IMG_TARGET,
            }
        })
        self.addParameter(nodata_mask_value)

        # =====================================================================
        # Report
        # =====================================================================

        add_section(self.SECTION_REPORT, 'Report')

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.REPORT,
                self.tr('Generate the radiometric normalization report (plots)'),
                defaultValue=True,
                optional=True
            )
        )

        report_beside_output = QgsProcessingParameterBoolean(
            self.REPORT_BESIDE_OUTPUT,
            self.tr('Save the report next to the output file'),
            defaultValue=False,
            optional=True
        )
        report_beside_output.setMetadata({
            'widget_wrapper': {
                'class': self._DEPENDENT_BOOL_WRAPPER,
                'enabled_by': self.REPORT,
            }
        })
        self.addParameter(report_beside_output)

        # =====================================================================
        # Advanced: algorithm tuning
        # =====================================================================

        parameter = \
            QgsProcessingParameterNumber(
                self.MAX_ITERS,
                self.tr('Maximum number of iterations'),
                type=Qgis.ProcessingNumberParameterType.Integer,
                defaultValue=DEFAULT_MAX_ITERS,
                minValue=1,
                optional=True
            )
        parameter.setFlags(parameter.flags() | Qgis.ProcessingParameterFlag.Advanced)
        self.addParameter(parameter)

        parameter = \
            QgsProcessingParameterNumber(
                self.CONV_THRESHOLD,
                self.tr('IR-MAD convergence threshold'),
                type=Qgis.ProcessingNumberParameterType.Double,
                defaultValue=DEFAULT_CONV_THRESHOLD,
                minValue=0,
                maxValue=1,
                optional=True
            )
        parameter.setFlags(parameter.flags() | Qgis.ProcessingParameterFlag.Advanced)
        self.addParameter(parameter)

        parameter = \
            QgsProcessingParameterNumber(
                self.NCP_THRESHOLD,
                self.tr('No-change pixel probability threshold'),
                type=Qgis.ProcessingNumberParameterType.Double,
                defaultValue=0.95,
                minValue=0,
                maxValue=1,
                optional=True
            )
        parameter.setFlags(parameter.flags() | Qgis.ProcessingParameterFlag.Advanced)
        self.addParameter(parameter)

        # =====================================================================
        # Output
        # =====================================================================

        add_section(self.SECTION_OUTPUT, 'Output')

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.NEG_TO_NODATA,
                self.tr('Convert negative values to nodata in normalized output'),
                defaultValue=False,
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.KEEP_MASK_LAYER,
                self.tr('Keep the nodata mask as a separate file (in the same output directory)'),
                defaultValue=False,
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                self.tr('Normalized output raster')
            )
        )

    @staticmethod
    def _report_folder(context):
        """The Processing session temporary folder used for this plugin's runs.

        A report the user did not ask to keep beside the output still needs a
        real location: the run's own workspace is deleted on completion. QGIS
        manages this folder's lifetime. The last-resort fallback is the system
        temporary directory, which nothing cleans up for us; it is only reached
        if `QgsProcessingUtils` is unavailable or both `tempFolder` overloads
        fail.
        """
        try:
            from qgis.core import QgsProcessingUtils
            try:
                folder = QgsProcessingUtils.tempFolder(context)
            except TypeError:          # builds whose tempFolder takes no context
                folder = QgsProcessingUtils.tempFolder()
        except Exception:
            folder = None
        return folder or tempfile.gettempdir()

    def processAlgorithm(self, parameters, context, feedback):
        """
        Here is where the processing itself takes place.
        """

        def get_inputfilepath(layer):
            if layer is None:
                raise QgsProcessingException(
                    self.tr('The reference/target raster layer is missing or invalid.'))
            source = layer.source()
            # Strip QGIS layername suffix if present (e.g. GeoPackage layers)
            path = source.split("|layername")[0]
            # Handle database/WMS/in-memory sources that aren't file paths
            if not os.path.exists(path):
                raise QgsProcessingException(
                    self.tr('Reference/target source is not a valid file path: {path}. '
                            'ArrNorm works on file-based rasters only; database, WMS '
                            'and in-memory sources are not supported.').format(path=path))
            return os.path.realpath(path)

        output_file = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)

        # Optional nodata values: None means auto-detect from the respective image.
        mask_ref_nodata_raw = parameters.get(self.MASK_REF_NODATA)
        if mask_ref_nodata_raw is not None and str(mask_ref_nodata_raw).strip():
            mask_ref_nodata = self.parameterAsDouble(parameters, self.MASK_REF_NODATA, context)
        else:
            mask_ref_nodata = None

        nodata_mask_value_raw = parameters.get(self.NODATA_MASK_VALUE)
        if nodata_mask_value_raw is not None and str(nodata_mask_value_raw).strip():
            nodata_mask_value = self.parameterAsDouble(parameters, self.NODATA_MASK_VALUE, context)
        else:
            nodata_mask_value = None

        report = self.parameterAsBoolean(parameters, self.REPORT, context)
        # An unchecked "next to the output" keeps the figures out of the output
        # directory; they are still embedded in this log either way.
        report_dir = (None if not report
                      or self.parameterAsBoolean(parameters, self.REPORT_BESIDE_OUTPUT, context)
                      else self._report_folder(context))

        arrnorm = Normalization(
            img_ref=get_inputfilepath(self.parameterAsRasterLayer(parameters, self.IMG_REF, context)),
            img_target=get_inputfilepath(self.parameterAsRasterLayer(parameters, self.IMG_TARGET, context)),
            max_iters=(DEFAULT_MAX_ITERS if parameters.get(self.MAX_ITERS) in (None, '') else
                       self.parameterAsInt(parameters, self.MAX_ITERS, context)),
            conv_threshold=(DEFAULT_CONV_THRESHOLD if parameters.get(self.CONV_THRESHOLD) in (None, '') else
                            self.parameterAsDouble(parameters, self.CONV_THRESHOLD, context)),
            ncp_threshold=(0.95 if parameters.get(self.NCP_THRESHOLD) in (None, '') else
                           self.parameterAsDouble(parameters, self.NCP_THRESHOLD, context)),
            neg_to_nodata=self.parameterAsBoolean(parameters, self.NEG_TO_NODATA, context),
            mask_ref=self.parameterAsBoolean(parameters, self.MASK_REF, context),
            mask_ref_nodata=mask_ref_nodata,
            nodata_mask=self.parameterAsBoolean(parameters, self.NODATA_MASK, context),
            nodata_mask_value=nodata_mask_value,
            keep_mask_layer=self.parameterAsBoolean(parameters, self.KEEP_MASK_LAYER, context),
            output_file=output_file,
            feedback=feedback,
            report=report,
            report_dir=report_dir)

        arrnorm.run()

        if feedback.isCanceled():
            # The run was cancelled by the user: report no results instead of
            # an output path that was never created.
            return {}

        return {self.OUTPUT: output_file}
