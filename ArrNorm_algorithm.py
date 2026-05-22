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

from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessingAlgorithm, QgsProcessingParameterDefinition,
                       QgsProcessingParameterRasterDestination, QgsProcessingParameterNumber,
                       QgsProcessingParameterRasterLayer, QgsProcessingParameterBoolean)

from ArrNorm.core.arrnorm import Normalization


class ArrNormAlgorithm(QgsProcessingAlgorithm):
    """
    This algorithm compute a specific statistic using the time
    series of all pixels across (the time) all raster in the specific band
    """

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
    OUTPUT = 'OUTPUT'

    # Value-less parameters used only to render section headers in the dialog.
    SECTION_REF_MASK = 'SECTION_REF_MASK'
    SECTION_OUTPUT = 'SECTION_OUTPUT'

    # Dotted path so the Processing framework imports the GUI wrapper lazily
    # (only when building the dialog), keeping headless execution import-safe.
    _NODATA_WRAPPER = 'ArrNorm.gui.wrappers.ImageNodataWidgetWrapper'
    _SECTION_WRAPPER = 'ArrNorm.gui.wrappers.SectionHeaderWidgetWrapper'

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
        html_help = '''
        <p>ArrNorm applies relative radiometric normalization to a <b>target image</b> using a \
        <b>reference image</b>. By leveraging the linear and affine invariance of the MAD \
        transformation, the IR-MAD algorithm identifies spectrally invariant pixels between the \
        two images, and Radcal fits a per-band linear regression on those pixels to produce the \
        normalized output.</p>

        <p>If the reference and target images are not on the same pixel grid, the reference is \
        automatically reprojected and clipped to match the target before processing.</p>

        <p><b>&#9888; Nodata masking is strongly recommended when nodata pixels are present.</b> \
        Nodata values are arbitrary fill numbers that do not represent actual surface reflectance. \
        Because IR-MAD relies on the multivariate covariance structure of all pixel pairs, these \
        fill values act as extreme outliers that distort the statistical model. Leaving nodata \
        unmasked can corrupt the covariance matrix, shift the canonical variates away from true \
        spectral change directions, and bias the per-band regression coefficients, propagating \
        radiometric errors across the entire normalized output. Masking nodata before processing \
        removes these outliers and yields a more accurate normalization.</p>

        <b>IR-MAD convergence threshold</b> (default 0.99) — Controls when the iterative algorithm stops. \
        The iteration halts when the maximum change in canonical correlations (δ) between successive \
        iterations falls below 1 − threshold (e.g. 0.99 → δ &lt; 0.01). Higher values enforce tighter \
        convergence, yielding more stable no-change weights. Use 0.99 for most cases; 0.999 for \
        maximum precision.<br/>
        <b>No-change pixel probability threshold</b> (default 0.95) — Determines which pixels are used for \
        radiometric calibration. After IR-MAD converges, each pixel receives a chi-square no-change \
        probability; only pixels above this threshold are included in the per-band regression. Higher \
        values select fewer but more reliable invariant pixels. If set too high, the algorithm may fail \
        due to insufficient no-change pixels.</p>
        '''
        return html_help

    def createInstance(self):
        return ArrNormAlgorithm()

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
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
        return QIcon(os.path.join(os.path.dirname(__file__), 'icons', 'arrnorm.svg'))

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

        add_section(self.SECTION_REF_MASK, 'Reference image')

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
                defaultValue=False,
                optional=True
            )
        )

        mask_ref_nodata = QgsProcessingParameterNumber(
            self.MASK_REF_NODATA,
            self.tr('Reference nodata value'),
            type=QgsProcessingParameterNumber.Type.Double,
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
        # Target/output image processing
        # =====================================================================

        add_section(self.SECTION_OUTPUT, 'Target/output image')

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
            type=QgsProcessingParameterNumber.Type.Double,
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

        # =====================================================================
        # Advanced: algorithm tuning
        # =====================================================================

        parameter = \
            QgsProcessingParameterNumber(
                self.MAX_ITERS,
                self.tr('Maximum number of iterations'),
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=25,
                optional=True
            )
        parameter.setFlags(parameter.flags() | QgsProcessingParameterDefinition.Flag.FlagAdvanced)
        self.addParameter(parameter)

        parameter = \
            QgsProcessingParameterNumber(
                self.CONV_THRESHOLD,
                self.tr('IR-MAD convergence threshold'),
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=0.99,
                optional=True
            )
        parameter.setFlags(parameter.flags() | QgsProcessingParameterDefinition.Flag.FlagAdvanced)
        self.addParameter(parameter)

        parameter = \
            QgsProcessingParameterNumber(
                self.NCP_THRESHOLD,
                self.tr('No-change pixel probability threshold'),
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=0.95,
                optional=True
            )
        parameter.setFlags(parameter.flags() | QgsProcessingParameterDefinition.Flag.FlagAdvanced)
        self.addParameter(parameter)

        # =====================================================================
        # Output
        # =====================================================================

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                self.tr('Normalized output raster')
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        """
        Here is where the processing itself takes place.
        """

        def get_inputfilepath(layer):
            source = layer.source()
            # Strip QGIS layername suffix if present (e.g. GeoPackage layers)
            path = source.split("|layername")[0]
            # Handle database/WMS/in-memory sources that aren't file paths
            if not os.path.exists(path):
                feedback.reportError(f"Reference/target source is not a valid file path: {path}")
                return None
            return os.path.realpath(path)

        output_file = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)

        # Optional nodata values: None means auto-detect from the respective image.
        mask_ref_nodata_raw = parameters.get(self.MASK_REF_NODATA)
        if mask_ref_nodata_raw is not None and str(mask_ref_nodata_raw).strip():
            mask_ref_nodata = float(mask_ref_nodata_raw)
        else:
            mask_ref_nodata = None

        nodata_mask_value_raw = parameters.get(self.NODATA_MASK_VALUE)
        if nodata_mask_value_raw is not None and str(nodata_mask_value_raw).strip():
            nodata_mask_value = float(nodata_mask_value_raw)
        else:
            nodata_mask_value = None

        arrnorm = Normalization(
            img_ref=get_inputfilepath(self.parameterAsRasterLayer(parameters, self.IMG_REF, context)),
            img_target=get_inputfilepath(self.parameterAsRasterLayer(parameters, self.IMG_TARGET, context)),
            max_iters=self.parameterAsInt(parameters, self.MAX_ITERS, context) or 25,
            conv_threshold=self.parameterAsDouble(parameters, self.CONV_THRESHOLD, context) or 0.99,
            ncp_threshold=self.parameterAsDouble(parameters, self.NCP_THRESHOLD, context) or 0.95,
            neg_to_nodata=self.parameterAsBoolean(parameters, self.NEG_TO_NODATA, context),
            mask_ref=self.parameterAsBoolean(parameters, self.MASK_REF, context),
            mask_ref_nodata=mask_ref_nodata,
            nodata_mask=self.parameterAsBoolean(parameters, self.NODATA_MASK, context),
            nodata_mask_value=nodata_mask_value,
            keep_mask_layer=self.parameterAsBoolean(parameters, self.KEEP_MASK_LAYER, context),
            output_file=output_file,
            feedback=feedback)

        arrnorm.run()

        return {self.OUTPUT: output_file}
