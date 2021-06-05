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
    MAX_ITERS = 'MAX_ITERS'
    PROB_THRES = 'PROB_THRES'
    NEG_TO_NODATA = 'NEG_TO_NODATA'
    NODATA_MASK = 'NODATA_MASK'
    IMG_TARGET = 'IMG_TARGET'
    OUTPUT = 'OUTPUT'

    def __init__(self):
        super().__init__()

    def tr(self, string, context=''):
        if context == '':
            context = self.__class__.__name__
        return QCoreApplication.translate(context, string)

    def shortHelpString(self):
        return None

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
        return QIcon(":/plugins/ArrNorm/icons/arrnorm.svg")

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.IMG_REF,
                self.tr('The REFERENCE image:'),
                optional=False
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.IMG_TARGET,
                self.tr('The TARGET image to normalize:')
            )
        )

        parameter = \
            QgsProcessingParameterNumber(
                self.MAX_ITERS,
                self.tr('Maximum number of iterations'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=25,
                optional=True
            )
        parameter.setFlags(parameter.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(parameter)

        parameter = \
            QgsProcessingParameterNumber(
                self.PROB_THRES,
                self.tr('No-change probability threshold'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.95,
                optional=True
            )
        parameter.setFlags(parameter.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(parameter)

        parameter = \
            QgsProcessingParameterBoolean(
                self.NEG_TO_NODATA,
                self.tr('Convert all negative values to NoData'),
                defaultValue=True,
                optional=True
            )
        parameter.setFlags(parameter.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(parameter)

        parameter = \
            QgsProcessingParameterBoolean(
                self.NODATA_MASK,
                self.tr('Create and apply nodata mask'),
                defaultValue=True,
                optional=True
            )
        parameter.setFlags(parameter.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(parameter)

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                self.tr('Output raster normalized')
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        """
        Here is where the processing itself takes place.
        """

        def get_inputfilepath(layer):
            return os.path.realpath(layer.source().split("|layername")[0])

        output_file = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)

        arrnorm = Normalization(
            img_ref=get_inputfilepath(self.parameterAsRasterLayer(parameters, self.IMG_REF, context)),
            img_target=get_inputfilepath(self.parameterAsRasterLayer(parameters, self.IMG_TARGET, context)),
            max_iters=self.parameterAsInt(parameters, self.MAX_ITERS, context),
            prob_thres= self.parameterAsDouble(parameters, self.PROB_THRES, context),
            neg_to_nodata=self.parameterAsBoolean(parameters, self.NEG_TO_NODATA, context),
            nodata_mask=self.parameterAsBoolean(parameters, self.NODATA_MASK, context),
            output=output_file,
            feedback=feedback)

        arrnorm.run()

        return {self.OUTPUT: output_file}
