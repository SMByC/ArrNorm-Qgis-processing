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

from qgis.core import QgsApplication

from ArrNorm.ArrNorm_provider import ArrNormProvider


class ArrNormPlugin:

    def __init__(self):
        self.provider = None

    def initProcessing(self):
        """Init Processing provider for QGIS >= 3.8."""
        registry = QgsApplication.processingRegistry()
        existing = registry.providerById('arrnorm')
        if existing is not None and existing is self.provider:
            return
        if existing is not None:
            raise RuntimeError('An ArrNorm processing provider is already registered.')
        # A missing registration may mean our previous C++ provider was
        # deleted by the registry. Never reuse that Python wrapper.
        self.provider = None
        provider = ArrNormProvider()
        if not registry.addProvider(provider):
            raise RuntimeError('Could not register the ArrNorm processing provider.')
        self.provider = provider

    def initGui(self):
        self.initProcessing()

    def unload(self):
        provider, self.provider = self.provider, None
        registry = QgsApplication.processingRegistry()
        if provider is not None and registry.providerById('arrnorm') is provider:
            registry.removeProvider(provider)
