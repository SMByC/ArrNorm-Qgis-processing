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
import sys
import inspect

from qgis.core import QgsApplication
from ArrNorm.ArrNorm_provider import ArrNormProvider

cmd_folder = os.path.split(inspect.getfile(inspect.currentframe()))[0]

if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)


class ArrNormPlugin:

    def __init__(self):
        self.provider = ArrNormProvider()

    def initProcessing(self):
        """Init Processing provider for QGIS >= 3.8."""
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        self.initProcessing()

    def unload(self):
        # unload dll
        if platform.system() == 'Windows':
            from ArrNorm.core.auxil.auxil import lib
            import ctypes
            try:
                ctypes.windll.kernel32.FreeLibrary.argtypes = [ctypes.wintypes.HMODULE]
                ctypes.windll.kernel32.FreeLibrary(lib._handle)
                del lib
            except:
                pass

        QgsApplication.processingRegistry().removeProvider(self.provider)
