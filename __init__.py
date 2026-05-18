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
import site


def pre_init_plugin():
    # check importing osgeo_utils else load extra python dependencies
    # https://pypi.org/project/gdal-utils/#files
    try:
        import osgeo_utils
    except ImportError:
        extra_libs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "extlibs"))
        if os.path.isdir(extra_libs_path):
            site.addsitedir(extra_libs_path)



# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load ArrNorm class from file ArrNorm.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    # load extra python dependencies
    pre_init_plugin()

    #
    from ArrNorm.ArrNorm_plugin import ArrNormPlugin
    return ArrNormPlugin()
