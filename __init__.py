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

from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtWidgets import QMessageBox

from ArrNorm.utils.extra_deps import load_install_extra_deps, WaitDialog


def pre_init_plugin(iface):
    app = QCoreApplication.instance()
    parent = iface.mainWindow()
    dialog = None
    log = ''
    try:
        for msg_type, msg_val in load_install_extra_deps():
            app.processEvents()
            if msg_type == 'log':
                log += msg_val
            elif msg_type == 'needs_install':
                dialog = WaitDialog(parent, 'ArrNorm - installing dependencies')
            elif msg_type == 'install_done':
                dialog.accept()
    except Exception as e:
        if dialog:
            dialog.accept()
        QMessageBox.critical(parent, 'ArrNorm - installing dependencies',
                             'An error occurred during the installation of Python packages. ' +
                             'Click on "Stack Trace" in the QGIS message bar for details.')
        raise RuntimeError('\nArrNorm: Error installing Python packages. Read install instruction: '
                           'https://github.com/SMByC/ArrNorm-Qgis-processing\nLog:\n' + log) from e


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load ArrNorm class from file ArrNorm.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    # load/install extra python dependencies
    # pre_init_plugin(iface)

    #
    from ArrNorm.ArrNorm_plugin import ArrNormPlugin
    return ArrNormPlugin()
