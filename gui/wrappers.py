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
 ***************************************************************************/

Custom Processing parameter widget wrappers for the ArrNorm algorithm dialog.
"""
from qgis.PyQt.QtWidgets import QWidget, QHBoxLayout, QLabel, QCheckBox

from qgis.core import QgsRasterLayer, QgsProcessingUtils

from processing.gui.wrappers import WidgetWrapper, DIALOG_STANDARD
from processing.tools import dataobjects

from qgis.gui import QgsDoubleSpinBox

from qgis.core import QgsProcessingParameterDefinition


def _resolve_layer(value, context):
    """Resolve a layer-parameter value to a QgsRasterLayer (or None)."""
    layer = value
    try:
        if hasattr(layer, 'valueAsString'):
            # QgsProcessingParameterRasterLayer-like dynamic value
            layer, _ = layer.valueAsString(context.expressionContext())
        if isinstance(layer, str) and layer:
            layer = QgsProcessingUtils.mapLayerFromString(layer, context)
    except Exception:
        return None
    if isinstance(layer, QgsRasterLayer) and layer.isValid():
        return layer
    return None


def _layer_nodata(layer):
    """Embedded nodata of band 1, or None when the layer declares none."""
    try:
        provider = layer.dataProvider()
        if provider is not None and provider.sourceHasNoDataValue(1):
            return float(provider.sourceNoDataValue(1))
    except Exception:
        pass
    return None


class ImageNodataWidgetWrapper(WidgetWrapper):
    """Nodata spin box bound to a checkbox (enable) and a raster layer (default).

    Wired via parameter metadata::

        param.setMetadata({'widget_wrapper': {
            'class': 'ArrNorm.gui.wrappers.ImageNodataWidgetWrapper',
            'enabled_by': '<boolean parameter name>',
            'layer_param': '<raster layer parameter name>',
        }})

    Behaviour (standard dialog):
      * disabled while the bound checkbox is unchecked;
      * when a layer is selected, if it declares a nodata value the spin box
        is pre-filled with it (explicit); otherwise it stays on "Auto" and the
        algorithm auto-detects (fallback 0);
      * the user can always override, and overrides are never clobbered.
    """

    def createWidget(self, enabled_by=None, layer_param=None):
        self._enabled_by = enabled_by
        self._layer_param = layer_param
        self._context = dataobjects.createContext()
        self._user_modified = False
        self._programmatic = False
        self._bool_wrapper = None
        self._layer_wrapper = None

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._spin = QgsDoubleSpinBox()
        self._spin.setDecimals(4)
        self._spin.setMinimum(-1.0e12)
        self._spin.setMaximum(1.0e12)
        self._spin.setValue(0.0)
        self._spin.setEnabled(False)
        self._spin.valueChanged.connect(self._on_user_change)
        layout.addWidget(self._spin, 1)

        self._auto = QCheckBox(self.tr('Auto (from image)'))
        self._auto.setChecked(True)
        self._auto.setToolTip(self.tr(
            'Read the nodata value from the image metadata (fallback: 0). '
            'Uncheck to enter an explicit value.'))
        self._auto.toggled.connect(self._on_auto_toggled)
        layout.addWidget(self._auto, 0)

        return container

    # -- value contract -----------------------------------------------------

    def setValue(self, value):
        self._programmatic = True
        try:
            if value is None or value == '':
                self._auto.setChecked(True)
                self._spin.setEnabled(False)
            else:
                self._auto.setChecked(False)
                self._spin.setEnabled(True)
                try:
                    self._spin.setValue(float(value))
                except (TypeError, ValueError):
                    self._spin.setValue(0.0)
        finally:
            self._programmatic = False

    def value(self):
        if self._auto.isChecked():
            return None
        return self._spin.value()

    # -- internal slots -----------------------------------------------------

    def _on_user_change(self, *args):
        if not self._programmatic:
            self._user_modified = True
        self.widgetValueHasChanged.emit(self)

    def _on_auto_toggled(self, checked):
        self._spin.setEnabled(not checked and self._row_enabled())
        if not self._programmatic:
            if checked:
                self._user_modified = False
                self._apply_layer_default(keep_auto=True)
            else:
                self._user_modified = True
        self.widgetValueHasChanged.emit(self)

    # -- cross-parameter binding (standard dialog only) ---------------------

    def postInitialize(self, wrappers):
        if self.dialogType != DIALOG_STANDARD:
            return
        for wrapper in wrappers:
            if wrapper is self:
                continue
            try:
                name = wrapper.parameterDefinition().name()
            except Exception:
                continue
            if self._enabled_by and name == self._enabled_by:
                self._bool_wrapper = wrapper
                self._safe_connect(wrapper, self._on_enable_changed)
            if self._layer_param and name == self._layer_param:
                self._layer_wrapper = wrapper
                self._safe_connect(wrapper, self._on_layer_changed)

        self._sync_enabled_state()
        self._apply_layer_default()

    @staticmethod
    def _safe_connect(wrapper, slot):
        signal = getattr(wrapper, 'widgetValueHasChanged', None)
        if signal is not None:
            try:
                signal.connect(slot)
            except Exception:
                pass

    def _row_enabled(self):
        if self._bool_wrapper is None:
            return True
        try:
            return bool(self._bool_wrapper.parameterValue())
        except Exception:
            return True

    def _sync_enabled_state(self):
        enabled = self._row_enabled()
        self._auto.setEnabled(enabled)
        self._spin.setEnabled(enabled and not self._auto.isChecked())

    def _on_enable_changed(self, *args):
        self._sync_enabled_state()
        if self._row_enabled():
            self._apply_layer_default()

    def _on_layer_changed(self, *args):
        self._apply_layer_default()

    def _apply_layer_default(self, keep_auto=False):
        if self._layer_wrapper is None:
            return
        if not keep_auto and self._user_modified:
            return
        if not self._row_enabled():
            return
        try:
            layer = _resolve_layer(self._layer_wrapper.parameterValue(), self._context)
        except Exception:
            layer = None
        nodata = _layer_nodata(layer) if layer is not None else None
        self._programmatic = True
        try:
            if nodata is not None:
                self._spin.setValue(nodata)
                if not keep_auto:
                    # Layer change path: switch to explicit mode.
                    self._auto.setChecked(False)
                    self._spin.setEnabled(self._row_enabled())
                # keep_auto path: Auto stays checked, spin stays greyed (set by caller).
            else:
                if not keep_auto:
                    # No declared nodata -> stay on Auto.
                    self._auto.setChecked(True)
                    self._spin.setEnabled(False)
        finally:
            self._programmatic = False


class ParameterSectionHeader(QgsProcessingParameterDefinition):
    """Value-less parameter that only renders a section header in the dialog.

    Not optional (so the dialog does not append a " [optional]" suffix to the
    header text) but it always accepts its empty value, so it never blocks
    execution and is harmless for batch / modeler / headless runs.
    """

    def __init__(self, name, description=''):
        super().__init__(name, description)

    def clone(self):
        copy = ParameterSectionHeader(self.name(), self.description())
        copy.setMetadata(self.metadata())
        return copy

    def type(self):
        return 'arrnorm_section_header'

    def checkValueIsAcceptable(self, value, context=None):
        return True

    def valueAsPythonString(self, value, context):
        return ''

    def asScriptCode(self):
        return ''


class SectionHeaderWidgetWrapper(WidgetWrapper):
    """Renders a ParameterSectionHeader as a full-width bold divider."""

    def createWidget(self, **kwargs):
        label = QLabel(self.parameterDefinition().description())
        label.setStyleSheet(
            'font-weight: bold; padding-top: 10px; padding-bottom: 2px; '
            'border-bottom: 1px solid palette(mid);')
        # ParametersPanel calls widget.setText(description) when the label is
        # None; QLabel.setText keeps this styling, so the result is stable.
        return label

    def createLabel(self):
        # None => the widget spans the full form width (no label column),
        # which is what makes it read as a section divider.
        return None

    def setValue(self, value):
        pass

    def value(self):
        return None
