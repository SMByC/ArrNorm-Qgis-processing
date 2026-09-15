"""Real offscreen QGIS/Qt tests; skipped explicitly on core-only installations."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
qgis = pytest.importorskip('qgis.core')
if qgis.Qgis.QGIS_VERSION_INT < 33600:
    pytest.skip('ArrNorm supports QGIS 3.36 and later', allow_module_level=True)
if not qgis.QgsApplication.prefixPath():
    qgis.QgsApplication.setPrefixPath(os.environ.get('QGIS_PREFIX_PATH', sys.prefix), True)
plugins = str(Path(qgis.QgsApplication.pkgDataPath()) / 'python' / 'plugins')
if plugins not in sys.path:
    sys.path.insert(0, plugins)
pytest.importorskip('processing.gui.wrappers')

from ArrNorm import classFactory
from ArrNorm.ArrNorm_algorithm import ArrNormAlgorithm
from ArrNorm.gui.wrappers import ImageNodataWidgetWrapper
from ArrNorm.tests.helpers import build_pair, write_raster
from qgis.PyQt.QtCore import QCoreApplication, QEvent


@pytest.fixture(scope='session')
def qgis_app():
    existing = qgis.QgsApplication.instance()
    if existing is not None:
        yield existing
        return
    app = qgis.QgsApplication([], False)
    app.initQgis()
    yield app
    app.exitQgis()


def wrapper():
    param = qgis.QgsProcessingParameterNumber('NODATA', optional=True,
                                              type=qgis.Qgis.ProcessingNumberParameterType.Double)
    return ImageNodataWidgetWrapper(param, SimpleNamespace())


@pytest.mark.parametrize('nodata', [-3.4028234663852886e38, np.nan, 0.123456])
def test_auto_never_serializes_rounded_metadata(qgis_app, tmp_path, nodata):
    path = tmp_path / 'image.tif'
    write_raster(path, [np.array([[10., nodata], [20., 30.]])], nodata=nodata)
    layer = qgis.QgsRasterLayer(str(path), 'raster')
    ui = wrapper()
    try:
        ui._layer_wrapper = SimpleNamespace(parameterValue=lambda: layer)
        ui._apply_layer_default()
        assert ui._auto.isChecked()
        assert ui.value() is None
        assert not ui._spin.isEnabled()
        assert 'Image nodata:' in ui._spin.toolTip()
    finally:
        ui.widget.deleteLater()
        ui.deleteLater()


def test_explicit_value_survives_layer_changes_and_disabled_state(qgis_app, tmp_path):
    write_raster(tmp_path / 'image.tif', [np.ones((2, 2))], nodata=-9999)
    layer = qgis.QgsRasterLayer(str(tmp_path / 'image.tif'), 'image')
    ui = wrapper()
    try:
        ui._bool_wrapper = SimpleNamespace(parameterValue=lambda: False)
        ui.setValue(0.1234567890123456)
        ui._layer_wrapper = SimpleNamespace(parameterValue=lambda: layer)
        ui._on_layer_changed()
        assert ui.value() == 0.1234567890123456
        assert not ui._spin.isEnabled()
        ui.setValue(None)
        assert ui.value() is None
        ui._bool_wrapper = SimpleNamespace(parameterValue=lambda: True)
        ui._on_enable_changed()
        assert ui.value() is None
        ui._auto.setChecked(False)
        ui._spin.setValue(-42)
        assert ui.value() == -42
    finally:
        ui.widget.deleteLater()
        ui.deleteLater()


def test_project_close_does_not_leave_an_effective_stale_sentinel(qgis_app, tmp_path):
    write_raster(tmp_path / 'image.tif', [np.ones((2, 2))], nodata=-9999)
    layer = qgis.QgsRasterLayer(str(tmp_path / 'image.tif'), 'image')
    project = qgis.QgsProject.instance()
    project.addMapLayer(layer)
    ui = wrapper()
    try:
        ui._layer_wrapper = SimpleNamespace(parameterValue=lambda: layer)
        ui._apply_layer_default()
        project.removeAllMapLayers()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        ui._on_layer_changed()  # wrapper may still expose a deleted SIP object
        assert ui.value() is None
    finally:
        ui.widget.deleteLater()
        ui.deleteLater()


def test_repeated_plugin_lifecycle_and_output_formats(qgis_app):
    plugin = classFactory(None)
    registry = qgis.QgsApplication.processingRegistry()
    try:
        for _ in range(3):
            plugin.initGui()
            plugin.initProcessing()  # repeated initialization is idempotent
            provider = registry.providerById('arrnorm')
            assert provider is plugin.provider
            algorithm = provider.algorithms()[0]
            assert algorithm.parameterDefinition('OUTPUT').defaultFileExtension() == 'tif'
            if qgis.Qgis.QGIS_VERSION_INT >= 40000:
                assert provider.supportedOutputRasterLayerFormatAndExtensions() == [
                    ('GeoTIFF', 'tif'), ('GeoTIFF', 'tiff')]
            command = algorithm.asPythonCommand({'IMG_REF': '/tmp/ref.tif',
                                                 'IMG_TARGET': '/tmp/target.tif',
                                                 'OUTPUT': '/tmp/out.tif'}, qgis.QgsProcessingContext())
            compile(command, '<processing>', 'exec')
            plugin.unload()
            plugin.unload()
            assert registry.providerById('arrnorm') is None
    finally:
        plugin.unload()


def test_headless_parameter_values_and_qgsproperty(qgis_app, tmp_path, monkeypatch):
    import ArrNorm.ArrNorm_algorithm as module
    build_pair(tmp_path)
    observed = {}

    class Capture:
        def __init__(self, **kwargs):
            observed.update(kwargs)
        def run(self):
            return observed['output_file']

    monkeypatch.setattr(module, 'Normalization', Capture)
    algorithm = ArrNormAlgorithm()
    algorithm.initAlgorithm()
    context = qgis.QgsProcessingContext()
    result = algorithm.processAlgorithm({
        'IMG_REF': str(tmp_path / 'ref.tif'), 'IMG_TARGET': str(tmp_path / 'tgt.tif'),
        'OUTPUT': str(tmp_path / 'out.tif'), 'NCP_THRESHOLD': 0,
        'NODATA_MASK_VALUE': qgis.QgsProperty.fromExpression('-9000 - 999'),
    }, context, qgis.QgsProcessingFeedback())
    assert observed['nodata_mask_value'] == -9999
    assert observed['ncp_threshold'] == 0
    assert observed['mask_ref_nodata'] is None
    assert result == {'OUTPUT': str(tmp_path / 'out.tif')}


def test_processing_cancel_returns_no_output(qgis_app, tmp_path):
    build_pair(tmp_path)

    class CancelFeedback(qgis.QgsProcessingFeedback):
        def pushInfo(self, msg):
            if 'Applying nodata mask' in msg:
                self.cancel()

    algorithm = ArrNormAlgorithm()
    algorithm.initAlgorithm()
    destination = tmp_path / 'out.tif'
    destination.write_bytes(b'previous output')
    result = algorithm.processAlgorithm({
        'IMG_REF': str(tmp_path / 'ref.tif'), 'IMG_TARGET': str(tmp_path / 'tgt.tif'),
        'OUTPUT': str(destination), 'NODATA_MASK': True,
    }, qgis.QgsProcessingContext(), CancelFeedback())
    assert result == {}
    assert destination.read_bytes() == b'previous output'


def test_processing_invalid_layer_has_actionable_exception(qgis_app, tmp_path):
    algorithm = ArrNormAlgorithm()
    algorithm.initAlgorithm()
    with pytest.raises(qgis.QgsProcessingException, match='missing or invalid'):
        algorithm.processAlgorithm({'IMG_REF': str(tmp_path / 'missing.tif'),
                                    'OUTPUT': str(tmp_path / 'out.tif')},
                                   qgis.QgsProcessingContext(), qgis.QgsProcessingFeedback())


def test_provider_recovers_when_its_own_registration_was_removed(qgis_app):
    plugin = classFactory(None)
    registry = qgis.QgsApplication.processingRegistry()
    try:
        plugin.initGui()
        previous = plugin.provider
        registry.removeProvider(previous)
        plugin.initProcessing()
        assert registry.providerById('arrnorm') is plugin.provider
        assert plugin.provider is not previous
    finally:
        existing = registry.providerById('arrnorm')
        if existing is not None:
            registry.removeProvider(existing)
        plugin.provider = None


def test_duplicate_instance_cannot_unload_another_instances_provider(qgis_app):
    owner, duplicate = classFactory(None), classFactory(None)
    registry = qgis.QgsApplication.processingRegistry()
    try:
        owner.initGui()
        with pytest.raises(RuntimeError, match='already registered'):
            duplicate.initGui()
        duplicate.unload()
        assert registry.providerById('arrnorm') is owner.provider
        assert owner.provider.algorithms()
    finally:
        existing = registry.providerById('arrnorm')
        if existing is not None:
            registry.removeProvider(existing)
        owner.provider = duplicate.provider = None


def test_qgis4_consumes_format_tuples_through_cpp_api(qgis_app):
    if qgis.Qgis.QGIS_VERSION_INT < 40000:
        pytest.skip('This checks the QGIS 4 non-virtual C++ extension API')
    plugin = classFactory(None)
    try:
        plugin.initGui()
        provider = plugin.provider
        # Invoke the C++ base implementation, which dispatches to the Python
        # format/extension override; calling the override alone proves nothing.
        extensions = qgis.QgsProcessingProvider.supportedOutputRasterLayerExtensions(provider)
        assert extensions == ['tif', 'tiff']
        parameter = provider.algorithms()[0].parameterDefinition('OUTPUT')
        assert '*.tif' in parameter.createFileFilter()
        assert '*.vrt' not in parameter.createFileFilter()
    finally:
        plugin.unload()


def test_processing_temporary_output_satisfies_directory_validation(qgis_app, tmp_path):
    build_pair(tmp_path)
    algorithm = ArrNormAlgorithm()
    algorithm.initAlgorithm()
    context = qgis.QgsProcessingContext()
    context.setTemporaryFolder(str(tmp_path))
    result = algorithm.processAlgorithm({
        'IMG_REF': str(tmp_path / 'ref.tif'), 'IMG_TARGET': str(tmp_path / 'tgt.tif'),
        'OUTPUT': qgis.QgsProcessing.TEMPORARY_OUTPUT,
    }, context, qgis.QgsProcessingFeedback())
    assert Path(result['OUTPUT']).is_file()
