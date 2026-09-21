"""Radiometric calibration: invariant selection, band ordering, output and reports."""
import numpy as np
import pytest
from ArrNorm.core import radcal, report
from ArrNorm.core import raster_io as rio
from ArrNorm.core.auxil.auxil import orthoregress
from ArrNorm.core.radcal import QgsProcessingException
from ArrNorm.tests.helpers import (
    Feedback,
    build_pair,
    make_mad_raster,
    make_normalization,
    read_raster,
    write_raster,
)
from osgeo import gdal


class TestRadcalRegressionStarvation:
    def test_band_without_nochange_pixels_after_nodata_filter_raises(self, tmp_path):
        # Only the nodata strip has high no-change probability. Filtering it
        # must raise instead of fitting an empty regression and writing NaN.
        rows, cols = 20, 20
        drv = gdal.GetDriverByName('GTiff')
        imad = drv.Create(str(tmp_path / 'MAD(ref&tgt).tif'), cols, rows, 5,
                          gdal.GDT_Float32)
        imad.SetGeoTransform((0.0, 10.0, 0.0, 0.0, 0.0, -10.0))
        imad.SetProjection('EPSG:4326')
        for b in range(1, 5):
            imad.GetRasterBand(b).WriteArray(np.zeros((rows, cols), np.float32))
        chisqr = np.full((rows, cols), 1000.0, np.float32)
        chisqr[0:12, :] = 0.0
        imad.GetRasterBand(5).WriteArray(chisqr)
        imad = None

        ref = np.full((rows, cols), 50.0, np.float32)
        tgt = np.full((rows, cols), 60.0, np.float32)
        tgt[0:12, :] = -9999.0
        write_raster(tmp_path / 'ref.tif', [ref] * 4)
        write_raster(tmp_path / 'tgt.tif', [tgt] * 4)

        with pytest.raises(QgsProcessingException, match='no-change pixel'):
            radcal.main(str(tmp_path / 'MAD(ref&tgt).tif'),
                        img_ref=str(tmp_path / 'ref.tif'),
                        img_tgt=str(tmp_path / 'tgt.tif'),
                        output=str(tmp_path / 'out.tif'),
                        out_dtype=gdal.GDT_Float32,
                        nodata_tgt=-9999.0,
                        ncp_threshold=0.95)


def test_band_order_and_block_size_do_not_change_fit(tmp_path):
    values = np.arange(1., 301.).reshape(60, 5)
    reference = [values * 2 + 3, values * 3 + 7]
    target = [values.copy(), values.copy()]
    reference[0][0, 0] = target[0][0, 0] = -9999
    reference[1][0, 0] = 3000  # must be excluded in either order
    write_raster(tmp_path / 'ref.tif', reference)
    write_raster(tmp_path / 'tgt.tif', target)
    mad = make_mad_raster(tmp_path, values.shape)
    for order, block, name in [([1, 2], 7, 'a.tif'), ([2, 1], 23, 'b.tif')]:
        radcal.main(mad, img_ref=str(tmp_path / 'ref.tif'), img_tgt=str(tmp_path / 'tgt.tif'),
                    output=str(tmp_path / name), pos=order, block_rows=block,
                    nodata_ref=-9999, nodata_tgt=-9999, feedback=Feedback())
    np.testing.assert_array_equal(read_raster(tmp_path / 'a.tif')[1], read_raster(tmp_path / 'b.tif')[0])
    np.testing.assert_allclose(read_raster(tmp_path / 'a.tif')[1][1:], reference[1][1:])


def test_integer_negative_conversion_has_known_sentinel(tmp_path):
    target = np.arange(1, 21).reshape(4, 5)
    # Fit on >= 10 only: inverse calibration maps the first pixels below zero.
    reference = np.maximum(2 * target - 15, 0)
    write_raster(tmp_path / 'ref.tif', [reference], dtype=gdal.GDT_UInt16)
    write_raster(tmp_path / 'tgt.tif', [target], dtype=gdal.GDT_UInt16)
    mad = tmp_path / 'mad.tif'
    write_raster(mad, [np.zeros(target.shape), np.where(target >= 10, 0, 1000)])
    radcal.main(str(mad), img_ref=str(tmp_path / 'ref.tif'), img_tgt=str(tmp_path / 'tgt.tif'),
                output=str(tmp_path / 'out.tif'), neg_nodata=65535, feedback=Feedback())
    np.testing.assert_array_equal(read_raster(tmp_path / 'out.tif'),
                                  np.where(2 * target - 15 < 0, 65535, 2 * target - 15))
    with rio.open_raster(tmp_path / 'out.tif') as ds:
        assert ds.GetRasterBand(1).GetNoDataValue() == 65535


def test_report_figures_do_not_change_global_backend(tmp_path):
    matplotlib = pytest.importorskip('matplotlib')
    from PIL import Image
    backend = matplotlib.get_backend()
    build_pair(tmp_path)
    mad = make_mad_raster(tmp_path, (120, 130), bands=4)
    feedback = Feedback()
    radcal.main(mad, img_ref=str(tmp_path / 'ref.tif'), img_tgt=str(tmp_path / 'tgt.tif'),
                output=str(tmp_path / 'out.tif'), graphics=True, feedback=feedback)
    assert matplotlib.get_backend() == backend
    # A fixed-size summary plus one row-per-band figure, each embedded once.
    for name in ('out_report.png', 'out_report_bands.png'):
        assert (tmp_path / name).read_bytes().startswith(b'\x89PNG')
    assert len(feedback.formatted_messages) == 2
    summary, per_band = feedback.formatted_messages
    for html, _ in (summary, per_band):
        assert 'data:image/png;base64,' in html
        assert str(tmp_path) not in html           # the figure, not its location
    assert 'RMSE before -> after' not in summary[1]
    assert 'Applied correction:' not in per_band[1]
    with Image.open(tmp_path / 'out_report.png') as image:
        assert 'RMSE before -> after (before storage conversion):' in image.info['Description']
    with Image.open(tmp_path / 'out_report_bands.png') as image:
        assert 'Applied correction:' in image.info['Description']
        assert 'sample-refit hold-out' in image.info['Description']
    band_lines = [message for message in feedback.messages if message.startswith('band ')]
    assert len(band_lines) == 4
    assert all('RMSE vs reference' in line and 'slope=' in line and 'offset=' in line
               and 'R=' in line for line in band_lines)
    assert not any('report written to:' in line or '0 clipped' in line
                   or 'same-scene sample split' in line for line in feedback.messages)


def test_reported_accuracy_is_exact_over_every_no_change_pixel(tmp_path):
    # RMSE is derived from the fit moments, so it must describe the whole
    # invariant population, not the bounded sample the figures draw. The log
    # rounds to four significant digits; the closed form itself is checked
    # against a brute-force oracle in test_statistics.py.
    values = np.arange(1., 401.).reshape(80, 5)
    target = [values.copy()]
    reference = [2 * values + 3 + ((-1.) ** values)]   # exact fit is impossible
    write_raster(tmp_path / 'ref.tif', reference)
    write_raster(tmp_path / 'tgt.tif', target)
    feedback = Feedback()
    radcal.main(make_mad_raster(tmp_path, values.shape, bands=1),
                img_ref=str(tmp_path / 'ref.tif'), img_tgt=str(tmp_path / 'tgt.tif'),
                output=str(tmp_path / 'out.tif'), feedback=feedback)
    line = next(m for m in feedback.messages if m.startswith('band 1: RMSE'))
    before, after = (float(part) for part in
                     line.split('RMSE vs reference ')[1].split(',')[0]
                         .split(' (')[0].split(' -> '))
    slope, intercept, _ = orthoregress(values.ravel(), reference[0].ravel())
    residual = reference[0].ravel() - (intercept + slope * values.ravel())
    assert after == pytest.approx(np.sqrt(np.mean(residual ** 2)), rel=1e-3)
    assert before == pytest.approx(
        np.sqrt(np.mean((reference[0].ravel() - values.ravel()) ** 2)), rel=1e-3)
    # Every invariant pixel is counted, not the bounded figure sample.
    assert f'no-change pixels used for calibration: {values.size:,}' in feedback.messages
    assert any('before rounding/clipping/masking' in message for message in feedback.messages)
    assert feedback.messages[-1].endswith(f'Output: {tmp_path / "out.tif"}')


def test_report_figures_use_the_caller_directory(tmp_path):
    pytest.importorskip('matplotlib')
    build_pair(tmp_path)
    workspace = tmp_path / 'work'
    workspace.mkdir()
    feedback = Feedback()
    radcal.main(make_mad_raster(tmp_path, (120, 130), bands=2),
                img_ref=str(tmp_path / 'ref.tif'), img_tgt=str(tmp_path / 'tgt.tif'),
                output=str(tmp_path / 'out.tif'), graphics=True, pos=[1, 2],
                report_path=str(workspace / 'calibration_report.png'), feedback=feedback)
    assert (workspace / 'calibration_report.png').read_bytes().startswith(b'\x89PNG')
    assert (workspace / 'calibration_report_bands.png').exists()
    assert not list(tmp_path.glob('out_report*.png'))
    assert len(feedback.formatted_messages) == 2


def test_normalization_keeps_reports_beside_the_published_output(tmp_path):
    # The run workspace is deleted on completion; a report written there would
    # take the log's download link with it.
    pytest.importorskip('matplotlib')
    build_pair(tmp_path)
    norm = make_normalization(tmp_path)
    assert isinstance(norm.feedback, Feedback)
    assert norm.run() == str(tmp_path / 'out.tif')
    assert not list(tmp_path.glob('.arrnorm-*'))
    for name in ('out_report.png', 'out_report_bands.png'):
        assert (tmp_path / name).read_bytes().startswith(b'\x89PNG')
    assert len(norm.feedback.formatted_messages) == 2
    # The log embeds the figures themselves; it does not restate their paths.
    assert all('<img src="data:image/png;base64,' in html
               and str(tmp_path) not in html
               for html, _ in norm.feedback.formatted_messages)


def test_disabled_report_produces_no_figures_or_embedded_messages(tmp_path):
    pytest.importorskip('matplotlib')
    build_pair(tmp_path)
    norm = make_normalization(tmp_path, report=False)
    assert isinstance(norm.feedback, Feedback)
    assert norm.graphics is False
    assert norm.run() == str(tmp_path / 'out.tif')
    assert sorted(p.name for p in tmp_path.iterdir()) == ['out.tif', 'ref.tif', 'tgt.tif']
    assert not norm.feedback.formatted_messages
    # Calibration results stay in the log: they cost nothing and are not figures.
    assert any(message.startswith('band 1: RMSE') for message in norm.feedback.messages)


def test_report_directory_keeps_the_figures_out_of_the_output_folder(tmp_path):
    pytest.importorskip('matplotlib')
    build_pair(tmp_path)
    elsewhere = tmp_path / 'session-temp'
    norm = make_normalization(tmp_path, report_dir=str(elsewhere))
    assert isinstance(norm.feedback, Feedback)
    assert norm.run() == str(tmp_path / 'out.tif')
    assert not list(tmp_path.glob('*_report*.png'))
    for name in ('out_report.png', 'out_report_bands.png'):
        assert (elsewhere / name).read_bytes().startswith(b'\x89PNG')
    # Relocating them changes nothing in the log: the figures are still embedded.
    assert len(norm.feedback.formatted_messages) == 2


def test_normalization_plots_the_imad_convergence_history(tmp_path):
    build_pair(tmp_path)
    norm = make_normalization(tmp_path, max_iters=4, conv_threshold=0.999999)
    norm.run()
    # One row of canonical correlations per completed iteration, four bands wide.
    assert len(norm.rhos) == 4
    assert all(len(row) == 4 and all(0.0 <= value <= 1.0 for value in row)
               for row in norm.rhos)


def test_report_failure_never_discards_a_written_calibration(tmp_path, monkeypatch):
    build_pair(tmp_path)

    def explode(*args, **kwargs):
        raise MemoryError('font cache exhausted')

    monkeypatch.setattr(report, 'build', explode)
    # Inject a renderer failure independently of whether matplotlib is installed.
    monkeypatch.setattr(radcal, 'matplotlib_available', lambda: True)
    norm = make_normalization(tmp_path)
    assert isinstance(norm.feedback, Feedback)
    assert norm.run() == str(tmp_path / 'out.tif')
    assert (tmp_path / 'out.tif').exists()
    assert any('font cache exhausted' in message and message.startswith('ERROR:')
               for message in norm.feedback.messages)
    # The exact accuracy table does not depend on matplotlib, so it survives.
    assert any(message.startswith('band 1: RMSE') for message in norm.feedback.messages)


def test_normalization_skips_graphics_without_matplotlib(tmp_path, monkeypatch):
    build_pair(tmp_path)
    monkeypatch.setattr(radcal, 'matplotlib_available', lambda: False)
    norm = make_normalization(tmp_path)
    assert isinstance(norm.feedback, Feedback)
    norm.run()
    assert norm.graphics is False
    assert not norm.feedback.formatted_messages
    assert not list(tmp_path.glob('*_report*.png'))
    # The accuracy table is exact and free, so it is still reported.
    assert any(message.startswith('band 1: RMSE') for message in norm.feedback.messages)


@pytest.mark.parametrize('broken_formatted', [False, True])
def test_plain_or_failed_formatted_feedback_keeps_results_and_one_link_per_report(
        tmp_path, monkeypatch, broken_formatted):
    pytest.importorskip('matplotlib')
    build_pair(tmp_path)
    norm = make_normalization(tmp_path)
    feedback = Feedback()

    def fail_formatting(*args):
        raise RuntimeError('formatted messages unavailable')

    monkeypatch.setattr(feedback, 'pushFormattedMessage', fail_formatting if broken_formatted else None)
    norm.feedback = feedback
    assert norm.run() == str(tmp_path / 'out.tif')
    for name, title in (('out_report.png', report._SUMMARY_TITLE),
                        ('out_report_bands.png', 'Radiometric calibration report - per band')):
        # One plain-text heading per figure, and no path restated in the log.
        assert sum(message == title for message in feedback.messages) == 1
        assert (tmp_path / name).exists()
    assert not any(str(tmp_path / 'out_report') in message for message in feedback.messages)
    assert sum('RMSE vs reference' in message for message in feedback.messages) == 4
    assert not any('sample-refit hold-out' in message for message in feedback.messages)
    assert sum('Cannot embed the calibration report' in message
               for message in feedback.messages) == (2 if broken_formatted else 0)


@pytest.mark.parametrize('full_scene', [False, True])
def test_report_ranges_cover_all_fitted_and_applied_pixels(tmp_path, monkeypatch, full_scene):
    x = np.linspace(10., 100., 1000).reshape(100, 10)
    x[-1, -1] = 1000  # fitted extreme deliberately absent from a tiny sample
    y = 1.2 * x + 3
    x[0, 0] = 2000    # valid target beyond reference overlap, still transformed
    y[0, 0] = -9999
    write_raster(tmp_path / 'ref.tif', [y])
    write_raster(tmp_path / 'tgt.tif', [x])
    captured = {}
    monkeypatch.setattr(report, 'emit', lambda *args, **kwargs: captured.update(kwargs))
    sampler = report.UniformSample
    monkeypatch.setattr(report, 'UniformSample', lambda: sampler(size=16))
    full_target = None
    if full_scene:
        full_target = str(tmp_path / 'full.tif')
        write_raster(full_target, [np.vstack((x, np.full_like(x, 4000)))])
    radcal.main(make_mad_raster(tmp_path, x.shape, bands=1),
                img_ref=str(tmp_path / 'ref.tif'), img_tgt=str(tmp_path / 'tgt.tif'),
                output=str(tmp_path / 'out.tif'), img_target=full_target,
                graphics=True, nodata_ref=-9999,
                block_rows=7, feedback=Feedback())
    assert captured['sample'].target.max() < 1000
    assert captured['fit_ranges'].bounds(0)[1] == 1000
    assert captured['ranges'].bounds(0)[1] == (4000 if full_scene else 2000)


def test_integer_report_explicitly_describes_model_not_rounded_output(tmp_path):
    rng = np.random.default_rng(19)
    x = rng.integers(10, 500, (80, 80))
    y = np.rint(.73 * x + 8)
    write_raster(tmp_path / 'ref.tif', [y], dtype=gdal.GDT_UInt16)
    write_raster(tmp_path / 'tgt.tif', [x], dtype=gdal.GDT_UInt16)
    feedback = Feedback()
    radcal.main(make_mad_raster(tmp_path, x.shape, bands=1),
                img_ref=str(tmp_path / 'ref.tif'), img_tgt=str(tmp_path / 'tgt.tif'),
                output=str(tmp_path / 'out.tif'), feedback=feedback)
    slope, intercept, _ = orthoregress(x.ravel(), y.ravel())
    model_rmse = np.sqrt(np.mean((intercept + slope * x - y) ** 2))
    stored_rmse = np.sqrt(np.mean((read_raster(tmp_path / 'out.tif').astype(float) - y) ** 2))
    assert abs(model_rmse - stored_rmse) > .2
    line = next(message for message in feedback.messages if message.startswith('band 1: RMSE'))
    assert f'-> {model_rmse:.4g}' in line
    assert any('before rounding/clipping/masking' in message for message in feedback.messages)
    assert any('output UInt16' in message for message in feedback.messages)


def test_report_counts_clipping_and_negative_masking_separately(tmp_path, monkeypatch):
    x = np.arange(1., 201.).reshape(20, 10)
    y = 2 * x - 15
    write_raster(tmp_path / 'ref.tif', [y])
    write_raster(tmp_path / 'tgt.tif', [x])
    mad = tmp_path / 'mad.tif'
    write_raster(mad, [np.zeros_like(x), np.where((x >= 10) & (x <= 100), 0, 1000)])
    captured = {}
    monkeypatch.setattr(report, 'emit', lambda *args, **kwargs: captured.update(kwargs))
    radcal.main(str(mad), img_ref=str(tmp_path / 'ref.tif'), img_tgt=str(tmp_path / 'tgt.tif'),
                output=str(tmp_path / 'out.tif'), out_dtype=gdal.GDT_Byte, neg_nodata=255,
                feedback=Feedback())
    assert captured['context']['conversion'] == [{'valid': 200, 'clipped': 65, 'negative_nodata': 7}]
    np.testing.assert_array_equal(read_raster(tmp_path / 'out.tif'),
                                  np.where(y < 0, 255, np.clip(y, 0, 255)))


@pytest.mark.parametrize('side', ['reference', 'target'])
@pytest.mark.parametrize('other_dtype', [gdal.GDT_Float32, gdal.GDT_Float64])
def test_explicit_fractional_nodata_does_not_bias_known_fit(tmp_path, side, other_dtype):
    values = np.arange(10., 110.).reshape(20, 5)
    target = [values.copy(), values[::-1].copy()]
    expected = np.array([2 * band + 3 for band in target])
    reference = expected.copy()
    if side == 'reference':
        reference[0, :2] = 0.1
    else:
        target[0][:2] = 0.1
    write_raster(tmp_path / 'ref.tif', reference,
                 dtype=gdal.GDT_Float32 if side == 'reference' else other_dtype)
    write_raster(tmp_path / 'tgt.tif', target,
                 dtype=gdal.GDT_Float32 if side == 'target' else other_dtype)
    destination = tmp_path / 'out.tif'
    radcal.main(make_mad_raster(tmp_path, values.shape),
                img_ref=str(tmp_path / 'ref.tif'), img_tgt=str(tmp_path / 'tgt.tif'),
                output=str(destination), nodata_ref=0.1 if side == 'reference' else None,
                nodata_tgt=0.1 if side == 'target' else None, feedback=Feedback())
    actual = read_raster(destination)
    np.testing.assert_allclose(actual[:, 2:], expected[:, 2:], atol=1e-6)
    if side == 'target':
        with rio.open_raster(destination) as ds:
            sentinel = ds.GetRasterBand(1).GetNoDataValue()
        assert np.all(actual[:, :2] == sentinel)


def test_fractional_sentinel_matches_each_vrt_bands_storage_precision(tmp_path):
    values = np.arange(10., 110.).reshape(20, 5)
    target = [values + .25, values[::-1] + .5]
    expected = np.array([2 * band + 3 for band in target])
    target[0][0] = target[1][1] = .1
    for i, dtype in enumerate((gdal.GDT_Float32, gdal.GDT_Float64)):
        write_raster(tmp_path / f'band{i}.tif', [target[i]], dtype=dtype)
    with rio.Raster(gdal.BuildVRT(str(tmp_path / 'tgt.vrt'),
                                  [str(tmp_path / f'band{i}.tif') for i in range(2)],
                                  separate=True), writable=True):
        pass
    with rio.open_raster(tmp_path / 'tgt.vrt') as vrt:
        assert vrt.GetRasterBand(1).DataType == gdal.GDT_Float32
        assert vrt.GetRasterBand(2).DataType == gdal.GDT_Float64
    write_raster(tmp_path / 'ref.tif', expected, dtype=gdal.GDT_Float64)
    destination = tmp_path / 'out.tif'
    radcal.main(make_mad_raster(tmp_path, values.shape),
                img_ref=str(tmp_path / 'ref.tif'), img_tgt=str(tmp_path / 'tgt.vrt'),
                output=str(destination), nodata_tgt=.1, feedback=Feedback())
    actual = read_raster(destination)
    assert np.all(actual[:, :2] == .1)
    np.testing.assert_allclose(actual[:, 2:], expected[:, 2:], atol=1e-9)


@pytest.mark.parametrize('out_dtype', [gdal.GDT_Float64, gdal.GDT_UInt16])
def test_nonfinite_calibration_is_not_hidden_by_clipping_or_nodata(tmp_path, monkeypatch, out_dtype):
    values = np.arange(1., 21.).reshape(4, 5)
    write_raster(tmp_path / 'ref.tif', [2 * values + 3])
    write_raster(tmp_path / 'tgt.tif', [values])
    monkeypatch.setattr(radcal.OrthogonalFit, 'coefficients', lambda self: (-np.inf, 0., 0.))
    with pytest.raises(QgsProcessingException, match='Non-finite calibration output'):
        radcal.main(make_mad_raster(tmp_path, values.shape, bands=1),
                    img_ref=str(tmp_path / 'ref.tif'), img_tgt=str(tmp_path / 'tgt.tif'),
                    output=str(tmp_path / 'out.tif'), out_dtype=out_dtype,
                    neg_nodata=np.nan if out_dtype == gdal.GDT_Float64 else 0,
                    feedback=Feedback())
