"""Report semantics: sampling, validation and honest numerical presentation."""
import numpy as np
import pytest
from ArrNorm.core import report


def test_holdout_shuffles_periodic_scan_order_and_reports_effect_sizes():
    x = np.arange(300, dtype=float)
    y = 1.2 * x + 3 + np.where(x % 3 == 0, 3., 0.)
    result = report.holdout(x, y)
    assert result == report.holdout(x, y)
    assert result['count'] == 100
    assert 1.2 < result['rmse'] < 1.7
    assert abs(result['bias']) < .5
    assert 'variance_ratio' in result
    assert not any(key.endswith('_p') for key in result)


def test_small_holdout_is_unavailable():
    assert report.holdout(np.arange(20.), np.arange(20.)) is None


def test_sample_pairs_and_holdout_are_block_size_independent():
    x = np.arange(1000.)[:, None]
    y = 1.2 * x + np.sin(x)
    samples = []
    for block in (7, 101, 1000):
        sample = report.UniformSample(size=100)
        for start in range(0, len(x), block):
            sample.update(x[start:start + block], y[start:start + block])
        samples.append(sample.pairs())
    for target, reference in samples[1:]:
        np.testing.assert_array_equal(target, samples[0][0])
        np.testing.assert_array_equal(reference, samples[0][1])


def test_unresolved_rmse_has_no_fold_or_perfection_claim():
    statistic = {'rmse_before': 100., 'rmse_after': np.nan, 'rmse_after_bound': .001}
    assert report._fold(statistic) is None
    assert '<=' in report._rmse(statistic, 'after')
    assert 'exactly' not in report._improvement({'rmse_before': 100., 'rmse_after': 0.})


def test_report_refuses_a_direct_destination_alias(tmp_path):
    pytest.importorskip('matplotlib')
    source = tmp_path / 'input.png'
    source.write_bytes(b'protected input')
    with pytest.raises(ValueError, match='overwrite'):
        report.build(str(source), [1], [(1., 0., 1.)], [], protected_paths=[str(source)])
    assert source.read_bytes() == b'protected input'


def test_undefined_and_zero_rmse_are_not_plotted_as_tiny_positive_errors():
    pytest.importorskip('matplotlib')
    from matplotlib.figure import Figure

    axis = Figure().subplots()
    report._agreement(axis, ['b1', 'b2'], [
        {'rmse_before': 10., 'rmse_after': np.nan},
        {'rmse_before': 20., 'rmse_after': 0.},
    ])
    positions = [offset for collection in axis.collections for offset in collection.get_offsets()]
    assert sorted(float(point[0]) for point in positions) == [10., 20.]
    assert any('After: n/a' in text.get_text() for text in axis.texts)
    assert any('After: 0 (computed)' in text.get_text() for text in axis.texts)


def test_agreement_shows_values_changes_and_distinguishable_overlapping_markers():
    pytest.importorskip('matplotlib')
    from matplotlib.figure import Figure

    axis = Figure().subplots()
    report._agreement(axis, ['Band 1', 'Band 2', 'Band 3', 'Band 4', 'Band 5'], [
        {'rmse_before': 8047., 'rmse_after': 15.88},
        {'rmse_before': 2., 'rmse_after': 5.},
        {'rmse_before': 4., 'rmse_after': 4.},
        {'rmse_before': 100., 'rmse_after': np.nan, 'rmse_after_bound': .001},
        {'rmse_before': 10., 'rmse_after': 10.01},
    ])
    text = {item.get_text(): item for item in axis.texts}
    improved = text['(507× lower RMSE)']
    worse = text['(2.5× higher RMSE)']
    assert worse.get_color() != improved.get_color()
    assert improved.xy == pytest.approx((np.sqrt(8047. * 15.88), 4))
    assert worse.xy == pytest.approx((np.sqrt(2. * 5.), 3))
    assert text['8,047'].xy == (8047., 4)
    assert text['15.9'].xy == (15.88, 4)
    assert '(Unchanged RMSE)' in text
    assert text['≤ 0.001'].xy == (.001, 1)
    assert '(Slightly higher RMSE)' in text
    assert text['10.01'].xy == (10.01, 0)
    coincident = [item for item in axis.texts if item.get_text() == '4']
    assert len(coincident) == 2
    assert {item.get_ha() for item in coincident} == {'left', 'right'}
    before, after = axis.get_legend_handles_labels()[0]
    assert before.get_facecolors().size == 0
    assert after.get_facecolors()[0, 3] == 1
    assert before.get_sizes()[0] > after.get_sizes()[0]
    np.testing.assert_array_equal(before.get_offsets()[2], after.get_offsets()[2])
    assert axis.get_xscale() == 'log'
    formatter = axis.xaxis.get_major_formatter()
    assert [formatter(value) for value in (.1, 1., 10., 1000.)] == ['0.1', '1', '10', '1,000']
    assert float(formatter(1e-8)) > 0


def test_agreement_does_not_round_a_small_change_to_one_times_or_unchanged():
    text, worse = report._agreement_change({'rmse_before': 1., 'rmse_after': 1.001})
    assert text == 'Slightly higher RMSE'
    assert worse is True
    text, worse = report._agreement_change({'rmse_before': 0., 'rmse_after': 3.})
    assert text == 'Higher RMSE'
    assert worse is True


def test_agreement_with_only_zero_or_unavailable_values_has_no_fake_log_positions():
    pytest.importorskip('matplotlib')
    from matplotlib.figure import Figure

    axis = Figure().subplots()
    report._agreement(axis, ['Band 1', 'Band 2'], [
        {'rmse_before': 0., 'rmse_after': 0.},
        {'rmse_before': np.nan, 'rmse_after': np.nan},
    ])
    assert axis.get_xscale() == 'log'
    assert not len(axis.get_xticks())
    assert not any(len(collection.get_offsets()) for collection in axis.collections)
    assert not axis.lines
    assert any('Before: n/a' in item.get_text() for item in axis.texts)
    assert any('After: n/a' in item.get_text() for item in axis.texts)


@pytest.mark.parametrize('case', ['overlap', 'positive_gap', 'negative_gap', 'constant', 'outlier'])
def test_distributions_share_bins_and_preserve_every_sample(tmp_path, monkeypatch, case):
    pytest.importorskip('matplotlib')
    from matplotlib.axes import Axes
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    x = np.linspace(10., 100., 64)
    y = .9 * x + 3 + np.sin(x)
    normalized = .9 * x + 3
    if case == 'positive_gap':
        x += 10000
    elif case == 'negative_gap':
        x -= 10000
    elif case == 'constant':
        x = y = normalized = np.full(64, 3.)
    elif case == 'outlier':
        x[-1] = 1e6
    figure = Figure(figsize=(5, 3))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    captured = []
    original = Axes.hist

    def capture(self, values, *args, **kwargs):
        result = original(self, values, *args, **kwargs)
        captured.append((self, np.asarray(values).copy(), result[0], result[1]))
        return result

    monkeypatch.setattr(Axes, 'hist', capture)
    report._distributions(axis, x, y, normalized)
    overview = [item for item in captured if item[0] is axis]
    assert len(overview) == 3
    for item, expected in zip(overview, (y, x, normalized)):
        np.testing.assert_array_equal(item[1], expected)
        np.testing.assert_array_equal(item[3], overview[0][3])
        assert item[2].sum() == len(expected)
        assert np.isfinite(item[3]).all()
        assert np.all(np.diff(item[3]) > 0)
    assert axis.get_legend_handles_labels()[1] == [
        'Reference', 'Target before', 'Target after (affine)',
    ]
    assert axis.get_xlim()[0] <= min(x.min(), y.min(), normalized.min())
    assert axis.get_xlim()[1] >= max(x.max(), y.max(), normalized.max())
    details = [item for item in captured if item[0] is not axis]
    if case in ('positive_gap', 'negative_gap'):
        assert len(details) == 2
        inset = axis.child_axes[0]
        assert inset.get_xlabel() == inset.get_ylabel() == ''
        assert len(inset.get_xticks()) > 0 and len(inset.get_yticks()) > 0
        for item, expected in zip(details, (y, normalized)):
            np.testing.assert_array_equal(item[1], expected)
            np.testing.assert_array_equal(item[3], details[0][3])
            assert item[2].sum() == len(expected)
        assert np.diff(details[0][3]).max() < np.diff(overview[0][3]).min() / 10
    else:
        assert not details
    # Exercise actual canvas rendering too, including constant-value bins/insets.
    figure.savefig(tmp_path / f'{case}.png')


def test_distribution_caption_distinguishes_full_fit_and_holdout():
    text, _ = report._bands_caption([1, 2], [(1., 3., .99), (2., 1., .98)], [
        {'count': 10, 'train_count': 20, 'rmse': .5, 'bias': -.1, 'variance_ratio': .97},
        None,
    ])
    assert 'same sampled pixels' in text
    assert 'full-fit affine model before clipping, masking and storage conversion' in text
    assert 'detail insets re-bin reference/after only' in text
    assert 'train=20, test=10, RMSE 0.5, bias -0.1, variance ratio 0.97' in text
    assert 'b2: unavailable (small or degenerate split)' in text


@pytest.mark.parametrize('clipped,negative', [(0, 0), (12, 0), (0, 5), (12, 5)])
def test_footer_keeps_inputs_scope_and_only_actual_output_adjustments(clipped, negative):
    pytest.importorskip('matplotlib')
    from matplotlib.figure import Figure

    context = {'reference': '/data/reference.tif', 'target': '/data/target.tif',
               'output': '/data/output.tif', 'threshold': .95, 'dtype': 'UInt16',
               'sample_count': 120,
               'conversion': [{'clipped': clipped, 'negative_nodata': negative, 'valid': 500}]}
    figure = Figure()
    report._footer(figure, context)
    text = figure.texts[0].get_text()
    assert text.splitlines()[0] == 'Reference: reference.tif | Target: target.tif'
    assert 'before output rounding, clipping and masking' in text
    assert 'output.tif' not in text and 'UInt16' not in text
    assert 'sample n=' not in text and 'probability' not in text and 'sample split' not in text
    assert ('Output adjustments:' in text) == bool(clipped or negative)
    # The valid count is the denominator that makes an adjustment count mean something.
    assert ('12 clipped' in text) == bool(clipped)
    assert ('5 negative set to NoData' in text) == bool(negative)
    assert ('(of 500 values).' in text) == bool(clipped or negative)
    assert len(text.splitlines()) == (3 if clipped or negative else 2)
    # Removing figure clutter must not discard the detailed metadata record.
    metadata_context = ' '.join(report._context_lines(context))
    assert 'output.tif' in metadata_context and 'output UInt16' in metadata_context
    assert 'no-change probability > 0.95' in metadata_context


@pytest.mark.parametrize('shift', [0., 10000.])
def test_per_band_footer_mentions_inset_binning_only_when_an_inset_is_present(shift):
    pytest.importorskip('matplotlib')
    from matplotlib.figure import Figure

    x = np.linspace(10., 100., 60) + shift
    sample = report.UniformSample()
    sample.update(x[:, None], (.8 * (x - shift) + 3)[:, None])
    figure = Figure(figsize=(10, 4))
    report._bands(figure, [1], [(.8, 3 - .8 * shift, 1.)], sample)
    assert bool(figure.axes[1].child_axes) == bool(shift)
    assert ('Insets use finer bins' in figure.texts[-1].get_text()) == bool(shift)


def test_fit_axes_follow_the_sample_and_png_metadata_keeps_complete_ranges(tmp_path, monkeypatch):
    pytest.importorskip('matplotlib')
    from ArrNorm.core.auxil.auxil import OrthogonalFit
    from PIL import Image

    x = np.linspace(10000., 10020., 120)
    y = .5 * x - 4900 + np.sin(x)
    sample = report.UniformSample(size=120)
    sample.update(x[:, None], y[:, None])
    training = np.concatenate((x, [9000., 30000.]))
    fit = OrthogonalFit()
    fit.update(training, .5 * training - 4900 + np.sin(training))
    support, applied = report.ValueRange(1), report.ValueRange(1)
    support.update(training[:, None])
    applied.update(np.array([[0.], [60000.]]))
    captured = []
    original = report._bands

    def capture(figure, *args, **kwargs):
        validation = original(figure, *args, **kwargs)
        captured.append(figure.axes[0])
        return validation

    monkeypatch.setattr(report, '_bands', capture)
    items = report.build(str(tmp_path / 'report.png'), [1], [fit.coefficients()],
                         [fit.statistics()], sample=sample, ranges=applied, fit_ranges=support)
    scatter = captured[0]
    density = scatter.collections[0]
    assert density.get_array().sum() == len(x)
    assert density.colorbar is not None
    assert density.colorbar.ax.get_ylabel() == 'Sampled pairs / bin (log)'
    ticks = density.colorbar.get_ticks()
    np.testing.assert_array_equal(ticks, np.round(ticks))
    assert ticks.min() >= density.get_array().min()
    assert ticks.max() <= density.get_array().max()
    low, high = scatter.get_xlim()
    assert low < x.min() and high > x.max()
    assert high - low < 1.2 * np.ptp(x)
    np.testing.assert_array_equal(scatter.lines[0].get_xdata(), [x.min(), x.max()])
    assert not scatter.patches  # no misleading extrapolation shading in a sample-only view
    with Image.open(items[1][0]) as image:
        description = image.info['Description']
    assert 'fitting pixels [9000, 3e+04]' in description
    assert 'application pixels [0, 6e+04]' in description
    assert 'not limited to the plotting sample' in description
    assert 'fitting pixels' not in items[1][2] and 'fitting pixels' not in items[1][3]


@pytest.mark.parametrize('count', [1, 4])
def test_uniform_hexbin_counts_use_an_accurate_single_colour_legend(count):
    pytest.importorskip('matplotlib')
    from matplotlib.figure import Figure

    figure = Figure()
    axis = figure.subplots()
    x = np.repeat([0., 5., 10.], count)
    bins = axis.hexbin(x, .8 * x + 1, gridsize=42, bins='log', cmap='viridis', mincnt=1)
    np.testing.assert_array_equal(bins.get_array(), [count] * 3)
    report._fit_density_legend(figure, axis, bins)
    assert bins.colorbar is None
    legend = axis.get_legend()
    assert legend.get_texts()[0].get_text() == f'{count} sampled {"pair" if count == 1 else "pairs"} / bin'
    np.testing.assert_allclose(legend.get_patches()[0].get_facecolor(),
                               np.asarray(bins.to_rgba(count)).ravel())


def test_text_results_only_log_nonzero_output_adjustments():
    from ArrNorm.core.auxil.auxil import OrthogonalFit
    from ArrNorm.tests.helpers import Feedback

    fit = OrthogonalFit()
    x = np.arange(10., 50.)
    fit.update(x, .8 * x + 3 + np.sin(x))
    feedback = Feedback()
    report.emit(feedback, feedback.pushInfo, [1, 2], [fit.coefficients()] * 2, [fit] * 2,
                context={'dtype': 'UInt16', 'conversion': [
                    {'valid': 100, 'clipped': 0, 'negative_nodata': 0},
                    {'valid': 100, 'clipped': 4, 'negative_nodata': 3},
                ]})
    adjustments = [line for line in feedback.messages if 'output adjustments:' in line]
    assert adjustments == ['band 2: output adjustments: 4 clipped; 3 negative set to NoData (of 100 values).']
    assert sum('RMSE vs reference' in line for line in feedback.messages) == 2
    assert sum('before rounding/clipping/masking' in line for line in feedback.messages) == 1


@pytest.mark.parametrize('cols,rows', [(130, 120), (536, 349), (240, 80), (80, 240)])
def test_summary_preserves_map_aspect_and_uses_per_band_column_padding(cols, rows):
    pytest.importorskip('matplotlib')
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    density = report.DensityGrid(cols, rows)
    valid = np.ones((rows, cols), dtype=bool)
    selected = np.random.default_rng(13).random((rows, cols)) < .08
    density.update(0, selected.ravel(), valid.ravel(), cols)
    statistics = [{'rmse_before': 20. + band, 'rmse_after': .5, 'slope': .8,
                   'intercept': -2., 'variance_ratio': .99} for band in range(4)]
    figure = Figure(figsize=(10, 7.4), dpi=200)
    canvas = FigureCanvasAgg(figure)
    report._summary(figure, [1, 2, 3, 4], statistics, density,
                    [[.7, .8, .9, .99], [.75, .85, .95, .999]])
    canvas.draw()
    agreement, correction, coverage, convergence = figure.axes[:4]
    # The map must stay proportional and centered, not stretched to share edges.
    assert coverage.bbox.width / coverage.bbox.height == pytest.approx(
        density.shape[1] / density.shape[0])
    assert np.mean(coverage.bbox.intervalx) == pytest.approx(np.mean(agreement.bbox.intervalx))
    renderer = canvas.get_renderer()
    bar = coverage.images[0].colorbar.ax.get_tightbbox(renderer)
    assert bar.x0 >= coverage.bbox.x1
    assert bar.x1 < convergence.get_tightbbox(renderer).x0

    # Compare the actual whitespace between decorated columns with the per-band
    # figure: ticks, titles, colour bars and labels must all have breathing room.
    sample = report.UniformSample()
    target = np.column_stack([np.linspace(10., 110., 120)] * 2)
    sample.update(target, .8 * target - 2)
    per_band = Figure(figsize=(10, 6.9), dpi=200)
    per_band_canvas = FigureCanvasAgg(per_band)
    report._bands(per_band, [1, 2], [(.8, -2., 1.)] * 2, sample)
    per_band_canvas.draw()
    band_renderer = per_band_canvas.get_renderer()
    band_gap = min(per_band.axes[i + 1].get_tightbbox(band_renderer).x0
                   - per_band.axes[i].get_tightbbox(band_renderer).x1 for i in (0, 2))
    summary_gap = min(right.get_tightbbox(renderer).x0 - left.get_tightbbox(renderer).x1
                      for left, right in ((agreement, correction), (coverage, convergence)))
    assert band_gap > 0
    assert summary_gap >= band_gap - 1  # allow one rendered pixel of layout rounding


@pytest.mark.parametrize('pairs', [4, 8])
def test_summary_labels_canonical_pairs_and_separates_iteration_subtitle(pairs):
    pytest.importorskip('matplotlib')
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    rhos = np.linspace(.5, .9, pairs)[None, :] + np.arange(3)[:, None] * .02
    statistics = [{'rmse_before': 20., 'rmse_after': .5, 'slope': .8,
                   'intercept': -2., 'variance_ratio': .99} for _ in range(pairs)]
    figure = Figure(figsize=(10, 7.4), dpi=200)
    canvas = FigureCanvasAgg(figure)
    report._summary(figure, list(range(1, pairs + 1)), statistics, None, rhos,
                    {'imad': {'termination': 'iteration limit', 'selected_iteration': 2,
                              'final_delta': .02, 'delta_threshold': .001}})
    canvas.draw()
    convergence = figure.axes[3]
    assert [text.get_text() for text in convergence.get_legend().get_texts()] == [
        f'Canonical pair {index + 1}' for index in range(pairs)]
    for index, line in enumerate(convergence.lines[:pairs]):
        np.testing.assert_array_equal(line.get_ydata(), rhos[:, index])
    assert '\n' not in convergence.get_title()
    assert 'iteration limit reached' in convergence.get_title()
    subtitle = next(text for text in convergence.texts if '3 iterations' in text.get_text())
    assert 'Max change: 0.02' in subtitle.get_text()
    assert 'Threshold: 0.001' in subtitle.get_text()
    assert subtitle.get_fontsize() < convergence.title.get_fontsize()
    assert 'selected solution: 2' in convergence.get_xlabel()


def test_no_change_colour_scale_has_zero_baseline_and_whole_pixel_counts():
    pytest.importorskip('matplotlib')
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    density = report.DensityGrid(4, 4)
    density.update(0, np.ones(16, bool), np.ones(16, bool), 4)
    statistics = [{'rmse_before': 20., 'rmse_after': .5, 'slope': .8,
                   'intercept': -2., 'variance_ratio': .99}]
    figure = Figure(figsize=(10, 7.4), dpi=200)
    canvas = FigureCanvasAgg(figure)
    report._summary(figure, [1], statistics, density, None)
    canvas.draw()
    coverage = figure.axes[2]
    bar = coverage.images[0].colorbar
    assert bar.mappable.get_clim() == (0., 1.)
    np.testing.assert_array_equal(bar.get_ticks(), np.round(bar.get_ticks()))
    assert bar.ax.get_ylabel() == 'Selected pixels / cell'
    subtitle = next(text for text in coverage.texts if '16 selected' in text.get_text())
    assert '100.0% of valid overlap' in subtitle.get_text()
    assert subtitle.get_fontsize() < coverage.title.get_fontsize()
