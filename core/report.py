"""Calibration accuracy table and report figures (GPLv2+).

Two bounded figures per run, embedded in the Processing log and written where
the caller asks — beside the calibrated raster, or in a folder it chose. The
log shows the figures themselves, not their paths, so relocating them costs the
reader nothing: a fixed-size summary that answers "did the normalization work"
independently of band count, and a per-band figure with one row per band.

Processing feedback stays concise; PNG Description metadata retains the detailed
diagnostic descriptions and run context.

Affine-model agreement covers every selected invariant pixel, derived from the
streaming regression moments at no extra I/O, with numerical-resolution limits.
Scatter/distribution panels and descriptive hold-outs use a bounded uniform sample,
never the first ones in scan order. Figures use an Agg canvas directly, never
change matplotlib's global backend, and never fail a run whose calibrated
raster is already written.
"""
import base64
import os
from html import escape
from pathlib import Path

import numpy as np

from .artifacts import validate_report_destination
from .auxil.auxil import OrthogonalFit

WIDTH = 1000           # display width in the Processing log, in logical pixels
SAMPLE_SIZE = 30000    # uniform sample kept for the scatters and hold-out refits
_SEED = 20260916       # fixed: the same scene reports the same sample twice
_LAYOUT_DPI = 100       # logical pixels per inch; preserves text/layout sizing
_DPI = 200              # render 3x the display resolution for sharp PNGs and HiDPI
_MAX_PIXEL_HEIGHT = 12600
_COLUMN_PAD = 1.08      # shared tight-layout padding, matching the per-band figure
_SUMMARY_TITLE = 'Radiometric normalization — summary'
_TOKEN = '__ARRNORM_PNG__'   # placeholder for the base64 payload in the caption
_VALIDATION_SCOPE = ('IR-MAD-selected pixels; the same-scene sample split is descriptive, '
                     'not independent scene validation.')


class UniformSample:
    """Bounded uniform sample across the entire invariant-pixel population.

    Every candidate row draws one key from a seeded generator and the k smallest
    keys are retained, which is a uniform sample without replacement over the
    whole scene. Keeping the first k rows instead would describe only a strip
    across the top of the image, and every sampled statistic would silently
    inherit that spatial bias. The seeded realization depends on pixel order;
    it is reproducible across block sizes for the same traversal.
    """

    def __init__(self, size=SAMPLE_SIZE, seed=_SEED):
        self.size = int(size)
        self._rng = np.random.default_rng(seed)
        self._keys = np.empty(0)
        self.target = None
        self.reference = None

    def update(self, target, reference):
        if self.size <= 0 or len(target) == 0:
            return
        keys = self._rng.random(len(target))
        kept, previous, matching = self._keys, self.target, self.reference
        if previous is not None and matching is not None:
            keys = np.concatenate((kept, keys))
            target = np.concatenate((previous, target))
            reference = np.concatenate((matching, reference))
        if len(keys) > self.size:
            keep = np.argpartition(keys, self.size - 1)[:self.size]
            keys, target, reference = keys[keep], target[keep], reference[keep]
        self._keys, self.target, self.reference = keys, target, reference

    def __len__(self):
        return 0 if self.target is None else len(self.target)

    def pairs(self):
        """Canonical key order makes downstream splitting independent of blocks."""
        if self.target is None or self.reference is None:
            return np.empty((0, 0)), np.empty((0, 0))
        order = np.argsort(self._keys)
        return self.target[order], self.reference[order]


class DensityGrid:
    """Coarse invariant-pixel counts accumulated during the fitting pass.

    A global count cannot show whether the fit is supported across the scene or
    only in one corner, which is the classic silent failure. This reduces the
    masks the calibration pass already computes, so it costs no extra reads.
    """

    def __init__(self, cols, rows, width=96):
        self.step = max(1, int(np.ceil(max(cols, rows) / max(1, width))))
        self.shape = (int(np.ceil(rows / self.step)), int(np.ceil(cols / self.step)))
        self.selected = np.zeros(self.shape)
        self.valid = np.zeros(self.shape)
        self.total_selected = 0
        self.total_valid = 0

    def update(self, offset, selected, valid, cols):
        self.total_selected += int(selected.sum())
        self.total_valid += int(valid.sum())
        rows = len(selected) // cols
        index = (offset + np.arange(rows)) // self.step
        np.add.at(self.selected, index, self._reduce(selected, rows, cols))
        np.add.at(self.valid, index, self._reduce(valid, rows, cols))

    def _reduce(self, mask, rows, cols):
        padded = np.pad(mask.reshape(rows, cols), ((0, 0), (0, (-cols) % self.step)))
        return padded.reshape(rows, -1, self.step).sum(axis=2)

    def coverage(self):
        """Percentage of the valid overlap that the calibration actually used."""
        return 100.0 * self.total_selected / self.total_valid if self.total_valid else float('nan')


class ValueRange:
    """Per-band range of the valid target pixels, for extrapolation checks.

    The fit is estimated on invariant pixels but applied to the whole scene;
    comparing the two ranges is the only way to see how far the calibration is
    extrapolated.
    """

    def __init__(self, bands):
        self.minimum = np.full(bands, np.inf)
        self.maximum = np.full(bands, -np.inf)

    def update(self, tile):
        if len(tile):
            self.minimum = np.minimum(self.minimum, tile.min(axis=0))
            self.maximum = np.maximum(self.maximum, tile.max(axis=0))

    def bounds(self, index):
        low, high = self.minimum[index], self.maximum[index]
        if not (np.isfinite(low) and np.isfinite(high)) or high <= low:
            return None
        return float(low), float(high)


def holdout(target, reference):
    """Descriptive validation of a sample refit on a shuffled held-out third.

    The published full-population fit is unchanged. Selection by IR-MAD precedes
    this split, and pixels may be spatially dependent: this is neither independent
    scene validation nor a significance test. A paired variance ratio alone does
    not have the independent-population F distribution.
    """
    index = np.random.default_rng(_SEED + 1).permutation(len(target))
    test, train = index[:len(index) // 3], index[len(index) // 3:]
    if len(train) < 8 or len(test) < 8:
        return None
    fit = OrthogonalFit()
    fit.update(target[train], reference[train])
    try:
        slope, intercept, _ = fit.coefficients()
    except ValueError:
        return None
    normalized = intercept + slope * target[test]
    observed = reference[test]
    variance = np.var(observed, ddof=1)
    ratio = float(np.var(normalized, ddof=1) / variance) if variance > 0 else float('nan')
    return {'count': len(test), 'train_count': len(train),
            'rmse': float(np.sqrt(np.mean((observed - normalized) ** 2))),
            'bias': float(np.mean(normalized - observed)), 'variance_ratio': ratio}


def _number(value, digits=4):
    return 'n/a' if value is None or not np.isfinite(value) else f'{value:.{digits}g}'


def _factor(value):
    return f'{value:.4g}'


def _rmse(statistic, when, digits=4):
    bound = statistic.get(f'rmse_{when}_bound', np.nan)
    if np.isfinite(bound):
        return f'<= {_number(bound, digits)} (round-off estimate)'
    return _number(statistic[f'rmse_{when}'], digits)


def _fold(statistic):
    """RMSE change as (text, ratio) of before over after, or None if undefined.

    Large reductions can round to -100% when formatted as integer percentages.
    A fold ratio retains the distinction without implying a zero residual.
    """
    before, after = statistic['rmse_before'], statistic['rmse_after']
    if not (np.isfinite(before) and np.isfinite(after)) or before <= 0 or after < 0:
        return None
    if after == 0:
        return 'computed RMSE 0', np.inf
    ratio = before / after
    if not np.isfinite(ratio) or ratio <= 0:
        return None
    if ratio > 1:
        return f'{_factor(ratio)}x lower', ratio
    if ratio < 1:
        return f'{_factor(1 / ratio)}x higher', ratio
    return 'unchanged', ratio


def _improvement(statistic):
    fold = _fold(statistic)
    return f' ({fold[0]})' if fold else ''


def _accuracy_lines(bands, statistics):
    return [f'band {band}: RMSE vs reference {_rmse(statistic, "before")} -> '
            f'{_rmse(statistic, "after")}{_improvement(statistic)}, '
            f'slope={_number(statistic["slope"], 6)}, offset={_number(statistic["intercept"], 6)}, '
            f'R={_number(statistic["correlation"])}'
            for band, statistic in zip(bands, statistics)]


def _summary_caption(bands, statistics, density):
    parts = [f'b{band} {_rmse(statistic, "before", 3)}->'
             f'{_rmse(statistic, "after", 3)}{_improvement(statistic)}'
             for band, statistic in zip(bands, statistics)]
    coverage = ''
    if density is not None and density.total_valid:
        coverage = (f' | {density.total_selected:,} no-change pixels, '
                    f'{density.coverage():.1f}% of the valid overlap')
    return (f'Affine-model RMSE before -> after (before storage conversion): {", ".join(parts)}{coverage}',
            _SUMMARY_TITLE)


def _bands_caption(bands, coefficients, validation, *, ranges=None, fit_ranges=None):
    parts = [f'b{band} slope={_number(slope)} offset={intercept:+.4g}'
             for band, (slope, intercept, _) in zip(bands, coefficients)]
    tested = [f'b{band}: train={item["train_count"]:,}, test={item["count"]:,}, '
              f'RMSE {_number(item["rmse"], 3)}, bias {_number(item["bias"], 3)}, '
              f'variance ratio {_number(item["variance_ratio"], 3)}'
              if item else f'b{band}: unavailable (small or degenerate split)'
              for band, item in zip(bands, validation)]
    holdout_text = f' | sample-refit hold-out (descriptive): {"; ".join(tested)}' if tested else ''
    text = (f'Applied correction: {", ".join(parts)}. Distributions use the same sampled pixels; '
            'target after is the full-fit affine model before clipping, masking and storage conversion. '
            'All curves within a view share bins; detail insets re-bin reference/after only.'
            f'{holdout_text}. {_VALIDATION_SCOPE}')
    if ranges is not None or fit_ranges is not None:
        def span(source, index):
            bounds = source.bounds(index) if source is not None else None
            return (f'[{_number(bounds[0])}, {_number(bounds[1])}]'
                    if bounds is not None else 'unavailable')

        complete = [f'b{band}: fitting pixels {span(fit_ranges, index)}, '
                    f'application pixels {span(ranges, index)}' for index, band in enumerate(bands)]
        text += (' Scatter axes show the sampled no-change pixels. Complete target-value ranges '
                 '(not limited to the plotting sample): ' + '; '.join(complete) + '.')
    return text, 'Radiometric calibration report - per band'


def _caption(title: str):
    """Heading plus the embedded figure; the log shows the report, not its path."""
    # A literal token, not str.format: rendered captions may contain braces.
    return ((f'<b>{escape(title)}</b><br/>'
             f'<img src="data:image/png;base64,{_TOKEN}" width="{WIDTH}"/>'),
            title)


def _png_metadata(context, description):
    """Retain detailed diagnostics in the file without flooding the dialog."""
    return {'Description': '\n'.join(line for line in (*_context_lines(context), description) if line)}


def _headroom(*axes, fraction=0.3):
    """Free space above the plotted data so a legend or note cannot cover it."""
    for axis in axes:
        low, high = axis.get_ylim()
        if not high > low:
            continue
        if axis.get_yscale() == 'log':
            if low > 0:
                axis.set_ylim(low, high * (high / low) ** fraction)
        else:
            axis.set_ylim(low, high + fraction * (high - low))


def _agreement_number(value, digits=3):
    """Readable plot values; retain scientific notation only at extreme scales."""
    if not np.isfinite(value) or value < 0:
        return 'n/a'
    if value == 0:
        return '0'
    if value < .001 or value >= 1e6:
        return f'{value:.{digits}g}'
    decimals = max(0, digits - 1 - int(np.floor(np.log10(value))))
    text = f'{value:,.{decimals}f}'
    return text.rstrip('0').rstrip('.') if decimals else text


def _agreement_change(statistic):
    """Concise plot-only fold change and deterioration flag; no inference from bounds."""
    before, after = statistic['rmse_before'], statistic['rmse_after']
    if (not (np.isfinite(before) and np.isfinite(after)) or min(before, after) < 0
            or any(np.isfinite(statistic.get(f'rmse_{when}_bound', np.nan))
                   for when in ('before', 'after'))):
        return '', False
    if before == after:
        return 'Unchanged RMSE', False
    worse = after > before
    direction = 'higher' if worse else 'lower'
    smaller, larger = min(before, after), max(before, after)
    if smaller == 0:
        return f'{direction.capitalize()} RMSE', worse
    with np.errstate(over='ignore'):
        ratio = larger / smaller
    if not np.isfinite(ratio):
        return f'{direction.capitalize()} RMSE', worse
    rounded = float(f'{ratio:.3g}')
    if rounded == 1:
        return f'Slightly {direction} RMSE', worse
    return f'{_agreement_number(rounded)}× {direction} RMSE', worse


def _panel_heading(axis, title, subtitle=None):
    """Consistent title/subtitle hierarchy across the summary panels."""
    axis.set_title(title, fontsize=9, pad=18 if subtitle else 6)
    if subtitle:
        axis.text(.5, 1.02, subtitle, transform=axis.transAxes,
                  ha='center', va='bottom', fontsize=8, color='0.35')


def _agreement(axis, labels, statistics):
    """Before and after RMSE per band, one segment per band on a log axis.

    Grouped bars on a linear axis hide the very result they exist to show: a
    calibration that works leaves a residual one to three orders of magnitude
    below the original difference, so the "after" bar is a sub-pixel sliver
    lying on the axis line. On a log axis both markers stay legible at any
    scale, and the segment joining them is the fold change itself.
    """
    from matplotlib.ticker import FuncFormatter, NullFormatter

    _panel_heading(axis, 'Agreement with the reference', 'Selected no-change pixels')
    values = np.array([[item[f'rmse_{when}'] for when in ('before', 'after')]
                       for item in statistics], float)
    bounds = np.array([[item.get(f'rmse_{when}_bound', np.nan)
                       for when in ('before', 'after')] for item in statistics], float)
    values = np.where(np.isfinite(bounds), bounds, values)
    usable = values.ravel()
    usable = usable[np.isfinite(usable) & (usable > 0)]
    axis.set_xscale('log')
    if usable.size:
        low, high = np.log(usable.min()), np.log(usable.max())
        # Reserve room outside the extreme markers for their value labels, even
        # when the bands span many orders of magnitude.
        padding = max(np.log(3), .25 * (high - low))
        axis.set_xlim(np.exp(low - padding), np.exp(high + padding))
    else:
        axis.set_xlim(1, 10)
        axis.set_xticks([])  # zeros / unavailable values have no log-axis position
    positions = np.arange(len(labels))[::-1]
    for i, (when, color) in enumerate((('before', '#cc6677'), ('after', '#4477aa'))):
        shown = np.isfinite(values[:, i]) & (values[:, i] > 0)
        bounded = np.isfinite(bounds[:, i])
        style = ({'facecolors': 'none', 'edgecolors': color, 'linewidths': 1.1}
                 if i == 0 else {'color': color})
        axis.scatter(values[shown & ~bounded, i], positions[shown & ~bounded],
                     s=44 if i == 0 else 24, zorder=3 + i,
                     label=f'{when.capitalize()} normalization', **style)
        axis.scatter(values[shown & bounded, i], positions[shown & bounded],
                     s=48 if i == 0 else 34, marker='<', zorder=3 + i, **style)
    for row, (left, right), statistic in zip(positions, values, statistics):
        paired = np.isfinite(left) and np.isfinite(right) and left > 0 and right > 0
        if paired:
            axis.plot([left, right], [row, row], color='#999999', linewidth=1.0, zorder=1)
        numbers = []
        before, after = statistic['rmse_before'], statistic['rmse_after']
        digits = 3
        if np.isfinite(before) and np.isfinite(after) and before != after:
            while (digits < 6
                   and _agreement_number(before, digits) == _agreement_number(after, digits)):
                digits += 1
        for when in ('before', 'after'):
            bound = statistic.get(f'rmse_{when}_bound', np.nan)
            value = statistic[f'rmse_{when}']
            numbers.append(f'≤ {_agreement_number(bound)}' if np.isfinite(bound)
                           else '0 (computed)' if value == 0 else _agreement_number(value, digits))
        for i, (value, number, color) in enumerate(
                zip((left, right), numbers, ('#cc6677', '#4477aa'))):
            if np.isfinite(value) and value > 0:
                # Labels face outwards from the segment, so coincident or very
                # close points still have two readable, correctly attached values.
                on_left = (i == (0 if left <= right else 1)) if paired else i == 0
                axis.annotate(number, (value, row), fontsize=7,
                              textcoords='offset points', xytext=(-6 if on_left else 6, 0),
                              ha='right' if on_left else 'left', va='center', color=color)
            else:
                # A zero or unavailable result cannot be attached to a log-axis
                # marker. Identify it explicitly without inventing a position.
                when = 'Before' if i == 0 else 'After'
                axis.annotate(f'{when}: {number}', (.02 if i == 0 else .98, row),
                              xycoords=('axes fraction', 'data'), fontsize=7,
                              textcoords='offset points', xytext=(0, 3.5),
                              ha='left' if i == 0 else 'right', va='bottom', color=color)
        change, worse = _agreement_change(statistic)
        if paired and change:
            # The geometric mean is the visual midpoint on a logarithmic axis.
            middle = np.exp((np.log(left) + np.log(right)) / 2)
            axis.annotate(f'({change})', (middle, row), fontsize=6.5,
                          textcoords='offset points', xytext=(0, 4), ha='center', va='bottom',
                          color='#a65e00' if worse else '0.25')
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _agreement_number(value)))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=7)
    axis.set_ylim(positions.min() - 0.7, positions.max() + 1.8)
    axis.set_xlabel('RMSE (reference-image units, log scale) - Lower is better'
                    + ('\n< marker: estimated round-off bound' if np.isfinite(bounds).any() else ''),
                    fontsize=8)
    axis.grid(True, axis='x', alpha=0.3)
    axis.tick_params(labelsize=7)
    axis.legend(fontsize=7, loc='upper center', ncol=2, framealpha=0.9)


def _context_lines(context):
    context = context or {}
    inputs = ' | '.join(f'{label}: {Path(context[key]).name}'
                        for key, label in (('reference', 'ref'), ('target', 'target'), ('output', 'output'))
                        if context.get(key))
    settings = []
    if 'threshold' in context:
        settings.append(f'no-change probability > {_number(context["threshold"])}')
    if 'dtype' in context:
        settings.append(f'output {context["dtype"]}')
    if 'sample_count' in context:
        settings.append(f'sample n={context["sample_count"]:,}')
    return inputs, ' | '.join(settings)


def _adjustments(clipped, negative, valid):
    """What storage conversion changed, out of how many written values.

    Empty when nothing was adjusted: an unconditional "0 clipped" is noise. The
    valid count is the denominator that makes the rest interpretable — a
    thousand clipped values means something different in a scene of 2,000 than
    in one of 20 million.
    """
    changes = []
    if clipped:
        changes.append(f'{clipped:,} clipped')
    if negative:
        changes.append(f'{negative:,} negative set to NoData')
    if not changes:
        return ''
    # The denominator qualifies the whole list: attaching it to one entry would
    # read as a count of that kind of value ("3 of 100 negative values").
    return '; '.join(changes) + (f' (of {valid:,} values).' if valid else '.')


def _footer(figure, context, note=None):
    """Keep image footers interpretive; full run settings remain in the log."""
    context = context or {}
    inputs = ' | '.join(f'{label}: {Path(context[key]).name}'
                        for key, label in (('reference', 'Reference'), ('target', 'Target'))
                        if context.get(key))
    lines = [inputs, '"After" values are calculated before output rounding, clipping and masking.']
    conversion = context.get('conversion', [])
    changes = _adjustments(sum(item['clipped'] for item in conversion),
                           sum(item['negative_nodata'] for item in conversion),
                           sum(item['valid'] for item in conversion))
    if changes:
        lines.append('Output adjustments: ' + changes)
    if note:
        lines.append(note)
    # Anchor in physical inches so a tall multiband page cannot push the footer
    # up into the last row's labels as a figure-height percentage would.
    figure.text(.02, .08 / figure.get_figheight(), '\n'.join(line for line in lines if line),
                fontsize=7, va='bottom',
                wrap=True)


def _summary(figure, bands, statistics, density, rhos, context=None):
    """Fixed 2x2 page: did it work, how much was changed, where, and convergence."""
    axes = figure.subplots(2, 2)
    index = np.arange(len(bands))
    labels = [f'Band {band}' for band in bands]

    _agreement(axes[0][0], labels, statistics)

    correction = axes[0][1]
    ratios = [statistic['variance_ratio'] for statistic in statistics]
    correction.axhline(1.0, color='0.6', linestyle='--', linewidth=0.8)
    correction.plot(index, [statistic['slope'] for statistic in statistics],
                    'o-', color='#4477aa', label='Slope')
    correction.plot(index, ratios, 's--', color='#117733', label='Variance ratio (after/ref.)')
    correction.set_ylabel('Slope / variance ratio', fontsize=8)
    offsets = correction.twinx()
    offsets.axhline(0.0, color='0.8', linestyle=':', linewidth=0.8)
    offsets.plot(index, [statistic['intercept'] for statistic in statistics],
                 '^:', color='crimson', label='Offset')
    offsets.set_ylabel('Offset (reference-image units)', fontsize=8, color='crimson')
    offsets.tick_params(labelsize=7, colors='crimson')
    handles = correction.get_legend_handles_labels()
    extra = offsets.get_legend_handles_labels()
    # Reserve the headroom and place the legend explicitly: matplotlib's "best"
    # placement cannot see the twin axis's series and covers the offsets.
    _headroom(correction, offsets)
    correction.legend(handles[0] + extra[0], handles[1] + extra[1], fontsize=7,
                      loc='upper center', ncol=3, framealpha=0.85)
    _panel_heading(correction, 'Normalization by band', 'After = offset + slope × before')

    from matplotlib.ticker import MaxNLocator

    coverage = axes[1][0]
    if density is not None and density.total_valid:
        counts = np.where(density.valid > 0, density.selected, np.nan)
        # Preserve the map's proportions and center it beneath the agreement
        # panel. Its visible edges need not match the wider plot above.
        image = coverage.imshow(counts, cmap='magma', interpolation='nearest', aspect='equal',
                                vmin=0, vmax=max(1, float(np.nanmax(counts))))
        coverage.set_anchor('C')
        # Place the colour bar separately so it cannot shift the map's centre.
        colorbar_axes = coverage.inset_axes([1.015, 0, .025, 1])
        bar = figure.colorbar(image, cax=colorbar_axes)
        # Pixel counts are whole numbers; the default locator labels them 0.25,
        # 0.75, ... on a sparse scene, which reads as a fraction of a pixel.
        bar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        bar.ax.tick_params(labelsize=7, pad=2, length=2)
        bar.set_label('Selected pixels / cell', fontsize=7, labelpad=3)
        _panel_heading(coverage, 'No-change pixels',
                       f'{density.total_selected:,} selected · {density.coverage():.1f}% of valid overlap')
        coverage.set_xlabel(f'Cell size: up to {density.step} × {density.step} image pixels', fontsize=8)
        coverage.set_yticks([])
        coverage.set_xticks([])
    else:
        _panel_heading(coverage, 'No-change pixels')
        coverage.text(0.5, 0.5, 'Spatial coverage unavailable',
                      ha='center', va='center', fontsize=9)
        coverage.set_axis_off()

    convergence = axes[1][1]
    if rhos is not None and len(rhos):
        rhos = np.atleast_2d(np.asarray(rhos, dtype=np.float64))
        iterations = np.arange(1, len(rhos) + 1)
        for column in range(rhos.shape[1]):
            convergence.plot(iterations, rhos[:, column], linewidth=1.0,
                             label=f'Canonical pair {column + 1}')
        status = (context or {}).get('imad', {})
        selected = status.get('selected_iteration')
        convergence.set_xlabel('Iteration'
                               + (f' (selected solution: {selected})' if selected is not None else ''),
                               fontsize=8)
        convergence.set_ylabel('Canonical correlation', fontsize=8)
        termination = {'converged': 'converged', 'iteration limit': 'iteration limit reached',
                       'numerical fallback': 'numerical fallback'}.get(status.get('termination'))
        title = f'IR-MAD: {termination}' if termination else 'IR-MAD iterations'
        details = [f'{len(rhos)} iteration' + ('s' if len(rhos) != 1 else '')]
        if status.get('final_delta') is not None:
            details.append(f'Max change: {_number(status["final_delta"], 3)}')
        if status.get('delta_threshold') is not None:
            details.append(f'Threshold: {_number(status["delta_threshold"], 3)}')
        _panel_heading(convergence, title, ' · '.join(details))
        if selected is not None:
            convergence.axvline(selected, color='0.5', linestyle=':', linewidth=.8)
        convergence.xaxis.set_major_locator(MaxNLocator(integer=True))
        if rhos.shape[1] <= 12:
            convergence.legend(fontsize=6, ncol=2 if rhos.shape[1] > 4 else 1)
        else:
            convergence.text(.02, .03, f'{rhos.shape[1]} canonical pairs',
                             transform=convergence.transAxes, fontsize=7)
    else:
        _panel_heading(convergence, 'IR-MAD convergence')
        convergence.text(0.5, 0.5, 'Iteration history unavailable',
                         ha='center', va='center', fontsize=9)
        convergence.set_axis_off()

    # The accuracy panel carries the band names on its own axis and is laid out
    # by _agreement; only the remaining panels share the band index.
    correction.set_xticks(index)
    correction.set_xticklabels(labels, fontsize=7)
    for axis in (correction, convergence):
        axis.tick_params(labelsize=7)
        axis.grid(True, alpha=0.25)
    figure.suptitle(_SUMMARY_TITLE, fontsize=11)
    _footer(figure, context)
    # The external colour bar follows the aspect-adjusted map box. Settle that
    # geometry first, then measure padding against its final position.
    for _ in range(2):
        figure.tight_layout(rect=(0.0, 0.085, 1.0, 0.96), w_pad=_COLUMN_PAD)


def _distributions(axis, target, reference, normalized):
    """Shared-bin counts on the same pixels, with optional reference/after detail.

    The overview retains all values, including tails. A detail inset is placed
    only in a sufficiently wide empty gap between raw-target and reference/after
    ranges. Its shared local bins recover shape lost to the overview's much wider
    bins without covering any distribution. Counts are comparable within a view;
    inset and overview bin widths differ and their heights must not be compared.
    """
    from matplotlib.ticker import MaxNLocator

    # Bounded square-root bin count: sparse invariant samples should not be split
    # into dozens of mostly empty bins; large samples stay legible at report size.
    count = min(60, max(12, int(np.ceil(np.sqrt(len(reference))))))
    combined = np.concatenate((reference, target, normalized))
    edges = np.histogram_bin_edges(combined, bins=count)
    styles = (('Reference', '0.25', '-', 1.5),
              ('Target before', '#cc6677', '--', 1.2),
              ('Target after (affine)', '#4477aa', (0, (4, 2)), 1.3))

    def draw(view, series, bins):
        for values, (label, color, linestyle, width) in series:
            view.hist(values, bins=bins, histtype='step', label=label,
                      color=color, linestyle=linestyle, linewidth=width)
        view.set_ylim(bottom=0)
        view.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        view.tick_params(labelsize=7)
        view.grid(True, alpha=.25)

    draw(axis, zip((reference, target, normalized), styles), edges)
    margin = .03 * (edges[-1] - edges[0])
    axis.set_xlim(edges[0] - margin, edges[-1] + margin)
    _headroom(axis, fraction=.3)
    axis.legend(fontsize=7, loc='upper center', ncol=3, framealpha=.9,
                handlelength=2, columnspacing=1)
    axis.set_xlabel('Value', fontsize=8)
    axis.set_ylabel('Sampled pixels / bin', fontsize=8)

    focus = np.concatenate((reference, normalized))
    low, high = float(focus.min()), float(focus.max())
    if target.min() > high:
        start, stop = high, float(target.min())
    elif target.max() < low:
        start, stop = float(target.max()), low
    else:
        return
    left, right = axis.get_xlim()
    gap = (stop - start) / (right - left)
    if gap < .42:
        return
    # Leave at least a bin of room on either side of the empty gap: a step
    # histogram extends past the actual sample extrema to its bin edges.
    width = min(.60, gap - 2 * (1 / count + .04))
    if width < .34:
        return
    center = ((start + stop) / 2 - left) / (right - left)
    detail = axis.inset_axes([center - width / 2, .15, width, .53])
    draw(detail, ((reference, styles[0]), (normalized, styles[2])),
         np.histogram_bin_edges(focus, bins=count))
    detail.set_title('Reference / after detail', fontsize=6)
    detail.xaxis.set_major_locator(MaxNLocator(nbins=3))
    detail.tick_params(labelsize=6)
    for spine in detail.spines.values():
        spine.set_color('0.6')


def _fit_density_legend(figure, axis, bins):
    """Explain occupied-hexagon counts without covering the fitted data."""
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter

    counts = bins.get_array()
    low, high = int(counts.min()), int(counts.max())
    if low == high:
        # No colour variation exists: a swatch is clearer than an artificial
        # continuous scale (especially a scale around a count of one).
        handles, _ = axis.get_legend_handles_labels()
        handles.append(Patch(facecolor=bins.to_rgba(low), edgecolor='none',
                             label=f'{low:,} sampled {"pair" if low == 1 else "pairs"} / bin'))
        axis.legend(handles=handles, fontsize=7, loc='upper left')
        return
    ticks = np.unique(np.rint(np.geomspace(low, high, 4)).astype(int))
    colorbar_axes = axis.inset_axes([1.02, 0, .03, 1])
    bar = figure.colorbar(bins, cax=colorbar_axes, ticks=ticks,
                          format=FuncFormatter(lambda value, _: f'{value:,.0f}'))
    bar.minorticks_off()
    bar.ax.tick_params(labelsize=6, pad=2, length=2)
    bar.set_label('Sampled pairs / bin (log)', fontsize=7, labelpad=3)
    axis.legend(fontsize=7, loc='upper left')


def _bands(figure, bands, coefficients, sample, context=None):
    """Sample-focused fits and distributions.

    The complete fit and application ranges are deliberately not plotted here:
    they can be orders of magnitude wider than the invariant sample and would
    compress it to a few pixels. `_bands_caption` reports them instead.
    """
    axes = figure.subplots(len(bands), 2, squeeze=False)
    validation = []
    target, reference = sample.pairs()
    for row, band in enumerate(bands):
        x = target[:, row]
        y = reference[:, row]
        slope, intercept, correlation = coefficients[row]
        normalized = intercept + slope * x
        validation.append(holdout(x, y))

        scatter = axes[row][0]
        bins = scatter.hexbin(x, y, gridsize=42, bins='log', cmap='viridis', mincnt=1,
                              linewidths=0)
        # The full application range can be orders of magnitude wider than the
        # invariant sample. Show that range in the caption, not in the fit axes.
        low, high = float(x.min()), float(x.max())
        line = np.array([low, high])
        scatter.plot(line, intercept + slope * line, color='crimson', linewidth=1.2,
                     label='affine fit')
        margin = .05 * (high - low or max(abs(low) * .01, 1.))
        scatter.set_xlim(low - margin, high + margin)
        pad = 0.05 * (y.max() - y.min() or 1.0)
        scatter.set_ylim(y.min() - pad, y.max() + pad)
        scatter.set_title(f'Band {band} | slope={_number(slope)} | offset={intercept:+.4g} '
                          f'| R={correlation:.4f}', fontsize=9)
        scatter.set_xlabel('Raw target (sampled no-change pixels)', fontsize=8)
        scatter.set_ylabel('Reference', fontsize=8)
        _fit_density_legend(figure, scatter, bins)

        distribution = axes[row][1]
        _distributions(distribution, x, y, normalized)
        distribution.set_title(f'Band {band} | value distributions', fontsize=9)

        for axis in (scatter, distribution):
            axis.tick_params(labelsize=7)
            axis.grid(True, alpha=0.25)

    figure.suptitle(f'Radiometric calibration report - per band '
                    f'({len(sample):,} uniformly sampled no-change pixels)', fontsize=11)
    has_insets = any(axis.child_axes for axis in axes[:, 1])
    _footer(figure, context,
            'Insets use finer bins; compare histogram heights within each view.' if has_insets else None)
    # Settle the external colour bars against the final plot positions.
    for _ in range(2):
        figure.tight_layout(rect=(0.0, 0.75 / figure.get_figheight(),
                                  1.0, 1.0 - 0.35 / figure.get_figheight()), w_pad=_COLUMN_PAD)
    return validation


def _staged(destination, staging):
    """Where a figure is written before publication moves it to `destination`."""
    if staging is None:
        return destination
    return os.path.join(staging, os.path.basename(destination))


def build(path, bands, coefficients, statistics, sample=None, density=None,
          ranges=None, convergence=None, staging=None, *, fit_ranges=None, context=None,
          protected_paths=()):
    """Render the report figures as (written, destination, html, text) items.

    With a `staging` directory the figures are written there and their captions
    still link to `path`, so a caller that publishes complete outputs only can
    move them after its own commit point without breaking the log's link.
    Layout inches and display width are independent of rendering DPI: increasing
    DPI adds image detail instead of shrinking the figure or enlarging its embed.
    """
    stem, extension = os.path.splitext(path)
    per_band = f'{stem}_bands{extension}'
    for destination in (path, per_band):
        validate_report_destination(destination, protected_paths)
        validate_report_destination(_staged(destination, staging), protected_paths)
    context = dict(context or {}, sample_count=len(sample) if sample is not None else 0)
    # Use an Agg canvas directly; never change another plugin's pyplot backend.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(WIDTH / _LAYOUT_DPI, 7.4), dpi=_DPI)
    FigureCanvasAgg(figure)
    _summary(figure, bands, statistics, density, convergence, context)
    description, title = _summary_caption(bands, statistics, density)
    details = '\n'.join([description, *_accuracy_lines(bands, statistics)])
    figure.savefig(_staged(path, staging), dpi=_DPI, metadata=_png_metadata(context, details))
    items = [(_staged(path, staging), path, *_caption(title))]

    if sample is None or len(sample) < 16:
        return items
    height = 2.9 * len(bands) + 1.1
    # Bound the raster allocation even during layout, before savefig runs.
    dpi = min(_DPI, _MAX_PIXEL_HEIGHT / height)
    figure = Figure(figsize=(WIDTH / _LAYOUT_DPI, height), dpi=dpi)
    FigureCanvasAgg(figure)
    validation = _bands(figure, bands, coefficients, sample, context)
    description, title = _bands_caption(bands, coefficients, validation,
                                        ranges=ranges, fit_ranges=fit_ranges)
    figure.savefig(_staged(per_band, staging), dpi=dpi, metadata=_png_metadata(context, description))
    items.append((_staged(per_band, staging), per_band, *_caption(title)))
    return items


def push(feedback, item):
    """Embed one rendered figure in the feedback log when the API supports it."""
    written, _destination, template, text = item
    if feedback is None:
        return
    pusher = getattr(feedback, 'pushFormattedMessage', None)
    if pusher is None:
        feedback.pushInfo(text)
        return
    try:
        with open(written, 'rb') as handle:
            encoded = base64.b64encode(handle.read()).decode('ascii')
    except OSError:
        encoded = None
    if encoded is None:
        feedback.pushInfo(text)
        return
    pusher(template.replace(_TOKEN, encoded), text)


def _warn(feedback, info, message):
    reporter = getattr(feedback, 'reportError', None)
    if reporter is None:
        info(f'WARNING: {message}')
    else:
        reporter(message, False)


def emit(feedback, info, bands, coefficients, fits, *, path=None, staging=None,
         sample=None, density=None, ranges=None, convergence=None, written=None,
         fit_ranges=None, context=None, protected_paths=()):
    """Log affine-model agreement and, when enabled, the embedded figures.

    Paths actually written are appended to `written` so a caller that stages
    them can publish them. A diagnostic must never fail a run whose calibrated
    raster is already written, so every failure here is downgraded to a warning.
    """
    try:
        statistics = [fit.statistics() for fit in fits]
        dtype = (context or {}).get('dtype')
        info('Calibration results' + (f' (output {dtype})' if dtype else '')
             + ': RMSE before -> after, before rounding/clipping/masking.')
        for line in _accuracy_lines(bands, statistics):
            info(line)
        for band, effects in zip(bands, (context or {}).get('conversion', [])):
            changes = _adjustments(effects['clipped'], effects['negative_nodata'], effects['valid'])
            if changes:
                info(f'band {band}: output adjustments: {changes}')
    except Exception as exc:                       # noqa: BLE001 - diagnostics only
        _warn(feedback, info, f'Calibration accuracy table unavailable: {exc}')
        return
    if path is None:
        return
    try:
        items = build(path, bands, coefficients, statistics, sample, density, ranges,
                      convergence, staging, fit_ranges=fit_ranges, context=context,
                      protected_paths=protected_paths)
    except Exception as exc:                       # noqa: BLE001 - diagnostics only
        _warn(feedback, info, f'Calibration report figures unavailable: {exc}')
        return
    for item in items:
        if written is not None:
            written.append(item[0])
        try:
            if feedback is None:
                info(item[3])
            else:
                push(feedback, item)
        except Exception as exc:                   # noqa: BLE001 - diagnostics only
            _warn(feedback, info, f'Cannot embed the calibration report: {exc}')
            info(item[3])
