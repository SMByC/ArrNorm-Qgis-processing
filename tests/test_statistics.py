"""Covariance and orthogonal-regression statistics checked against NumPy/SVD."""
import numpy as np
import pytest
from ArrNorm.core.auxil.auxil import Cpm, OrthogonalFit, orthoregress


@pytest.mark.parametrize('block', [1, 17, 1024])
def test_streaming_regression_matches_svd_oracle(block):
    rng = np.random.default_rng(5)
    x = rng.normal(size=1000) + 100000
    y = 2 * x + 3 + rng.normal(0, .1, size=len(x))
    centered = np.column_stack((x - x.mean(), y - y.mean()))
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    slope = axes[0, 1] / axes[0, 0]
    fit = OrthogonalFit()
    for start in range(0, len(x), block):
        fit.update(x[start:start + block], y[start:start + block])
    actual_slope, intercept, corr = fit.coefficients()
    assert actual_slope == pytest.approx(slope, rel=1e-9)
    assert intercept == pytest.approx(y.mean() - slope * x.mean(), abs=1e-4)
    assert corr == pytest.approx(np.corrcoef(x, y)[0, 1], rel=1e-9)
    assert not any(isinstance(value, (list, tuple)) for value in vars(fit).values())


@pytest.mark.parametrize('block', [1, 97, 5000])
def test_moment_statistics_match_a_brute_force_oracle(block):
    # The report derives before/after accuracy from the accumulated moments
    # instead of a second pass, so the closed form must reproduce the residuals
    # computed directly from every pair.
    rng = np.random.default_rng(11)
    x = rng.normal(50, 12, size=5000)
    y = 1.4 * x + 9 + rng.normal(0, 3, size=len(x))
    fit = OrthogonalFit()
    for start in range(0, len(x), block):
        fit.update(x[start:start + block], y[start:start + block])
    statistics = fit.statistics()

    normalized = statistics['intercept'] + statistics['slope'] * x
    assert statistics['count'] == len(x)
    assert statistics['rmse_after'] == pytest.approx(
        np.sqrt(np.mean((y - normalized) ** 2)), rel=1e-9)
    assert statistics['rmse_before'] == pytest.approx(
        np.sqrt(np.mean((y - x) ** 2)), rel=1e-9)
    assert statistics['mean_difference'] == pytest.approx(y.mean() - x.mean(), rel=1e-9)
    assert statistics['variance_ratio'] == pytest.approx(
        np.var(normalized, ddof=1) / np.var(y, ddof=1), rel=1e-9)
    # Zero by construction: the fitted line passes through the centroid, which
    # is why the report withholds a subset instead of reporting an in-sample bias.
    assert np.mean(y - normalized) == pytest.approx(0.0, abs=1e-9 * np.abs(y).mean())


@pytest.mark.parametrize('count', [0, 1])
def test_regression_primitives_reject_too_few_points(count):
    with pytest.raises(ValueError, match='two finite'):
        orthoregress(np.ones(count), np.ones(count))


def test_covariance_matches_numpy_across_blocks():
    data = np.random.default_rng(7).normal(size=(200, 4))
    acc = Cpm(4)
    for block in np.array_split(data, 13):
        acc.update(block)
    np.testing.assert_allclose(acc.means(), data.mean(axis=0), atol=1e-9)
    np.testing.assert_allclose(acc.covariance(), np.cov(data, rowvar=False), atol=1e-8)


@pytest.mark.parametrize('block', [1, 97, 5000])
def test_near_affine_rmse_is_bounded_instead_of_claiming_exact_zero(block):
    rng = np.random.default_rng(31)
    x = rng.uniform(1000, 10000, 5000).astype(np.float32).astype(float)
    y = (1.2 * x + 3).astype(np.float32).astype(float)
    fit = OrthogonalFit()
    for start in range(0, len(x), block):
        fit.update(x[start:start + block], y[start:start + block])
    slope, intercept, _ = fit.coefficients()
    actual = np.sqrt(np.mean((y - (intercept + slope * x)) ** 2))
    result = fit.statistics()
    assert actual > 0
    assert np.isnan(result['rmse_after'])
    assert result['rmse_after_bound'] >= actual


def test_moment_resolution_accounts_for_large_centroid_offsets():
    rng = np.random.default_rng(7)
    x = 1e15 + rng.uniform(0, 1e4, 5000)
    y = 1.2 * x + 3
    fit = OrthogonalFit()
    fit.update(x, y)
    slope, intercept, _ = fit.coefficients()
    actual = np.sqrt(np.mean((intercept + slope * x - y) ** 2))
    result = fit.statistics()
    assert actual > 0
    assert np.isnan(result['rmse_after'])
    assert result['rmse_after_bound'] >= actual
