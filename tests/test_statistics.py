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
