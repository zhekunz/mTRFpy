import math
import warnings
import numpy as np
import pytest
from mtrf import stats
from mtrf.model import TRF, load_sample_data
from mtrf.stats import crossval, nested_crossval, permutation_distribution

n = np.random.randint(3, 5)
stimulus, response, fs = load_sample_data(n_segments=n)


def _cuda_available():
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return torch.cuda.is_available()


def test_nested_crossval():
    tmin, tmax = np.random.uniform(-0.1, 0.05), np.random.uniform(0.1, 0.4)
    reg = np.random.uniform(0, 10)
    trf = TRF()
    splits = np.random.randint(2, 5)
    metric, best_regularization = nested_crossval(
        trf, stimulus, response, fs, tmin, tmax, reg, splits
    )


def test_pearsonr():
    vec1 = np.random.uniform(0, 1000, 100)
    vec2 = np.random.uniform(0, 1000, 100)
    r1 = np.corrcoef(vec1, vec2)[0, 1]
    r2 = stats.pearsonr(vec1, vec2)
    assert np.allclose(r1, r2)


def test_crossval():
    for direction in [1, -1]:
        tmin, tmax = np.random.uniform(-0.1, 0.05), np.random.uniform(0.1, 0.4)
        reg = np.random.uniform(0, 10)
        trf = TRF(direction=direction)
        splits = np.random.randint(2, n + 1)
        metric = crossval(trf, stimulus, response, fs, tmin, tmax, reg, splits)
        assert np.isscalar(metric)
        metric = crossval(
            trf, stimulus, response, fs, tmin, tmax, reg, splits, average=False
        )
        if direction == 1:
            assert len(metric) == response[0].shape[-1]
        else:
            assert len(metric) == stimulus[0].shape[-1]


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("preload", [True, False])
def test_crossval_torch_cpu_matches_numpy(direction, preload):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    stimulus_np = [rng.normal(size=(30, 2)) for _ in range(4)]
    response_np = [rng.normal(size=(30, 3)) for _ in range(4)]
    kwargs = dict(
        fs=10,
        tmin=0,
        tmax=0.2,
        regularization=1.0,
        k=2,
        seed=42,
        verbose=False,
    )

    expected = crossval(
        TRF(direction=direction, preload=preload),
        stimulus_np,
        response_np,
        **kwargs,
    )
    stimulus_torch = [torch.asarray(x) for x in stimulus_np]
    response_torch = [torch.asarray(y) for y in response_np]
    actual = crossval(
        TRF(direction=direction, preload=preload),
        stimulus_torch,
        response_torch,
        **kwargs,
    )

    assert isinstance(actual, torch.Tensor)
    assert actual.device.type == "cpu"
    np.testing.assert_allclose(actual.item(), expected, rtol=1e-7, atol=1e-8)


@pytest.mark.skipif(
    not _cuda_available(),
    reason="CUDA GPU is unavailable",
)
def test_crossval_torch_cuda_matches_numpy():
    import torch

    rng = np.random.default_rng(1)
    stimulus_np = [rng.normal(size=(40, 2)) for _ in range(4)]
    response_np = [rng.normal(size=(40, 3)) for _ in range(4)]
    kwargs = dict(
        fs=10,
        tmin=0,
        tmax=0.2,
        regularization=1.0,
        k=2,
        seed=42,
        verbose=False,
    )

    expected = crossval(TRF(), stimulus_np, response_np, **kwargs)
    stimulus_cuda = [torch.asarray(x, device="cuda") for x in stimulus_np]
    response_cuda = [torch.asarray(y, device="cuda") for y in response_np]
    actual = crossval(TRF(), stimulus_cuda, response_cuda, **kwargs)

    assert actual.is_cuda
    np.testing.assert_allclose(
        actual.detach().cpu().item(), expected, rtol=1e-6, atol=1e-7
    )


def test_permutation():
    tmin, tmax = np.random.uniform(-0.1, 0.05), np.random.uniform(0.1, 0.4)
    n_permute = np.random.randint(2, 5)
    reg = np.random.uniform(0, 10)
    trf = TRF()
    metric = permutation_distribution(
        trf,
        stimulus,
        response,
        fs,
        tmin,
        tmax,
        reg,
        n_permute,
        k=-1,
    )
    assert len(metric) == n_permute


def test_permutation_average_false():
    tmin, tmax = np.random.uniform(-0.1, 0.05), np.random.uniform(0.1, 0.4)
    n_permute = np.random.randint(5, 10)
    reg = np.random.uniform(0, 10)
    trf = TRF()
    n_channels = response[0].shape[-1]
    metric = permutation_distribution(
        trf, stimulus, response, fs, tmin, tmax, reg, n_permute, average=False
    )
    assert metric.shape == (n_permute, n_channels)


def test_permutation_too_many():
    trf = TRF()
    n_permute = math.factorial(n) + 1
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        permutation_distribution(
            trf,
            stimulus,
            response,
            fs,
            tmin=0,
            tmax=0.1,
            regularization=1,
            n_permute=n_permute,
        )
    assert len(w) == 1
    assert "will be repeated" in str(w[0].message)
